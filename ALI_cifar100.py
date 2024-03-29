import parser
import ssl
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import timm
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_ALI
from engine_ALI_cifar100 import train_one_epoch
import models_teacher


def get_args_parser():
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--model', default='vit_tiny_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--output_dir', default='./output_dir_tiny',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_tiny',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--teacher_path",
                        default='',
                        type=str)
    parser.add_argument("--teacher_model", default="vit_base", type=str)
    parser.add_argument('--intermediate', default=10, type=int,
                        help='Distill intermediate layer of teacher')
    return parser


class TwoCropsTransform:
    def __init__(self, common_transform, teacher_transform, student_transform2):
        self.common_transform = common_transform
        self.teacher_transform = teacher_transform
        self.student_transform2 = student_transform2

    def __call__(self, x):
        x = self.common_transform(x)
        imgT = self.teacher_transform(x)
        imgS = self.student_transform2(x)
        return [imgT, imgS]


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    common_transform = transforms.Compose(
        [transforms.RandomResizedCrop((224, 224), scale=(0.2, 1.0), interpolation=3),
         transforms.RandomHorizontalFlip()])
    teacher_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    student_transform = transforms.Compose(
        [transforms.ColorJitter(0.4, 0.4, 0.4),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset_train = datasets.CIFAR100(root='./data', train=True,
                                      download=True, transform=TwoCropsTransform(common_transform, teacher_transform,
                                                                                 student_transform))
    print(dataset_train)
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    print("Sampler_train = %s" % str(sampler_train))
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    model = models_ALI.__dict__[args.model]()
    teacher = models_teacher.__dict__[args.teacher_model]()
    checkpoint = torch.load(args.teacher_path, map_location='cpu')
    print("Load pre-trained checkpoint from: %s" % args.teacher_path)
    checkpoint_model = checkpoint['model']
    for k in ['head.weight', 'head.bias']:
        # if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        #     print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]
    teacher.load_state_dict(checkpoint_model)
    teacher.eval()
    model.to(device)
    teacher.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()  # 单线程返回1
    # if args.lr is None:  # only base_lr is specified
    #     args.lr = args.blr * eff_batch_size / 256
    #
    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    if os.path.exists(args.output_dir):
        ckpt = os.listdir(args.output_dir)
        if len(ckpt) > 0:
            for i in range(800):
                if "checkpoint-{}.pth".format(str(i)) in ckpt:
                    args.resume = "./output_dir/" + "checkpoint-{}.pth".format(str(i))
                    print(args.resume)
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, teacher, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
