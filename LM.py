import torch
import torch.nn as nn
import torch.nn.functional as F

from _base import Distiller

"""
这段代码定义了一个名为dkd_loss的函数，它用于计算基于decoupling的知识蒸馏损失。具体来说，该函数实现的操作为：

1. 调用_get_gt_mask和_get_other_mask函数生成两个掩码gt_mask和other_mask；
2. 分别对logits_student和logits_teacher使用softmax函数求取预测概率分布pred_student和pred_teacher，
并将其按照gt_mask和other_mask划分成两个子集；
3. 对于两个子集内的概率分布，分别计算其log并执行kl散度计算，得到tckd_loss和nckd_loss；
4. 将tckd_loss和nckd_loss加权求和得到最终的知识蒸馏损失。

其中，tckd_loss被称为Teacher-Corrected KD Loss，是模型输出与教师模型输出之间的KL散度，
用于保证学生模型的预测结果更接近于教师模型的预测结果；nckd_loss被称为Negative-Certainty KD Loss，
是一种针对教师模型置信度较高的样本的惩罚机制，用于防止学生模型在这些样本上过于自信而忽略更难分类的样本。
alpha和beta是超参数，用于调整两种损失的权重；temperature则是温度参数，用于调整预测分布的熵。
"""


def LM_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    """
    这段代码是一个 softmax 函数的调用，其中 logits_teacher 是模型预测的概率分布，temperature 是一个温度参数，
    gt_mask 是一个与目标数据相关的掩码，dim=1 表示对输入张量的第二个维度进行 softmax 操作。

    具体地说，logits_teacher / temperature 执行了一个分布缩放操作，将模型输出的原始分数值按照一定的温度因子进行缩放。
    然后，减去一个与目标数据相关的掩码 1000.0 * gt_mask，这个掩码通常是为了限制模型只能预测出目标数据中存在的类别，
    而不会预测出不存在的类别。

    最后，通过 softmax 函数将缩放后的概率分布归一化为一个概率分布，使得所有类别的概率之和为 1，
    以便于进一步计算损失函数并进行优化。
    """
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    """
    这段代码实现了一个nckd_loss（Noisy Collaborative Knowledge Distillation）的损失函数计算。
    其中，该损失函数用于教师-学生模型中，主要目的是让学生模型能够更好地学习到教师模型中的知识。

    具体而言，这段代码使用了KL散度作为损失函数的度量方式。KL散度可以用来衡量两个概率分布之间的距离，
    在这里用来衡量学生模型预测结果与教师模型预测结果之间的差异性。

    代码中，log_pred_student_part2表示学生模型预测结果的对数值，pred_teacher_part2表示教师模型预测结果的值，
    temperature表示温度参数，target表示目标值，即用来计算损失函数的标签值。
    代码中，首先使用F.kl_div函数计算出学生模型预测结果与教师模型预测结果之间的KL散度，然后将其乘以温度的平方，
    再除以标签值的数量，最终得到整个nckd_loss的损失值。
    """
    nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (temperature ** 2)
            / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


"""
这段代码定义了一个名为_get_gt_mask的函数，
它的作用是根据输入的logits和target生成一个掩码mask。具体来说，该函数实现的操作为：

1. 将target转换成一维张量，以便与logits对应；
2. 利用torch.zeros_like(logits)创建一个形状与logits相同、元素均为0的张量，并将其转换为布尔型张量；
3. 通过调用scatter_方法，在mask张量上将target指定位置上的值设为1，即使得mask张量在target索引处为True；
4. 返回处理后的mask张量。

总体来说，该函数的作用是获取给定target标签所对应的类别的掩码(mask)，
以便在训练过程中计算知识蒸馏损失。
"""


def _get_gt_mask(logits, target):
    # 这段代码将数组target重新调整为一维数组。NumPy中的reshape()方法允许你改变数组的形状而不改变其数据。
    # 在这里，参数-1被使用，它代表一个未知的维度大小；数组会根据其长度和其他维度来自动计算该维度的大小。因此，
    # 如果target原来是一个形状为(n, m)的二维数组，那么经过这个操作后，它会变成一个形状为(n*m,)的一维数组。
    target = target.reshape(-1)
    """
    torch.zeros_like(logits)：创建一个与 logits 具有相同大小的全零张量。
    scatter_(1, target.unsqueeze(1), 1)：将值为1的张量散布到 mask 张量的指定位置上。其中，
    第一个参数 1 指定了在哪个维度上进行散布操作，
    第二个参数 target.unsqueeze(1) 是一个列向量，表示需要置为1的位置索引，
    第三个参数 1 表示要置为1的值。
    .bool()：将 mask 张量转换为布尔类型。
    """
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


"""
这段代码定义了一个名为_get_other_mask的函数，
它的作用是根据输入的logits和target生成一个掩码mask。具体来说，该函数实现的操作为：

1. 将target转换成一维张量，以便与logits对应；
2. 利用torch.ones_like(logits)创建一个形状与logits相同、元素均为1的张量，
并将其转换为布尔型张量；
3. 通过调用scatter_方法，在mask张量上将target指定位置上的值设为0，
即使得mask张量在target索引处为False；
4. 返回处理后的mask张量。

总体来说，该函数的作用是获取除给定target标签以外的所有类别所对应的掩码(mask)，
以便在训练过程中计算知识蒸馏损失。
"""


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    """
    torch.ones_like(logits)：创建一个与 logits 张量形状相同的全1张量。
    .scatter_(1, target.unsqueeze(1), 0)：将值为0的张量散布到 mask 张量的指定位置上。
    其中，第一个参数 1 指定了在哪个维度上进行散布操作，第二个参数 target.unsqueeze(1) 是一个列向量，
    表示需要置为0的位置索引，第三个参数 0 表示要置为0的值。
    .bool()：将 mask 张量转换为布尔类型。
    """
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


"""
这段代码定义了一个名为cat_mask的函数，
它的功能是将输入张量t按照给定的两个掩码mask1和mask2进行分组，并将每个分组内的元素相加。具体来说，该函数实现的操作为：

1. 通过将掩码mask1与t相乘，得到一个只保留mask1对应位置上元素的张量t1；
2. 对于t1沿着第1维度求和，即将一个batch中所有样本mask1对应位置上的元素相加得到一个1维张量；
3. 同样地，通过将掩码mask2与t相乘，得到一个只保留mask2对应位置上元素的张量t2；
4. 对于t2沿着第1维度求和，即将一个batch中所有样本mask2对应位置上的元素相加得到一个1维张量；
5. 将t1和t2在第1维度上进行拼接，得到一个新的张量rt。

最后，函数返回新的张量rt。
"""


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


"""
这段代码是一个名为DKD的类，它实现了Decoupled Knowledge Distillation。
这个方法可以用于在深度神经网络中从一个教师模型到一个学生模型的知识传递，
以使得学生模型能够获得更好的性能。

在初始化函数中，该类接受三个参数student、teacher和cfg。
其中，student是需要训练的学生模型，teacher是提供指导的教师模型，cfg是一些超参数的设置。

在forward_train函数中，首先通过调用student对输入图像进行前向计算，
得到学生模型的输出logits_student。接着，使用torch.no_grad()将教师模型的参数固定住，
通过调用teacher对同一个输入图像再次进行前向计算，得到教师模型的输出logits_teacher。

然后，根据论文提出的公式，该方法计算两种损失：交叉熵损失（loss_ce）和
基于decoupling的知识蒸馏损失（loss_dkd）。
其中，loss_ce表示学生模型的输出与真实标签之间的差异；
loss_dkd是通过计算教师模型和学生模型的输出之间的相似性来衡量知识蒸馏的损失。
至于具体的知识蒸馏公式，可以参考代码中的dkd_loss函数实现。

最后，函数返回logits_student和一个包含两种损失的字典losses_dict，用于计算总体损失。
"""


class DKD(Distiller):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, student, teacher, cfg):
        super(DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha = cfg.DKD.ALPHA
        self.beta = cfg.DKD.BETA
        self.temperature = cfg.DKD.T
        self.warmup = cfg.DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
