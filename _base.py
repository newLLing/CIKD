import torch
import torch.nn as nn
import torch.nn.functional as F

"""
这段代码定义了一个名为“Distiller”的类，该类继承自PyTorch中的nn.Module类。它有两个参数“student”和“teacher”，
分别表示学生模型和教师模型。

此外，该类具有以下方法：

- \_\_init\_\_()方法： 初始化Distiller类，并将输入的“student”和“teacher”保存到类属性中。
- train()方法：设置训练模式，同时将教师模型设置为评估模式。
- get_learnable_parameters()方法：获取学生模型中可学习的参数。
- get_extra_parameters()方法：计算由蒸馏器引入的额外参数。
- forward_train()方法：执行蒸馏方法的训练过程。
- forward_test()方法：执行蒸馏方法的测试过程。
- forward()方法： 根据当前是否处于训练模式来选择执行forward_train()或forward_test()方法。
"""


class Distiller(nn.Module):
    """
    这段代码实现了一个Distiller（蒸馏器）类，
    其中该类的实例对象可以用来进行教师-学生模型之间的知识蒸馏（knowledge distillation）。
    在该类的初始化函数中，首先调用了父类的初始化函数，
    然后将传入的student和teacher参数保存在该类的成员变量self.student和self.teacher中。
    其中，student表示学生模型，而teacher表示教师模型。
    """
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.student = student
        self.teacher = teacher
    """
    这段代码实现了Distiller类中的train函数，该函数用于设置该类的训练模式。

    具体而言，在该函数中，首先根据传入的参数mode设置该类的training成员变量，
    以控制是否将该类的实例对象设为训练模式。然后，循环遍历该类的所有子层，并根据training参数设置每一层的训练模式，
    以确保所有子层都处于正确的训练模式。

    此外，由于该类的主要功能是进行教师-学生模型之间的知识蒸馏，因此在训练过程中，
    需要将教师模型设置为评估模式（即self.teacher.eval()），以免影响到蒸馏的效果。

    最后，该函数返回该类的实例对象本身，以便可以使用链式编程调用该函数。
    """
    def train(self, mode=True):
        # teacher as eval mode by default
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.teacher.eval()
        return self

    def get_learnable_parameters(self):
        # if the method introduces extra parameters, re-impl this function
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        # calculate the extra parameters introduced by the distiller
        return 0

    def forward_train(self, **kwargs):
        # training function for the distillation method
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


"""
这段代码定义了一个名为“Vanilla”的类，该类继承自PyTorch中的nn.Module类。它有一个参数“student”，表示学生模型。

此外，该类还具有以下方法：

- \_\_init\_\_()方法：初始化Vanilla类，并将输入的“student”保存到类属性中。
- get_learnable_parameters()方法：获取学生模型中可学习的参数。
- forward_train()方法：执行标准的训练过程，即在学生模型上进行前向传播，计算交叉熵损失并返回结果和损失值。
- forward()方法： 根据当前是否处于训练模式来选择执行forward_train()或forward_test()方法。
- forward_test()方法：执行测试过程，在学生模型上进行前向传播并返回结果。
"""


class Vanilla(nn.Module):
    def __init__(self, student):
        super(Vanilla, self).__init__()
        self.student = student

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = F.cross_entropy(logits_student, target)
        return logits_student, {"ce": loss}

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]
