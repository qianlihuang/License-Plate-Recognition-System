# 导入必要的PyTorch库
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个名为STNet的类，它继承自nn.Module


class STNet(nn.Module):
    def __init__(self):
        super(STNet, self).__init__()
# 在类的构造函数中定义两个神经网络模块：localization和fc_loc，用于学习空间变换的参数。其中，localization是由Conv2d、MaxPool2d和ReLU层组成的顺序网络，用于提取输入图像的特征；fc_loc是由全连接层组成的顺序网络，用于从特征图中提取变换参数。
        self.localization = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 32, kernel_size=5),
            nn.MaxPool2d(3, stride=3),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 14 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
# 初始化fc_loc的权重和偏置。其中，将fc_loc的第二层（线性层）的权重设为0，偏置设为[1, 0, 0, 0, 1, 0]，表示初始状态下无需进行变换。
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(
            [1, 0, 0, 0, 1, 0], dtype=torch.float))

# 定义前向传播函数forward。其中，xs是通过localization对输入x进行特征提取得到的特征图；theta是通过fc_loc从xs中提取的变换参数；f32fwd是定义的自定义前向传播函数，用于对输入x进行空间变换。
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 14 * 2)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        x = self.f32fwd(x, theta)

        return x

# 定义自定义前向传播函数f32fwd。其中，通过affine_grid和grid_sample函数实现对输入x的空间变换。

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def f32fwd(self, x, theta):
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x
