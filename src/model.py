# -*- coding: utf-8 -*-
"""Mean field B-CNN model."""


import torch
import torchvision
from torch import nn
from MPNCOV import MPNCOV
from builder import ConvBuilder
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benckmark = True


__all__ = ['BCNN']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-01-09'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-05-21'
__version__ = '13.7'


class BCNN(torch.nn.Module):
    """Mean field B-CNN model.

    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> mean field bilinear pooling
    -> fc.

    The network accepts a 3*448*448 input, and the relu5-3 activation has shape
    512*28*28 since we down-sample 4 times.

    Attributes:
        _is_all, bool: In the all/fc phase.
        features, torch.nn.Module: Convolution and pooling layers.
        bn, torch.nn.Module.
        gap_pool, torch.nn.Module.
        mf_relu, torch.nn.Module.
        mf_pool, torch.nn.Module.
        fc, torch.nn.Module.
    """
    def __init__(self, num_classes, is_all):
        """Declare all needed layers.

        Args:
            num_classes, int.
            is_all, bool: In the all/fc phase.
        """
        torch.nn.Module.__init__(self)
        self._is_all = is_all

        if self._is_all:
            # Convolution and pooling layers of VGG-16.
            self.features = torchvision.models.vgg16(pretrained=True).features
            self.features = torch.nn.Sequential(*list(self.features.children())
                                                [:-2])  # Remove pool5.

        # Mean filed pooling layer.
        self.relu5_3 = torch.nn.ReLU(inplace=False)

        # Classification layer.
        self.fc = torch.nn.Linear(
            in_features=512 * 512, out_features=num_classes, bias=True)

        if not self._is_all:
            self.apply(BCNN._initParameter)

    def _initParameter(module):
        """Initialize the weight and bias for each module.

        Args:
            module, torch.nn.Module.
        """
        if isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, val=1.0)
            torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out',
                                          nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, torch.nn.Linear):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, val=0.0)

    def forward(self, X):
        """Forward pass of the network.

        Args:
            X, torch.Tensor (N*3*448*448).

        Returns:
            score, torch.Tensor (N*200).
        """
        # Input.
        N = X.size()[0]
        if self._is_all:
            assert X.size() == (N, 3, 448, 448)
            X = self.features(X)
        assert X.size() == (N, 512, 28, 28)

        # The main branch.
        X = self.relu5_3(X)
        assert X.size() == (N, 512, 28, 28)

        # Classical bilinear pooling.
        X = torch.reshape(X, (N, 512, 28 * 28))
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28 * 28)
        assert X.size() == (N, 512, 512)
        X = torch.reshape(X, (N, 512 * 512))

        # Normalization.
        # X = torch.sign(X) * torch.sqrt(torch.abs(X) + 1e-5)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)

        # Classification.
        X = self.fc(X)
        return X

class Densenet169(nn.Module):
    def __init__(self):
        super(Densenet169, self).__init__()

        self.base = torchvision.models.densenet169()
        for parma in self.base.parameters():
            parma.requires_grad = False
        self.Pooling = nn.MaxPool2d((10, 10))
        self.base.classifier = nn.Sequential(nn.Linear(in_features=1664, out_features=512),
                                             nn.ReLU(),
                                             nn.Linear(in_features=512, out_features=2),
                                             nn.Sigmoid())

    #
    def forward(self, x):
        x1 = self.base.features(x)
        x2 = self.Pooling(x1)
        x2 = x2.view(x2.size(0), -1)
        y_pred = self.base.classifier(x2)
        return y_pred

class Densenet169_mpn(nn.Module):
    def __init__(self):
        super(Densenet169_mpn, self).__init__()

        self.base = torchvision.models.densenet169()
        for parma in self.base.parameters():
            parma.requires_grad = False
        self.Pooling = nn.MaxPool2d((10, 10))
        self.base.classifier = nn.Sequential(nn.Linear(in_features=32896, out_features=512),
                                             nn.ReLU(),
                                             nn.Linear(in_features=512, out_features=2),
                                             nn.Sigmoid())
        self.mpn = MPNCOV(5, True, True, 1664, 256)

    #
    def forward(self, x):
        x1 = self.base.features(x)
        # x2 = self.Pooling(x1)
        x2 = self.mpn(x1)
        x2 = x2.view(x2.size(0), -1)
        y_pred = self.base.classifier(x2)
        return y_pred

class simpleNet(nn.Module):
    def __init__(self):
        super(simpleNet, self).__init__()
        self.Conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.linear = nn.Sequential(nn.Linear(135424, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(),
                                    nn.Linear(4096, 2),
                                    nn.ReLU())

    def forward(self, X):
        x1 = self.Conv1(X)
        # print("x1.shape")
        # print(x1.shape)
        # print(x1.size(0))
        x = x1.view(x1.size(0), -1)
        # print(x.size())
        # x2 = x1.view(-1, 1024)
        # print(x2.shape)
        # print(x2.size())
        x3 = self.linear(x)
        # print(x3.shape)
        return x3

class simpleNet_mpn(nn.Module):
    def __init__(self):
        super(simpleNet_mpn, self).__init__()
        self.Conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.mpn = MPNCOV(5, True, True, 256, 256)
        self.linear = nn.Sequential(nn.Linear(32896, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(),
                                    nn.Linear(4096, 2),
                                    nn.ReLU())

    def forward(self, X):
        x1 = self.Conv1(X)
        # print("x1.shape")
        # print(x1.shape)
        # print(x1.size(0))
        x2 = self.mpn(x1)
        x = x2.view(x2.size(0), -1)
        # print(x.size())
        # x2 = x1.view(-1, 1024)
        # print(x2.shape)
        # print(x2.size())
        x3 = self.linear(x)
        # print(x3.shape)
        return x3

class simpleNet2(nn.Module):
    def __init__(self):
        super(simpleNet2, self).__init__()
        self.Conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7, 7)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.mpn = MPNCOV(5, True, True, 512, 256)
        self.linear = nn.Sequential(nn.Linear(247808, 2000),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(2000, 2),
                                    nn.ReLU())

    def forward(self, X):
        x1 = self.Conv1(X)
        # print("x1.shape")
        # print(x1.shape)
        # print(x1.size(0))
        # x2 = self.mpn(x1)
        x = x1.view(x1.size(0), -1)
        # print(x.size())
        # x2 = x1.view(-1, 1024)
        # print(x2.shape)
        # print(x2.size())
        x3 = self.linear(x)
        # print(x3.shape)
        return x3

class simpleNet2_mpn(nn.Module):
    def __init__(self):
        super(simpleNet2_mpn, self).__init__()
        self.Conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7, 7)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.mpn = MPNCOV(5, True, True, 512, 256)
        self.linear = nn.Sequential(nn.Linear(32896, 2000),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(2000, 2),
                                    nn.ReLU())

    def forward(self, X):
        x1 = self.Conv1(X)
        # print("x1.shape")
        # print(x1.shape)
        # print(x1.size(0))
        x2 = self.mpn(x1)
        x = x2.view(x2.size(0), -1)
        # print(x.size())
        # x2 = x1.view(-1, 1024)
        # print(x2.shape)
        # print(x2.size())
        x3 = self.linear(x)
        # print(x3.shape)
        return x3

class simpleNet3(nn.Module):
    def __init__(self):
        super(simpleNet3, self).__init__()
        self.Conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11, 11)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5, 5)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.linear = nn.Sequential(nn.Linear(225792, 2000),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(2000, 2),
                                    nn.ReLU())

    def forward(self, X):
        x1 = self.Conv1(X)
        # print("x1.shape")
        # print(x1.shape)
        # print(x1.size(0))
        x = x1.view(x1.size(0), -1)
        # print(x.size())
        # x2 = x1.view(-1, 1024)
        # print(x2.shape)
        # print(x2.size())
        x3 = self.linear(x)
        # print(x3.shape)
        return x3

class simpleNet3_mpn(nn.Module):
    def __init__(self):
        super(simpleNet3_mpn, self).__init__()
        self.Conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11, 11)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(5, 5)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3)),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.mpn = MPNCOV(5, True, True, 512, 256)
        self.linear = nn.Sequential(nn.Linear(32896, 2000),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(2000, 2),
                                    nn.ReLU())

    def forward(self, X):
        x1 = self.Conv1(X)
        # print("x1.shape")
        # print(x1.shape)
        # print(x1.size(0))
        x2 = self.mpn(x1)
        x = x2.view(x2.size(0), -1)
        # print(x.size())
        # x2 = x1.view(-1, 1024)
        # print(x2.shape)
        # print(x2.size())
        x3 = self.linear(x)
        # print(x3.shape)
        return x3

class Inception(nn.Module):
    def __init__(self):
        super(Inception, self).__init__()

        self.base = torchvision.models.inception_v3()
        self.features = torch.nn.Sequential(*list(self.base.children())[:-5])
        self.mpn = MPNCOV(5, True, True, 768, 256)
        self.base.aux_logits = False
        for parma in self.base.parameters():
            parma.requires_grad = False
        self.base.classifier = nn.Sequential(nn.Linear(32896, 2000),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(2000, 2),
                                    nn.ReLU())

    #list(self.base.children())[0]
    def forward(self, x):
        x1 = self.features(x)
        x2 = self.mpn(x1)
        x3 = x2.view(x2.size(0), -1)
        y_pred = self.base.classifier(x3)
        return y_pred

class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()

        self.base = torchvision.models.alexnet()
        self.mpn = MPNCOV(5, True, True, 256, 256)
        for parma in self.base.parameters():
            parma.requires_grad = False
        self.base.classifier = nn.Sequential(nn.Linear(in_features=9216, out_features=2),
                                             nn.Sigmoid())
        self.linear = nn.Sequential(nn.Linear(32896, 2000),
                                    nn.ReLU(),

                                    nn.Linear(2000, 2),
                                    nn.ReLU())

    # nn.Dropout(p=0.5),
    #
    def forward(self, x):
        x1 = self.base.features(x)
        x2 = self.mpn(x1)
        x3 = x2.view(x2.size(0), -1)

        y_pred = self.linear(x3)
        return y_pred

class Densenet_mpn(nn.Module):
    def __init__(self):
        super(Densenet_mpn, self).__init__()

        self.base = torchvision.models.DenseNet()
        for parma in self.base.parameters():
            parma.requires_grad = False
        self.Pooling = nn.MaxPool2d((10, 10))
        self.base.classifier = nn.Sequential(nn.Linear(in_features=32896, out_features=512),
                                             nn.ReLU(),
                                             nn.Linear(in_features=512, out_features=2),
                                             nn.Sigmoid())
        self.mpn = MPNCOV(5, True, True, 1664, 256)

    #
    def forward(self, x):
        x1 = self.base(x)
        # x2 = self.Pooling(x1)
        x2 = self.mpn(x1)
        x2 = x2.view(x2.size(0), -1)
        y_pred = self.base.classifier(x2)
        return y_pred

class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()

        self.base = torchvision.models.vgg11_bn()
        for parma in self.base.parameters():
            parma.requires_grad = False
        self.Pooling = nn.MaxPool2d((10, 10))
        self.base.classifier = nn.Sequential(nn.Linear(in_features=32896, out_features=2000),
                                             nn.ReLU(),
                                             nn.Linear(in_features=2000, out_features=2),
                                             nn.Sigmoid())
        self.mpn = MPNCOV(5, True, True, 512, 256)

    def forward(self, x):
        x1 = self.base.features(x)
        # x2 = self.Pooling(x1)
        x2 = self.mpn(x1)
        x2 = x2.view(x2.size(0), -1)
        y_pred = self.base.classifier(x2)
        return y_pred

class simple_mpn_ACNet(nn.Module):

    def __init__(self):
        super(simple_mpn_ACNet, self).__init__()

        sq = ConvBuilder.Sequential(self)
        sq.add_module('conv1',
                      ConvBuilder.Conv2dBNReLU(self, in_channels=3, out_channels=64, kernel_size=3, stride=1))
        sq.add_module('relu', ConvBuilder.ReLU(self))
        sq.add_module('maxpool1', ConvBuilder.Maxpool2d(self, kernel_size=3, stride=2))
        sq.add_module('conv2',
                      ConvBuilder.Conv2dBNReLU(self, in_channels=64, out_channels=96, kernel_size=3, stride=1))
        sq.add_module('relu', ConvBuilder.ReLU(self))
        sq.add_module('maxpool1', ConvBuilder.Maxpool2d(self, kernel_size=3, stride=2))

        sq.add_module('conv3',
                      ConvBuilder.Conv2dBNReLU(self, in_channels=96, out_channels=128, kernel_size=3, stride=1))
        sq.add_module('relu', ConvBuilder.ReLU(self))

        sq.add_module('conv4',
                      ConvBuilder.Conv2dBNReLU(self, in_channels=128, out_channels=256, kernel_size=3, stride=1))
        sq.add_module('relu', ConvBuilder.ReLU(self))
        sq.add_module('conv5',
                      ConvBuilder.Conv2dBNReLU(self, in_channels=256, out_channels=256, kernel_size=3, stride=1))
        sq.add_module('relu', ConvBuilder.ReLU(self))
        sq.add_module('maxpool2', ConvBuilder.Maxpool2d(self, kernel_size=3, stride=2))

        self.stem = sq
        self.mpn = MPNCOV(5, True, True, 256, 256)
        self.linear = nn.Sequential(nn.Linear(32896, 2000),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(2000, 2),
                                    nn.ReLU())

        # self.flatten = builder.Flatten()
        # self.linear1 = builder.Linear(in_features=32896, out_features=2000)
        # self.relu = builder.ReLU()
        # self.Dropout(p=0.5)
        # self.linear2 = builder.Linear(in_features=2000, out_features=2)
        # self.relu = builder.ReLU()

        def forward(self, x):
            out = self.stem(x)
            out = self.mpn(out)
            out = out.view(out.size(0), -1)
            out = self.Linear(out)
            # out = self.flatten(out)
            # out = self.linear1(out)
            # out = self.relu(out)
            # out = self.linear2(out)
            return out

class CropLayer_simpleNet(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer_simpleNet, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, self.rows_to_crop:-self.rows_to_crop, :]

class ACBlock_simpleNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False):
        super(ACBlock_simpleNet, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True,
                                        padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = CropLayer_simpleNet(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = CropLayer_simpleNet(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())
            # return square_outputs
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())
            return square_outputs + vertical_outputs + horizontal_outputs

class ACsimpleNet_mpn(nn.Module):
    def __init__(self):
        super(ACsimpleNet_mpn, self).__init__()
        self.Conv1 = nn.Sequential(ACBlock_simpleNet(in_channels=3, out_channels=64, kernel_size=3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   ACBlock_simpleNet(in_channels=64, out_channels=96, kernel_size=3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   ACBlock_simpleNet(in_channels=96, out_channels=128, kernel_size=3),
                                   nn.ReLU(),
                                   ACBlock_simpleNet(in_channels=128, out_channels=256, kernel_size=3),
                                   nn.ReLU(),
                                   ACBlock_simpleNet(in_channels=256, out_channels=256, kernel_size=3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.mpn = MPNCOV(5, True, True, 256, 256)
        self.linear = nn.Sequential(nn.Linear(32896, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(),
                                    nn.Linear(4096, 2),
                                    nn.ReLU())

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.mpn(x1)
        x = x2.view(x2.size(0), -1)
        x3 = self.linear(x)
        return x3

class ver_CropLayer_kernal_size_11(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(ver_CropLayer_kernal_size_11, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, :, 5:-5]

class hor_CropLayer_kernal_size_11(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(hor_CropLayer_kernal_size_11, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, 5:-5, :]

class ver_CropLayer_kernal_size_7(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(ver_CropLayer_kernal_size_7, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, :, 3:-3]

class hor_CropLayer_kernal_size_7(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(hor_CropLayer_kernal_size_7, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, 3:-3, :]

class ver_CropLayer_kernal_size_5(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(ver_CropLayer_kernal_size_5, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, :, 2:-2]

class hor_CropLayer_kernal_size_5(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(hor_CropLayer_kernal_size_5, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 0:
            return input[:, :, :, self.cols_to_crop:-self.cols_to_crop]
        else:
            return input[:, :, 2:-2, :]

class ACBlock_kernal_size_11(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False):
        super(ACBlock_kernal_size_11, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True,
                                        padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = ver_CropLayer_kernal_size_11(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = hor_CropLayer_kernal_size_11(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(11, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 11),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())
            # return square_outputs
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())
            return square_outputs + vertical_outputs + horizontal_outputs

class ACBlock_kernal_size_7(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False):
        super(ACBlock_kernal_size_7, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True,
                                        padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = ver_CropLayer_kernal_size_7(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = hor_CropLayer_kernal_size_7(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(7, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 7),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())
            # return square_outputs
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())
            return square_outputs + vertical_outputs + horizontal_outputs

class ACBlock_kernal_size_5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 padding_mode='zeros', deploy=False):
        super(ACBlock_kernal_size_5, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(kernel_size, kernel_size), stride=stride,
                                        padding=padding, dilation=dilation, groups=groups, bias=True,
                                        padding_mode=padding_mode)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False,
                                         padding_mode=padding_mode)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (center_offset_from_origin_border + 1, center_offset_from_origin_border)
            hor_pad_or_crop = (center_offset_from_origin_border, center_offset_from_origin_border + 1)
            if center_offset_from_origin_border >= 0:
                self.ver_conv_crop_layer = nn.Identity()
                ver_conv_padding = ver_pad_or_crop
                self.hor_conv_crop_layer = nn.Identity()
                hor_conv_padding = hor_pad_or_crop
            else:
                self.ver_conv_crop_layer = ver_CropLayer_kernal_size_5(crop_set=ver_pad_or_crop)
                ver_conv_padding = (0, 0)
                self.hor_conv_crop_layer = hor_CropLayer_kernal_size_5(crop_set=hor_pad_or_crop)
                hor_conv_padding = (0, 0)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 1),
                                      stride=stride,
                                      padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 5),
                                      stride=stride,
                                      padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False,
                                      padding_mode=padding_mode)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())
            # return square_outputs
            vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(vertical_outputs)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())
            horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(horizontal_outputs)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())
            return square_outputs + vertical_outputs + horizontal_outputs

class ACsimpleNet2_mpn(nn.Module):
    def __init__(self):
        super(ACsimpleNet2_mpn, self).__init__()
        self.Conv1 = nn.Sequential(ACBlock_kernal_size_7(in_channels=3, out_channels=96, kernel_size=7),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   ACBlock_kernal_size_5(in_channels=96, out_channels=256, kernel_size=5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   ACBlock_simpleNet(in_channels=256, out_channels=512, kernel_size=3),
                                   nn.ReLU(),
                                   ACBlock_simpleNet(in_channels=512, out_channels=512, kernel_size=3),
                                   nn.ReLU(),
                                   ACBlock_simpleNet(in_channels=512, out_channels=512, kernel_size=3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.mpn = MPNCOV(5, True, True, 512, 256)
        self.linear = nn.Sequential(nn.Linear(32896, 2000),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(2000, 2),
                                    nn.ReLU())

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.mpn(x1)
        x = x2.view(x2.size(0), -1)
        x3 = self.linear(x)
        return x3

class ACsimpleNet3_mpn(nn.Module):
    def __init__(self):
        super(ACsimpleNet3_mpn, self).__init__()
        self.Conv1 = nn.Sequential(ACBlock_kernal_size_11(in_channels=3, out_channels=64, kernel_size=11),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   ACBlock_kernal_size_5(in_channels=64, out_channels=256, kernel_size=5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2),
                                   ACBlock_simpleNet(in_channels=256, out_channels=256, kernel_size=3),
                                   nn.ReLU(),
                                   ACBlock_simpleNet(in_channels=256, out_channels=256, kernel_size=3),
                                   nn.ReLU(),
                                   ACBlock_simpleNet(in_channels=256, out_channels=256, kernel_size=3),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=3, stride=2))
        self.mpn = MPNCOV(5, True, True, 256, 256)
        self.linear = nn.Sequential(nn.Linear(32896, 2000),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(2000, 2),
                                    nn.ReLU())

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.mpn(x1)
        x = x2.view(x2.size(0), -1)
        x3 = self.linear(x)
        return x3

class simple(nn.Module):
    def __init__(self):
        super(simple, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3))
        self.Relu1 = nn.ReLU()
        self.max1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv2 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))
        self.Relu2 = nn.ReLU()
        self.max2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.Conv3 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3))
        self.Relu3 = nn.ReLU()
        self.Conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3))
        self.Relu4 = nn.ReLU()
        self.Conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3))
        self.Relu5 = nn.ReLU()
        self.max5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.linear1 = nn.Linear(135424, 4096)
        self.Relu6 = nn.ReLU()
        self.droup = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.Relu7 = nn.ReLU()
        self.linear3 = nn.Linear(4096, 2)
        self.Relu8 = nn.ReLU()

    def forward(self, X):
        x = self.Conv1(X)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = self.Relu1(x)
        x = self.max1(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = self.Conv2(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = self.Relu2(x)
        x = self.max2(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = self.Conv3(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = self.Relu3(x)
        x = self.Conv4(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = self.Relu4(x)
        x = self.Conv5(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = self.Relu5(x)
        x = self.max5(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = self.Relu6(x)
        x = self.droup(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = self.linear2(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = self.Relu7(x)
        x = self.linear3(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        x = self.Relu8(x)
        print("x.shape")
        print(x.shape)
        print(x.size(0))
        return x