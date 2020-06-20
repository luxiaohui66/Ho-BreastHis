import torch.nn as nn
from builder import ConvBuilder

class simple_mpn_ACNet(nn.Module):

    def __init__(self, builder:ConvBuilder):
        super(VCNet, self).__init__()
        self.bd = builder
        sq = builder.Sequential()
        sq.add_module('conv1', builder.Conv2dBNReLU(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1))
        sq.add_module('relu', builder.ReLU())
        sq.add_module('maxpool1', builder.Maxpool2d(kernel_size=3, stride=2))
        sq.add_module('conv2', builder.Conv2dBNReLU(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1))
        sq.add_module('relu', builder.ReLU())
        sq.add_module('maxpool1', builder.Maxpool2d(kernel_size=3, stride=2))

        sq.add_module('conv3', builder.Conv2dBNReLU(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1))
        sq.add_module('relu', builder.ReLU())

        sq.add_module('conv4', builder.Conv2dBNReLU(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1))
        sq.add_module('relu', builder.ReLU())
        sq.add_module('conv5', builder.Conv2dBNReLU(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1))
        sq.add_module('relu', builder.ReLU())
        sq.add_module('maxpool2', builder.Maxpool2d(kernel_size=3, stride=2))
        
        self.stem = sq
        self.mpn = MPN(5, True, True, 256, 256)
        self.linear = nn.Sequential(nn.Linear(32896, 2000),
                      nn.ReLU(),
                      nn.Dropout(p=0.5),
                      nn.Linear(2000, 2)
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


def create_vc(cfg, builder):
    return VCNet(num_classes=10, builder=builder, deps=cfg.deps)
def create_vh(cfg, builder):
    return VCNet(num_classes=100, builder=builder, deps=cfg.deps)
