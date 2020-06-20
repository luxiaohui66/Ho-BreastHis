import Breast_extract_test_data
import torch
import model
import torchvision
from torch import nn
import numpy as np
import torch.nn.functional as F
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# from xgboost import XGBClassifier
import gc
train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
        ])
test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
        ])
train_data = Breast_extract_test_data.BreastCancer(
            root=r'/home/cad-1/Lijiasen/dataset/Dataset_crop_224_into_five/100x', train=True,
            transform=train_transforms, download=True)
test_data = Breast_extract_test_data.BreastCancer(
            root=r'/home/cad-1/Lijiasen/dataset/Dataset_crop_224_into_five/100x', train=False,
            transform=test_transforms, download=True)
train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False)
test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1, shuffle=False,
            num_workers=0, pin_memory=False)
net = torch.nn.DataParallel(model.simpleNet_mpn()).cuda()
net.load_state_dict(torch.load(r'/home/cad-1/Lijiasen/breast_cancer/src/model/100mpn_1zhe/bcnn_all_epoch_61.pth'))
net.eval()
softmax = torch.nn.LogSoftmax()

model = nn.Sequential(net.module.Conv1[0:12],net.module.mpn.conv_dr_block[0:3])
# block2 = net.module.base.features[5:7]
# block3 = net.module.base.features[7:9]
# block4 = net.module.base.features[9:11]
# norm = net.module.base.features[9:11]

block1_feature_extract = []
block2_feature_extract = []
block3_feature_extract = []
block4_feature_extract = []
test_block1_feature_extract = []
test_block2_feature_extract = []
test_block3_feature_extract = []
test_block4_feature_extract = []
for instances, labels in train_loader:
    # Data.
    instances = instances.cuda()
    labels = labels.cuda()
    labels = labels.data.cpu().numpy()
    # Forward pass.
    block1_feature = model(instances)
    del instances
    block1_feature_avg_pool = torch.nn.functional.adaptive_avg_pool2d(block1_feature, (1, 1))
    del block1_feature
    block1_feature_dim2 = block1_feature_avg_pool.view(block1_feature_avg_pool.size(0), -1)
    block1_feature_dim2 = block1_feature_dim2.data.cpu().numpy()
    test_block1_feature_extract.append(block1_feature_dim2)
    gc.collect()
np.save('simpleNet_mpn_train.npy', test_block1_feature_extract)


    # block2_feature = block2(block1_feature)
    # del block1_feature
#     block2_feature_avg_pool = torch.nn.functional.adaptive_avg_pool2d(block2_feature, (1, 1))
#     del block2_feature
#     block2_feature_dim2 = block2_feature_avg_pool.view(block2_feature_avg_pool.size(0), -1)
#     block2_feature_dim2 = block2_feature_dim2.data.cpu().numpy()
#     test_block2_feature_extract.append(block2_feature_dim2)
#     gc.collect()
# np.save('400x_test_block2_feature_extract.npy', test_block2_feature_extract)
    #
    # block3_feature = block3(block2_feature)
    # del block2_feature
#     block3_feature_avg_pool = torch.nn.functional.adaptive_avg_pool2d(block3_feature, (1, 1))
#     del block3_feature
#     block3_feature_dim2 = block3_feature_avg_pool.view(block3_feature_avg_pool.size(0), -1)
#     block3_feature_dim2 = block3_feature_dim2.data.cpu().numpy()
#     test_block3_feature_extract.append(block3_feature_dim2)
# np.save('400x_test_block3_feature_extract.npy', test_block3_feature_extract)

#     block4_feature = block4(block3_feature)
#     del block3_feature
#     block4_feature_avg_pool = torch.nn.functional.adaptive_avg_pool2d(block4_feature, (1, 1))
#     del block4_feature
#     block4_feature_dim2 = block4_feature_avg_pool.view(block4_feature_avg_pool.size(0), -1)
#     block4_feature_dim2 = block4_feature_dim2.data.cpu().numpy()
#     test_block4_feature_extract.append(block4_feature_dim2)
# np.save('400x_test_block4_feature_extract.npy', test_block4_feature_extract)

#
#
#
#