#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune all layers for bilinear CNN.

This is the second step.
"""


import os
import time

import torch
import torchvision
import Breast
import cub200
import model
import model_inception_resnet_v2
from builder import ConvBuilder
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type(torch.FloatTensor)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True


__all__ = ['BCNNManager']
__author__ = 'Hao Zhang'
__copyright__ = '2018 LAMDA'
__date__ = '2018-01-11'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2018-05-19'
__version__ = '13.1'


class BCNNManager(object):
    """Manager class to train bilinear CNN.

    Attributes:
        _is_all, bool: In the all/fc phase.
        _options, dict<str, float/int>: Hyperparameters.
        _paths, dict<str, str>: Useful paths.
        _net, torch.nn.Module: Bilinear CNN.
        _criterion, torch.nn.Module: Cross-entropy loss.
        _optimizer, torch.optim.Optimizer: SGD with momentum.
        _scheduler, tirch.optim.lr_scheduler: Reduce learning rate when plateau.
        _train_loader, torch.utils.data.DataLoader.
        _test_loader, torch.utils.data.DataLoader.
    """
    def __init__(self, options, paths):
        """Prepare the network, criterion, optimizer, and data.

        Args:
            options, dict<str, float/int>: Hyperparameters.
            paths, dict<str, str>: Useful paths.
        """
        print('Prepare the network and data.')

        # Configurations.
        self._options = options
        self._paths = paths

        # Network.

        self._net = torch.nn.DataParallel(model.simple()).cuda()
        # self._net.load_state_dict(torch.load(r'H:\LiJiaSen\python_project\simple\src\model\bcnn_all_epoch_34.pth'),
        #                                          strict=False)

        print(self._net)
        self._criterion = torch.nn.CrossEntropyLoss().cuda()

        # Optimizer.
        self._optimizer = torch.optim.SGD(
            self._net.parameters(), lr=self._options['base_lr'],
            momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer, mode='max', factor=0.1, patience=8, verbose=True,
            threshold=1e-4)

        # Data.

        train_transforms = torchvision.transforms.Compose([
            # torchvision.transforms.RandomResizedCrop(size=448,
            #                                             scale=(0.8, 1.0)),
            torchvision.transforms.Resize(size=(224, 224)),
            # torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
            #                                     std=(0.229, 0.224, 0.225)),

        ])
        test_transforms = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(size=448),
            # torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
            #                                  std=(0.229, 0.224, 0.225)),
        ])
        train_data = Breast.BreastCancer(
            root=r'I:\\Dataset_crop_224_into_five\400x', train=True,
            transform=train_transforms, download=True)
        test_data = Breast.BreastCancer(
            root=r'I:\\Dataset_crop_224_into_five\400x', train=False,
            transform=test_transforms, download=True)

        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'], shuffle=True,
            num_workers=4, pin_memory=False)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=64,
            shuffle=False, num_workers=4, pin_memory=False)

    def train(self):
        """Train the network."""
        print('Training.')
        self._net.train()
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc\tTime')
        for t in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            tic = time.time()
            for instances, labels in self._train_loader:
                # Data.
                instances = instances.cuda()
                labels = labels.cuda()

                # Forward pass.
                score = self._net(instances)
                loss = self._criterion(score, labels)

                with torch.no_grad():
                    epoch_loss.append(loss.item())
                    # Prediction.
                    prediction = torch.argmax(score, dim=1)
                    num_total += labels.size(0)
                    num_correct += torch.sum(prediction == labels).item()

                # Backward pass.
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                del instances, labels, score, loss, prediction
            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
                print('*', end = '')
                save_path = os.path.join(
                    self._paths['model'],
                    'bcnn_%s_epoch_%d.pth' % (
                        'all', t + 1))
                torch.save(self._net.state_dict(), save_path)
            toc = time.time()
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%\t\t%4.2f min' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_acc,
                   test_acc, (toc - tic) / 60))
            self._scheduler.step(test_acc)
        print('Best at epoch %d, test accuaray %4.2f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        """Compute the train/test accuracy.

        Args:
            data_loader: Train/Test DataLoader.

        Returns:
            Train/Test accuracy in percentage.
        """
        with torch.no_grad():
            self._net.eval()
            num_correct = 0
            num_total = 0
            for instances, labels in data_loader:
                # Data.
                instances = instances.cuda()
                labels = labels.cuda()

                # Forward pass.
                score = self._net(instances)

                # Predictions.
                prediction = torch.argmax(score, dim=1)
                num_total += labels.size(0)
                num_correct += torch.sum(prediction == labels).item()
            self._net.train()  # Set the model to training phase
        return 100 * num_correct / num_total


def main():
    """The main function."""
    import argparse
    parser = argparse.ArgumentParser(
        description='Train mean field bilinear CNN on CUB200.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, default=1e-3,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=8,
                        help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int, default=60,
                        help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        default=0.001, help='Weight decay.')
    # parser.add_argument('--pretrained', dest='pretrained', type=str,
    #                     required=False, help='Pre-trained model.')
    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    project_root = os.popen('pwd').read().strip()
    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
    }
    paths = {
        'cub200': os.path.join(project_root, 'data', 'cub200'),
        'aircraft': os.path.join(project_root, 'data', 'aircraft'),
        'model': os.path.join(project_root, 'model'),
        # 'pretrained': (os.path.join(project_root, 'model', args.pretrained)
        #                if args.pretrained else None),
    }
    for d in paths:
        if d == 'pretrained':
            assert paths[d] is None or os.path.isfile(paths[d])
        else:
            assert os.path.isdir(paths[d])

    manager = BCNNManager(options, paths)
    manager.train()


if __name__ == '__main__':
    main()
