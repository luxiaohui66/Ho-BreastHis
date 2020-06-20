import Breast_test_data
import torch
import model
import torchvision
from torch import nn
import torch.nn.functional as F
test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            # torchvision.transforms.CenterCrop(size=448),
            # torchvision.transforms.Resize(size=(460, 460)),
            torchvision.transforms.ToTensor(),
            # torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
            #                                  std=(0.229, 0.224, 0.225)),
        ])
test_data = Breast_test_data.BreastCancer(
            root=r'I:\happy\40x', train=False,
            transform=test_transforms, download=True)
test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=1,
            shuffle=False, num_workers=0, pin_memory=False)
net = torch.nn.DataParallel(model.simpleNet()).cuda()
net.load_state_dict(torch.load(r'D:\lijiasen\simpleNet\40x\bcnn_all_epoch_95.pth'))
net.eval()
softmax = torch.nn.LogSoftmax()
# strict=False
num_correct = 0
num_total = 0
i = 1
for instances, labels in test_loader:
    # Data.
    instances = instances.cuda()
    labels = labels.cuda()

    # Forward pass.
    score = net(instances)
    score = F.softmax(score)


    # Predictions.
    prediction = torch.argmax(score, dim=1)
    num_total += labels.size(0)
    num_correct += torch.sum(prediction==labels).item()
    if prediction != labels:
        print(i)
    i = i + 1

acc = 100 * num_correct / num_total
print('Image level acc is:')
print(acc)
print('num_total')
print(num_total)

