# @title 默认标题文本 { form-width: "150px" }
# 1/导入库
# import ding_models#后加的
import efficientnet

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np


# 2/数据集实例：
def get_x(path):
    """Gets the x value from the image filename"""
    return (float(int(path[3:6])) - 50.0) / 50.0


def get_y(path):
    """Gets the y value from the image filename"""
    return (float(int(path[7:10])) - 50.0) / 50.0


class XYDataset(torch.utils.data.Dataset):

    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image = PIL.Image.open(image_path)
        x = float(get_x(os.path.basename(image_path)))
        y = float(get_y(os.path.basename(image_path)))

        if float(np.random.rand(1)) > 0.5:
            image = transforms.functional.hflip(image)
            x = -x

        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy()[::-1].copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image, torch.tensor([x, y]).float()


# dataset = XYDataset('dataset_xy', random_hflips=False)
dataset = XYDataset('/content/drive/MyDrive/dataset_xy', random_hflips=False)

# 3/把数据集分为训练集和测试集
test_percent = 0.2
num_test = int(test_percent * len(dataset))
# 第二种方案
torch.manual_seed(0)  # pytorch 官方给的案例
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

# 4/创建数据加载器以批量加载数据
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

# 5/经网络模型
  #efficientnet_b0
# model = models.efficientnet_b0(pretrained=True)#efficientnet_b0
# model = efficientnet.efficientnet_b0(pretrained=True)#efficientnet_b0
model = efficientnet.efficientnet_b0(pretrained=False)#efficientnet_b0


model.classifier = torch.nn.Linear(1280, 2)  #efficientnet_b0

# efficientnet_b0
# !wget https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth
# model = ding_models.efficientnet_b0(pretrained=False)
# model.load_state_dict(torch.load('efficientnet_b0_rwightman-3dd342df.pth'))
# model.classifier = torch.nn.Linear(1280, 2)


device = torch.device('cuda')
model = model.to(device)



# 6/回归训练
NUM_EPOCHS = 70
BEST_MODEL_PATH = '/content/best_steering_model_xy_efficientnet_b0.pth'
best_loss = 1e9
best_loss1 = 0.0

optimizer = optim.Adam(model.parameters())
logs = []
for epoch in range(NUM_EPOCHS):
    log = {'train': [], 'test': []}
    model.train()
    train_loss = 0.0
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        train_loss += float(loss)
        log['train'].append(loss)
        loss.backward()
        optimizer.step()
    train_loss /= len(train_loader)

    model.eval()
    test_loss = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = F.mse_loss(outputs, labels)
        log['test'].append(loss)
        test_loss += float(loss)
    test_loss /= len(test_loader)
    logs.append(log)
    torch.save(logs, 'log.pth')
    print('%f, %f' % (train_loss, test_loss))
    
    if test_loss < best_loss:
        # torch.save(model.state_dict(), BEST_MODEL_PATH)
        torch.save(model.state_dict(), BEST_MODEL_PATH, _use_new_zipfile_serialization=False)
        best_loss = test_loss
        best_loss1 = best_loss
    
print('%f' % (best_loss1))
