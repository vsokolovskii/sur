import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms as T
import matplotlib.pyplot as plt

import pandas as pd
import cv2
import os
import albumentations as A


if os.uname()[1].startswith('supergpu'):
    from safe_gpu import safe_gpu
    gpu_owner = safe_gpu.GPUOwner(1)

plot_loss = []

class PngsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        image_file = self.annotations.iloc[idx, 0]
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        y_label = int(self.annotations.iloc[idx, 1])

        if self.transform:
            image = self.transform(image=image)['image']

        return (image, y_label)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding=2)
        self.bnl1 = nn.BatchNorm2d(num_features=10)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.do1 = nn.Dropout(0.15)

        self.conv2 = nn.Conv2d(10, 18, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.do2 = nn.Dropout(0.4)

        self.conv3 = nn.Conv2d(18, 32, kernel_size=5, padding=2)
        self.bnl2 = nn.BatchNorm2d(num_features=32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.do3 = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(in_features=32 * (10 ** 2), out_features=256)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(in_features=256, out_features=1)

    def forward(self, x):
        x = x[:, None, :, :]

        x = self.conv1(x)
        x = self.bnl1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        s = self.do1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.do2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.do3(x)

        x = x.view(-1, 32 * 10 * 10)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)

        return x
    

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    acc_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        output = model(data).flatten()
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6]).to(device))
        loss = loss(output, target)
        acc_loss += loss
        loss.backward()
        optimizer.step()
    # calculate loss for the whole epoch
    loss = acc_loss / len(train_loader)
    #plot_loss.append(loss)
    print(f"Epoch: {epoch} Training Loss: {loss.item()}")
    wandb.log({'Training loss': loss.item()})

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
            output = model(data).flatten()
            test_loss += torch.nn.BCEWithLogitsLoss()(output, target).item()  # sum up batch loss
            output = torch.sigmoid(output)
            pred = output.round()  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    wandb.log({'Validation loss': test_loss, 'Accuracy': correct/len(test_loader.dataset)})
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)")

def main(show_image=False):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5, rotate_limit=30, border_mode=cv2.BORDER_REPLICATE),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
    ])
    # create dataloader
    train_loader = torch.utils.data.DataLoader(PngsDataset('pngs-train.csv', transform=transform), batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(PngsDataset('pngs-test.csv', transform=transform), batch_size=4, shuffle=False)
    iterator = iter(train_loader)
    # save tensor as image
    if show_image:
        transform1 = T.ToPILImage()
        for i in range(10):
            img, label = next(iterator)
            img = transform1(img[0])
            img.show()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # initialize wandb
    wandb.init(project='sur', mode="offline")
    # create model
    model = CNN().to(device)
    # create optimizer
    #optimizer = optim.SGD(model.parameters(), lr=1e-8, momentum=0.9)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-2)
    # train model
    for epoch in range(1, 5000):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, val_loader)
        # save modekj
        torch.save(model.state_dict(), 'model.pt')
        # save model to wandb
        wandb.save('model.pt')
        if os.uname()[1].startswith('supergpu'):
            pass
            #trigger_sync()
    plt.plot(plot_loss)
        

if __name__ == '__main__':
    main(show_image=False)
