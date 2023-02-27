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

config = dict(
    epochs=2000,
    batch_size=4,
    learning_rate=1e-3,
    weight_decay=1e-2,
    dropout=0.4,
    num_workers=0,
    wandb_run_desc="pos_weight1",
    train_data_desc_file="pngs-train.csv",
    test_data_desc_file="pngs-test.csv",
    optimizer=torch.optim.Adagrad,
    optimizer_name='Adagrad',
    wandb_mode='offline')


if os.uname()[1].startswith('supergpu'):
    from safe_gpu import safe_gpu
    gpu_owner = safe_gpu.GPUOwner(1)
    config['wandb_mode'] = 'offline'

if config['wandb_mode'] == 'offline':
    from wandb_osh.hooks import TriggerWandbSyncHook
    trigger_sync = TriggerWandbSyncHook()


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
    for _, (data, target) in enumerate(train_loader):
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
    print(f"Epoch: {epoch} Training Loss: {loss.item()}")
    wandb.log({'Training loss': loss.item()}, step=epoch)

def validate(model, device, valid_loader, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.type(torch.FloatTensor).to(device), target.type(torch.FloatTensor).to(device)
            output = model(data).flatten()
            val_loss += torch.nn.BCEWithLogitsLoss()(output, target).item()  # sum up batch loss
            output = torch.sigmoid(output)
            pred = output.round()  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(valid_loader.dataset)
    wandb.log({'Validation loss': val_loss, 'Accuracy': correct/len(valid_loader.dataset)}, step=epoch)
    print(f"Test set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(valid_loader.dataset)} ({100. * correct / len(valid_loader.dataset):.0f}%)")

def make(config):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5, rotate_limit=30, border_mode=cv2.BORDER_REPLICATE),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
    ])
    # create data loaders
    train_loader = torch.utils.data.DataLoader(PngsDataset('pngs-train.csv', transform=transform), batch_size=config['batch_size'], shuffle=True)
    val_loader = torch.utils.data.DataLoader(PngsDataset('pngs-test.csv', transform=transform), batch_size=config['batch_size'], shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create model
    model = CNN().to(device)
    # create optimizer
    optimizer = config['optimizer'](model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])

    return model, device, train_loader, val_loader, optimizer


def main(show_image=False):
    run_name = f"lr-{config['learning_rate']}_l2-{config['weight_decay']}_dropout-{config['dropout']}_batch-{config['batch_size']}_optim-{str(config['optimizer_name'])}_run_{config['wandb_run_desc']}"
    # initialize wandb
    with wandb.init(project='sur', mode=config['wandb_mode'], name=run_name):
        # config
        wandb.config.update(config)
        # create model
        model, device, train_loader, val_loader, optimizer = make(config)
        # train model
        for epoch in range(1, config['epochs'] + 1):
            train(model, device, train_loader, optimizer, epoch)
            validate(model, device, val_loader, epoch)
            # save modekj
            torch.save(model.state_dict(), 'model.pt')
            # save model to wandb
            wandb.save('model.pt')
            if config['wandb_mode'] == 'offline':
                trigger_sync()
        

if __name__ == '__main__':
    main(show_image=False)
