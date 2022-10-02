import json
import pickle
import os
import time
import copy

import numpy as np
from skimage import io

from tqdm import tqdm, tqdm_notebook
from PIL import Image
from pathlib import Path

import torch
from torchvision import transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

DEVICE = torch.device("cuda")
BATCH_SIZE = 32
NUM_EPOCHS = 6
feature_extract = True


class CarDataset(Dataset):
    """
    A dataset with pictures, which parallel loads them from folders,
    scans and turns them into end tensors
    """

    def __init__(self, files):
        super().__init__()
        # file list for loading
        self.files = sorted(files)

        self.len_ = len(self.files)

        self.label_encoder = LabelEncoder()

        # create label for image
        self.labels = [" ".join([path.parent.parent.name, path.parent.name]) for path in self.files]
        self.label_encoder.fit(self.labels)

        with open('label_encoder.pkl', 'wb') as le_dump_file:
            pickle.dump(self.label_encoder, le_dump_file)

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file).convert('RGB')
        image.load()
        return image

    def __getitem__(self, index):
        # tarnsform image to pytorch tensor
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        x = self.load_sample(self.files[index])
        x = transform(x)
        label = self.labels[index]
        label_id = self.label_encoder.transform([label])
        y = label_id.item()
        return x, y


def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    val_loss = []
    train_loss = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss.append(epoch_loss)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, val_loss, train_loss

# Set train data
TRAIN_DIR = Path('data/train/av-photos')
train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))
train_val_labels = [" ".join([path.parent.parent.name, path.parent.name]) for path in train_val_files]

# Split data to train and validation
train_files, val_files = train_test_split(train_val_files, test_size=0.25,
                                          stratify=train_val_labels, random_state=1)
val_dataset = CarDataset(val_files)
train_dataset = CarDataset(train_files)

CLASS_NUM = len(set(train_val_labels))

# Load pretraining model
efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
efficientnet.classifier.fc = torch.nn.Linear(in_features=efficientnet.classifier.fc.in_features, out_features=CLASS_NUM)

# Send the model to GPU
model_ft = efficientnet.to(DEVICE)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Make dataloader dict for training
dataloaders_dict = {}
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
dataloaders_dict['train'] = train_loader
dataloaders_dict['val'] = val_loader

# Set a criterion
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, val_hist, train_hist, val_loss, train_loss = train_model(model_ft,
                                                                   dataloaders_dict,
                                                                   criterion,
                                                                   optimizer_ft,
                                                                   num_epochs=NUM_EPOCHS)
# Save model
torch.save({'model_state_dict': model_ft.state_dict(),
            'optimizer_state_dict': optimizer_ft.state_dict()}, 'FN-EfficientNet_b18_class_weight.tar')
