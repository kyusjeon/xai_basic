import time
import copy
import argparse

from PIL import Image as image
from tqdm import tqdm

import torch
from torch import nn, optim
from torchvision import models, transforms, datasets

def train_model(model, criterion, optimizer, scheduler, num_epochs:int):
    """train the model

    Parameters
    ----------
    model : torchvision.models
        set target model
    criterion : torch.nn.modules.loss
        set target loss
    optimizer : torch.optim
        set target optimizer
    scheduler : torch.optim.lr_scheduler
        set scheduler
    num_epochs : int
        set epochs

    Returns
    -------
    torchvision.models
        trained model
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser('train', add_help=False)
    parser.add_argument('--data_dir', default='.', type=str)
    parser.add_argument('--save_dir', default='./model.ckpt', type=str)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--step_size', default=7, type=int)
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    args = parser.parse_args()
    
    data_dir = args.data_dir
    
    data_transforms = {'train': transforms.Compose([
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ]),
                    'val': transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ]),
                    }

    image_datasets = datasets.ImageFolder(data_dir)
    dataset_sizes = len(image_datasets)
    train_datasets = torch.utils.data.Subset(image_datasets, 
                                            range(dataset_sizes // 10 * 9)
                                            )
    val_datasets = torch.utils.data.Subset(image_datasets, 
                                        range(dataset_sizes // 10 * 9, dataset_sizes)
                                        )
    train_datasets.dataset.transform = data_transforms['train']
    val_datasets.dataset.transform = data_transforms['val']

    dataloaders = {'train': torch.utils.data.DataLoader(train_datasets, 
                                                        batch_size=args.batch_size,
                                                        shuffle=True, 
                                                        num_workers=args.num_workers
                                                        ),
                'val': torch.utils.data.DataLoader(val_datasets, 
                                                batch_size=args.batch_size,
                                                shuffle=True, 
                                                num_workers=args.num_workers
                                                )
                }

    dataset_sizes = {'train': len(train_datasets),
                    'val': len(val_datasets)
                    }
    class_names = image_datasets.classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set model
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 9)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.SGD(model.parameters(), 
                            lr=args.learning_rate, 
                            momentum=args.momentum
                            )
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, 
                                                step_size=args.step_size, 
                                                gamma=args.gamma
                                                )

    model_ft = train_model(model, 
                        criterion, 
                        optimizer_ft, 
                        exp_lr_scheduler,
                        num_epochs=args.epochs
                        )
    
    torch.save(model_ft.state_dict(), args.save_dir)
