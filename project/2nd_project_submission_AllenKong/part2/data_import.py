import torch
from torchvision import datasets, transforms


def transform_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_data_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])

    test_data_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                    [0.229, 0.224, 0.225])])

    train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
    valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_data_transforms)
    test_image_datasets = datasets.ImageFolder(test_dir, transform=test_data_transforms)

    train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=32, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_image_datasets, batch_size=32)
    test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=32)
    return train_image_datasets, valid_image_datasets, test_image_datasets, train_dataloaders, valid_dataloaders, test_dataloaders