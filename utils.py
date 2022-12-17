import numpy as np
import matplotlib.pyplot as plt
import json

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchattacks import *
import robustbench
from robustbench.utils import load_model, clean_accuracy

def generate_adv(model, attack):
    """
        generate adversarial attacks 
        model: the model to be attacked 
        attack: the name of the attack 
    """
    if attack == "pgd":
        atk = PGD(model, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True)
    elif attack == "fgsm":
        atk = FGSM(model)
    elif attack == "pixle":
        atk = Pixle(model)
    elif attack == "nifgsm":
        atk = NIFGSM(model)
    elif attack == "autoattack":
        atk = AutoAttack(model)
    elif attack == "vnifgsm":
        atk = VNIFGSM(model)
    elif attack =="vmifgsm":
        atk = VMIFGSM(model)
    
    return atk


def get_pred(model, images, device):
    """
        Get the predictions from the model 
        model: the model 
        images: the images as the input 
        device: the device to use cuda / cpu 
    """
    logits = model(images.to(device))
    _, pres = logits.max(dim=1)
    return pres.cpu()

def imshow(img, title):
    """
        Show the image 
        image: the image 
        title: the title of the image 
    
    """
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()


def load_dataset(n_examples, loader, batch_size = 100) :
    """
        Load the dataset for generating adversarial samples 
        
        n_examples: the number of examples 
        loader: the data loader 
        batch_size: the batch size 
        
    """

    x_list, y_list = [], []
    for i, (x, y) in enumerate(loader):
        x_list.append(x)
        #print(x.shape)
        y_list.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_list_tensor = torch.cat(x_list)
    y_list_tensor = torch.cat(y_list)

    if n_examples is not None:
        x_list_tensor = x_list_tensor[:n_examples]
        y_list_tensor = y_list_tensor[:n_examples]

    return x_list_tensor, y_list_tensor


def CIFAR10(batch_size=128, finetune = False, input_size = 224, test_batch_size=128):
    """
    The cifar 10 dataset 
    batch_size: the batch size default is 128
    finetune: whether to transfomer the dataset 
    input_size: the input size 
    test_batch_size: the batch size of test dataset 
    
    """

    if finetune:
        transformer = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    else:
        transformer = {
        'train': transforms.Compose([
            
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
          
            transforms.ToTensor()
        ]),
    }
        

    train_dataset = datasets.CIFAR10('./datasets/CIFAR-10', train=True,
                                     download=True, transform=transformer['train'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = datasets.CIFAR10('./datasets/CIFAR-10', train=False, download=True,
                                   transform=transformer['val'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataset, val_dataset, train_loader, val_loader



def CIFAR100(batch_size=128, finetune = False, input_size = 224, test_batch_size=128):
    """
    The cifar 100 dataset 
    batch_size: the batch size default is 128
    finetune: whether to transfomer the dataset 
    input_size: the input size 
    test_batch_size: the batch size of test dataset 
    
    """

    if finetune:
        transformer = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    else:
        transformer = {
        'train': transforms.Compose([
            
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
          
            transforms.ToTensor()
        ]),
    }
        


    train_dataset = datasets.CIFAR100('./datasets/CIFAR-100', train=True,
                                     download=True, transform=transformer['train'])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = datasets.CIFAR100('./datasets/CIFAR-100', train=False, download=True,
                                   transform=transformer['val'])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataset, val_dataset, train_loader, val_loader


def test_attack(val_loader, attack, model, device = "cuda"):
    """
    test the accuracy with the attacked testloader 
    
    val_loader: the validation loader 
    atttack: the attack used for generating adversarial samples 
    model: the model of to be tested 
    device: the device for testing 
    
    """
    step_counter = 0
    acc_counter = 0
    for i, (images, labels) in enumerate(val_loader):
        adv_images_adv = attack(images, labels)
        adv_images_adv= adv_images_adv.to(device)
        model = model.to(device)
        labels = labels.to(device)
        acc = clean_accuracy(model, adv_images_adv, labels)
        acc_counter  += acc*len(images)

        step_counter += len(images)

    print('[Model loaded]')
    print('Acc: %2.2f %%'%(acc_counter/step_counter*100))
    print("adv model") 
    return acc_counter / step_counter 



