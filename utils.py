import numpy as np
import matplotlib.pyplot as plt
import json

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import torchattacks
from robustbench.utils import load_model
from torchattacks import *


# model = load_model('Standard', norm='Linf').to(device) #or customize
def generate_adv(model, attack, images, labels, device="cuda"):
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
    adv_images = atk(images, labels)
    return adv_images


def get_pred(model, images, device):
    logits = model(images.to(device))
    _, pres = logits.max(dim=1)
    return pres.cpu()

def imshow(img, title):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()


def load_dataset(n_examples, loader, batch_size = 100) :

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

    return x_test_tensor, y_test_tensor


def CIFAR10(batch_size=128, test_batch_size=128):
    transform = transforms.Compose(
        [transforms.ToTensor()])
    train_dataset = datasets.CIFAR10('./datasets/CIFAR-10', train=True,
                                     download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = datasets.CIFAR10('./datasets/CIFAR-10', train=False, download=True,
                                   transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataset, val_dataset, train_loader, val_loader
