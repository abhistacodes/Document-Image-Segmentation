#import libraries

import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

import matplotlib.pyplot as plt # type: ignore

#loading bar for loops
# from tqdm.auto import tqdm # type: ignore

from sklearn.model_selection import train_test_split # type: ignore

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchinfo import summary # type: ignore

import pandas as pd


#plot training curves

def plot_training_curves(train_loss, val_loss):
    plt.figure(figsize=(10, 6))

    plt.plot(
        train_loss,
        label="Training Loss"
    )

    plt.plot(
        val_loss,
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_validation_dice(dice_values):
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(dice_values) + 1),
        dice_values,
        marker="o",
        label="Validation Dice"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice Score")
    plt.xticks(range(1, len(dice_values) + 1))
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_validation_iou(iou_values):
    plt.figure(figsize=(10, 6))

    plt.plot(
        range(1, len(iou_values) + 1),
        iou_values,
        marker="o",
        label="Validation IoU"
    )
    
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("Validation Mean IoU")
    plt.legend()
    plt.grid(True)
    plt.show()