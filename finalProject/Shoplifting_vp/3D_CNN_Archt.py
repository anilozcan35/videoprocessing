# importların yapılması
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torchinfo import summary

import videoprocessing.finalProject.Shoplifting_vp.dataloader as dataloader
import videoprocessing.finalProject.Shoplifting_vp.utils as utils

import time
import importlib

# reloading of dataloader
importlib.reload(dataloader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#