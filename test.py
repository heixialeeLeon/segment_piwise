import numpy as np
import torch

from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize
from torchvision.transforms import ToTensor, ToPILImage

from piwise.dataset import VOC12
from piwise.network import FCN8, FCN16, FCN32, UNet, PSPNet, SegNet
from piwise.criterion import CrossEntropyLoss2d
from piwise.transform import Relabel, ToLabel, Colorize, Colorize2
from piwise.visualize import Dashboard

color_transform = Colorize2()
image_transform = ToPILImage()
input_transform = Compose([
    CenterCrop(256),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform = Compose([
    CenterCrop(256),
    ToLabel(),
    Relabel(255, 21),
])


model_file = "weights/model_20.pth"
image_file = "2007_000999.jpg"
model = SegNet(22)
#model = FCN16(22)
model = model.cuda()

model.load_state_dict(torch.load(model_file))

image = Image.open(image_file)
image = input_transform(image)
image = image.cuda()
image = Variable(image, volatile=True).unsqueeze(0)
label = model(Variable(image, volatile=True))
#label = model(Variable(image, volatile=True).unsqueeze(0))
label = label[0].data.max(0)
label = label[1]
label = color_transform(label)
#label = color_transform(label[0].data.max(0)[1])
#image_transform(label).save("example")
label = image_transform(label)
label.save("img1.png")
print("Done")