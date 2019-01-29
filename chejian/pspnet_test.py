import numpy as np
import torch

from PIL import Image
from argparse import ArgumentParser

from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage


from model.pspnet.pspnet import *
from piwise.criterion import CrossEntropyLoss2d
from piwise.transform import Relabel, ToLabel, Colorize, Colorize2
from piwise.visualize import Dashboard
import os

color_transform = Colorize2()
image_transform = ToPILImage()
input_transform = Compose([
    Resize(320),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),
])
target_transform = Compose([
    Resize(320, interpolation=Image.NEAREST),
    ToLabel(),
    Relabel(255, 21),
])

def getImageFiles(folder):
    filelist = os.listdir(folder)
    images=list()
    for item in filelist:
        images.append("{}/{}".format(folder,item))
    return images

def processImage(net, outputFolder, imageFile):
    image = Image.open(imageFile)
    image = input_transform(image)
    image = image.cuda()
    image = Variable(image, volatile=True).unsqueeze(0)
    label = net(Variable(image, volatile=True))
    label = label[0].data.max(0)
    label = label[1]
    label = color_transform(label)
    label = image_transform(label)
    imageName = imageFile.split('/')[-1]
    outputImages = f"{outputFolder}/{imageName}"
    label.save(outputImages)
    print(f"process {outputImages} done")

if __name__ == "__main__":
    model_file = "weights/model_50.pth"
    #model = SegNet(3,12)
    #model = FCN16(12)
    model = PSPNet(n_classes= 12, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18')
    model.eval()
    model = model.cuda()

    model.load_state_dict(torch.load(model_file))

    imageFiles = getImageFiles("testImages")
    for item in imageFiles:
        processImage(model,"outputImages",item)


# model_file = "weights/model_20.pth"
# image_file = "2007_000999.jpg"
# model = SegNet(22)
# #model = FCN16(22)
# model.eval()
# model = model.cuda()
#
# model.load_state_dict(torch.load(model_file))
#
# image = Image.open(image_file)
# image = input_transform(image)
# image = image.cuda()
# image = Variable(image, volatile=True).unsqueeze(0)
# label = model(Variable(image, volatile=True))
# #label = model(Variable(image, volatile=True).unsqueeze(0))
# label = label[0].data.max(0)
# label = label[1]
# label = color_transform(label)
# #label = color_transform(label[0].data.max(0)[1])
# #image_transform(label).save("example")
# label = image_transform(label)
# label.save("img1.png")
# print("Done")