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
from piwise.transform import Relabel, ToLabel, Colorize
from piwise.visualize import Dashboard
import torch.optim as optim
import argparse

NUM_CHANNELS = 3
NUM_CLASSES = 22

color_transform = Colorize()
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


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument('--datadir', required=False, default="/data_1/data/VOC2012/VOCdevkit/VOC2012")
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--save_interval', default=10, type=int, metavar='N',
                    help='number of epochs to save the model')
parser.add_argument('--batch_size', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--lr_step', '--learning-rate step', default=10, type=int,
                    help='learning rate step')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights',
                    help='Directory for saving checkpoint models')
parser.add_argument('--steps-loss', type=int, default=100)
args = parser.parse_args()
print(args)

def train():
    model = SegNet(NUM_CLASSES)
    #model = FCN16(NUM_CLASSES)

    if args.cuda:
        model = model.cuda()

    weight = torch.ones(22)
    weight[21] = 0

    loader = DataLoader(VOC12(args.datadir, input_transform, target_transform),
                        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)

    if args.cuda:
        criterion = CrossEntropyLoss2d(weight.cuda())
    else:
        criterion = CrossEntropyLoss2d(weight)

    #optimizer = Adam(model.parameters())
    optimizer = SGD(model.parameters(), lr =args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [10, 20, 30, 40], gamma=0.1)

    for epoch in range(0, args.epochs+1):
        epoch_loss = []
        scheduler.step(epoch)
        for step, (images, labels) in enumerate(loader):
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.data[0])

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average} (epoch: {epoch}, step: {step})')

        if epoch % 2 == 0:
            save_filename = "{}/model_{}.pth".format(args.save_folder,epoch)
            torch.save(model.state_dict(), save_filename)

if __name__ == "__main__":
    train()
