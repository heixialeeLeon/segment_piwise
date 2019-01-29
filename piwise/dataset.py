import numpy as np
import os

from PIL import Image
import cv2
from torchvision import transforms

from torch.utils.data import Dataset

EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, f'{basename}{extension}')

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        # self.images_root = os.path.join(root, 'images')
        # self.labels_root = os.path.join(root, 'labels')
        self.images_root = os.path.join(root, 'JPEGImages')
        self.labels_root = os.path.join(root, 'SegmentationClass')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        # p_img = np.array(label)
        # cv2.imshow("test",p_img)
        # cv2.waitKey(-1)

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)

def showImage(pil_image):
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imshow("test",cv_image)
    cv2.waitKey(-1)

def showTensorImage(tensor_image):
    pil_image = transforms.ToPILImage()(tensor_image).convert('RGB')
    showImage(pil_image)

if __name__ == "__main__":
    from torchvision.transforms import Compose, CenterCrop, Normalize
    from torchvision.transforms import ToTensor, ToPILImage
    from piwise.transform import Relabel, ToLabel, Colorize
    image_transform = ToPILImage()
    input_transform = Compose([
        CenterCrop(30),
        ToTensor(),
        #Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    target_transform = Compose([
        CenterCrop(30),
        ToLabel(),
        #Relabel(255, 21),
    ])

    dataset = VOC12("/data_1/data/VOC2012/VOCdevkit/VOC2012", input_transform, target_transform)
    for image, label in dataset:
        print(label)
        #showTensorImage(image)
