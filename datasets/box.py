import numpy as np
import os
from PIL import Image
import torch
import json
from torch.utils.data import Dataset
from torchvision.transforms import Compose
home_dir = os.getcwd()
print(home_dir)
os.chdir(home_dir)

import cv2
import matplotlib.pyplot as plt 
import seaborn as sns

from utils.transforms import OneHotEncode, OneHotEncode_smooth

def load_image(file):
    return Image.open(file)

def read_img_list(filename):
    with open(filename) as f:
        img_list = []
        for line in f:
            # img_list.append(line[:-1])
            img_list.append(line.strip('\n'))
    return np.array(img_list)

class BoxSet(Dataset):

    TRAIN_LIST = "ImageSets/train.txt"
    VAL_LIST = "ImageSets/val.txt"

    def __init__(self, root, data_root, img_transform = Compose([]),\
     label_transform=Compose([]), co_transform=Compose([]),\
      train_phase=True,split=1,labeled=True,seed=0,label_correction=False):
        np.random.seed(666)
        self.n_class = 7
        self.root = root
        self.data_root = data_root
        self.images_root = os.path.join(self.data_root, 'box', 'JPEGImages')
        # print("images_root: ", self.images_root)
        self.labels_root = os.path.join(self.data_root, 'box', 'SCRLabels') # SemanticLabels BoxLabels
        # print("labels_root: ", self.labels_root)
        self.elabels_root = os.path.join(self.data_root, 'box', 'EvaluateLabels')
        self.clabels_root = os.path.join(self.data_root, 'box', 'SCRLabelsTrue')
        self.json_root = os.path.join(self.data_root, 'box', 'json')
        self.img_list = read_img_list(os.path.join(self.data_root,'box',self.TRAIN_LIST)) \
                        if train_phase else read_img_list(os.path.join(self.data_root,'box',self.VAL_LIST))
        self.split = split
        self.labeled = labeled
        n_images = len(self.img_list)
        self.img_l = np.random.choice(range(n_images),int(n_images*split),replace=False) # Labeled Images
        self.img_u = np.array([idx for idx in range(n_images) if idx not in self.img_l],dtype=int) # Unlabeled Images
        if self.labeled:
            self.img_list = self.img_list[self.img_l]
        else:
            self.img_list = self.img_list[self.img_u]
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.co_transform = co_transform
        self.train_phase = train_phase
        self.label_correction = label_correction

    def __getitem__(self, index):
        filename = self.img_list[index]

        with open(os.path.join(self.images_root,filename+'.png'), 'rb') as f:
            image = load_image(f).convert('RGB')
            # image.save(filename + '_org.png')
        
        with open(os.path.join(self.labels_root,filename+'.png'), 'rb') as f:
            label = load_image(f).convert('L')
            # label = Image.fromarray(np.uint8(label_pil))
        
        with open(os.path.join(self.elabels_root,filename+'.bmp'), 'rb') as f:
            elabel = load_image(f).convert('L')
        
        with open(os.path.join(self.clabels_root,filename+'.bmp'), 'rb') as f:
            # clabel = load_image(f).convert('L')
            label_pil = load_image(f).convert('L')
            label_img = np.array(label_pil)
            # print("clabel: ", np.unique(label_img))
            label_img[label_img==255] = 1
            label_img[label_img==127] = 255
            clabel = Image.fromarray(np.uint8(label_img))

        # image, label = self.co_transform((image, label))
        image, label, elabel, clabel = self.co_transform((image, label, elabel, clabel))
        # image, label, elabel, clabel, image_org = self.co_transform((image, label, elabel, clabel, image_org))
        # image, label, elabel= self.co_transform((image, label, elabel))
        image = self.img_transform(image)
        label = self.label_transform(label)
        clabel = self.label_transform(clabel)
        elabel = self.label_transform(elabel)

        if self.train_phase:
            return image, label, clabel, elabel, image
        else:
            return image, label, clabel, elabel, filename

    def __len__(self):
        return len(self.img_list)


def test():
    from utils.transforms import RandomSizedCrop, IgnoreLabelClass, ToTensorLabel, NormalizeOwn,ZeroPadding, OneHotEncode, RandomSizedCrop3
    from torchvision.transforms import ToTensor,Compose
    import matplotlib.pyplot as plt

    imgtr = [ToTensor(),NormalizeOwn()]
    # sigmoid 
    labtr = [IgnoreLabelClass(),ToTensorLabel(tensor_type=torch.FloatTensor)]
    cotr = [RandomSizedCrop3((512,512))]

    dataset_dir = '/media/data/seg_dataset'
    trainset = Corrosion(home_dir, dataset_dir,img_transform=Compose(imgtr), 
                           label_transform=Compose(labtr),co_transform=Compose(cotr),
                           split=args.split,labeled=True)
    trainloader = DataLoader(trainset_l,batch_size=1,shuffle=True,
                               num_workers=2,drop_last=True)

    for batch_id, (img, mask, _, emask) in enumerate(trainloader):
        img, mask, emask = img.numpy()[0], mask.numpy()[0], emask.numpy()[0]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        ax1.imshow(img)
        ax2.imshow(mask)
        ax3.imshow(emask)
        plt.show()

if __name__ == '__main__':
    test()
