import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader,Dataset

def read_data(img_num, txt, idx):
    f = open(txt, 'r')
    IMG_URLs = []
    for i in range(img_num):
        line = f.readline()
        url = line.split()[idx]
        IMG_URLs = np.append(IMG_URLs,url)
    f.close()
    return IMG_URLs

def get_img(img_path):
    img = Image.open(img_path)
    return img_pil


class customData(Dataset):
    def __init__(self, img_path, txt_path,data_transforms=None,): # loader = default_loader, dataset = ''
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_label = [int(line.strip().split('\t')[-1]) for line in lines]
        self.data_transforms = data_transforms
        # self.dataset = dataset
        # self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = Image.open(img_name) # loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms(img) # [self.dataset]
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label
