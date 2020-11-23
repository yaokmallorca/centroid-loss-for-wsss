from __future__ import unicode_literals
import random
import numpy as np
from collections import OrderedDict
import torch

from datasets.box import BoxSet
import generators.unet as unet
from torchvision import transforms
from torchvision.transforms import ToTensor,Compose
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from torchsummary import summary

from utils.transforms import IgnoreLabelClass, ToTensorLabel, NormalizeOwn,ZeroPadding, OneHotEncode, RandomSizedCrop4, ResizedImage4
from utils.lr_scheduling import poly_lr_scheduler, poly_lr_step_scheduler
from utils.validate import val
from utils.metrics import scores
from utils.Clustering_loss import CentroidsLoss
from functools import reduce

import os
import os.path as osp
import cv2
import argparse
import PIL.Image as Image
import datetime 

# from utils.log import setup_logging, ResultsLog, save_checkpoint, export_args_namespace

Reconstruct = False
ASPP = False
home_dir = os.path.dirname(os.path.realpath(__file__))


DATASET_PATH = '/media/data/seg_dataset/corrosion/JPEGImages'
def parse_args():

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("prefix",
                        help="Prefix to identify current experiment")

    parser.add_argument("dataset_dir", default='/media/data/seg_dataset', 
                        help="A directory containing img (Images) and cls (GT Segmentation) folder")

    parser.add_argument("--mode", choices=('base','label_correction'),default='base',
                        help="base (baseline),label_correction")

    parser.add_argument("--lam_adv",default=0.5,
                        help="Weight for Adversarial loss for Segmentation Network training")

    parser.add_argument("--lam_semi",default=0.1,
                        help="Weight for Semi-supervised loss")

    parser.add_argument("--nogpu",action='store_true',
                        help="Train only on cpus. Helpful for debugging")

    parser.add_argument("--max_epoch",default=300,type=int,
                        help="Maximum iterations.")

    parser.add_argument("--start_epoch",default=0,type=int,
                        help="Resume training from this epoch")

    parser.add_argument("--snapshot", default='snapshots',
                        help="Snapshot to resume training")

    parser.add_argument("--snapshot_dir",default=os.path.join(home_dir,'data','snapshots'),
                        help="Location to store the snapshot")

    parser.add_argument("--batch_size",default=4,type=int, # 10
                        help="Batch size for training")

    parser.add_argument("--val_orig",action='store_true',
                        help="Do Inference on original size image. Otherwise, crop to 320x320 like in training ")

    parser.add_argument("--d_label_smooth",default=0.1,type=float,
                        help="Label smoothing for real images in Seg network")

    parser.add_argument("--no_norm",action='store_true',
                        help="No Normalizaion on the Images")

    parser.add_argument("--init_net",choices=('imagenet','mscoco', 'unet'),default='mscoco',
                        help="Pretrained Net for Segmentation Network")

    parser.add_argument("--g_lr",default=1e-4,type=float, # 1e-4
                        help="lr for generator")

    parser.add_argument("--seed",default=1,type=int,
                        help="Seed for random numbers used in semi-supervised training")

    parser.add_argument("--wait_semi",default=0,type=int,
                        help="Number of Epochs to wait before using semi-supervised loss")

    parser.add_argument("--split",default=1.0,type=float) # 0.5
    # args = parser.parse_args()

    parser.add_argument("--lr_step", default='8000,15000', type=str, 
                        help='Steps for decreasing learning rate')
    args = parser.parse_args()

    return args

'''
    Snapshot the Best Model
'''
def snapshot(model,valoader,epoch,best_miou,snapshot_dir,prefix):
    miou = val(model,valoader,nclass=7)
    snapshot = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'miou': miou
    }
    if miou > best_miou:
        best_miou = miou
        torch.save(snapshot,os.path.join(snapshot_dir,'{}.pth.tar'.format(prefix)))

    print("[{}] Curr mIoU: {:0.4f} Best mIoU: {}".format(epoch,miou,best_miou))
    return best_miou


def make_D_label(label, ignore_mask):
    ignore_mask = np.expand_dims(ignore_mask, axis=1)
    # print("ignore_mask: ", ignore_mask.shape)
    D_label = np.ones(ignore_mask.shape)*label
    # print("D_label: ", D_label.shape)
    D_label[ignore_mask] = 255 # 255
    # print("D_label: ", D_label.shape)
    D_label = Variable(torch.FloatTensor(D_label)).cuda(args.gpu)
    # print("D_label: ", D_label.size())
    return D_label

def one_hot_float(soft_label):
    soft1 = soft_label
    soft0 = 1. - soft_label
    soft0, soft1 = soft0[np.newaxis,:,:], soft1[np.newaxis,:,:]
    one_hot = np.concatenate([soft0, soft1], axis=0)
    return one_hot

def label_smooth(target, epsilon=0.1, n_classes=2):
    batch_size, h, w = target.size()
    one_hot = torch.zeros(batch_size, n_classes, h, w).cuda()
    one_hot.scatter_(1, target.unsqueeze(1), 1)
    # soft_target = target.float()
    return ((1. - epsilon)*one_hot + (epsilon/n_classes))

def denormalizeimage(images, mean=(0., 0., 0.), std=(1., 1., 1.)):
    """Denormalize tensor images with mean and standard deviation.
    Args:
        images (tensor): N*C*H*W
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    images = images.cpu().numpy()
    # N*C*H*W to N*H*W*C
    images = images.transpose((0,2,3,1))
    images *= std
    images += mean
    images *=255.0
    # N*H*W*C to N*C*H*W
    images = images.transpose((0,3,1,2))
    return torch.tensor(images)
'''
    Use PreTrained Model for Initial Weights
'''
def init_weights(model,init_net):
    if init_net == 'imagenet':
        # Pretrain on ImageNet
        inet_weights = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        del inet_weights['fc.weight']
        del inet_weights['fc.bias']
        state = model.state_dict()
        state.update(inet_weights)
        model.load_state_dict(state)
    elif init_net == 'mscoco':
        # TODO: Upload the weights somewhere to use load.url()
        filename = os.path.join(home_dir,'data','MS_DeepLab_resnet_pretrained_COCO_init.pth')
        print(filename)
        assert(os.path.isfile(filename))
        saved_net = torch.load(filename)
        new_state = model.state_dict()
        saved_net = {k.partition('Scale.')[2]: v for i, (k,v) in enumerate(saved_net.items())}
        new_state.update(saved_net)
        model.load_state_dict(new_state)
    elif init_net == 'unet':
        unet.init_weights(model, init_type='normal')


def train_base_box(generator,steps,optimG,trainloader,valoader,args):
    best_miou = -1

    adacen_loss = CentroidsLoss(7, norm=True)
    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()
        #  (img, mask, cmask, _, _, img_org)
        for batch_id, (img, mask, cmask, emask, img_names) in enumerate(trainloader):
            N, _, H, W = img.shape
            if args.nogpu:
                img,mask,cmask,emask = Variable(img),Variable(mask),Variable(cmask),Variable(emask)
            else:
                img,mask,cmask,emask = Variable(img.cuda()),Variable(mask.cuda()),Variable(cmask.cuda()),Variable(emask.cuda())

            if torch.isnan(img).sum() > 0:
                print("img has nan")

            if torch.isnan(mask).sum() > 0:
                print("mask has nan")

            cmask = cmask.unsqueeze(1)
            itr = len(trainloader)*(epoch) + batch_id
            if Reconstruct:
                cpmap, cprec = generator(img)
            else:
                cpmap = generator(img)

            # prob = nn.Softmax2d()(cpmap)
            cprob = nn.LogSoftmax(dim=1)(cpmap)
            soft_pred = nn.Softmax2d()(cpmap)
            
            # print("cprob: ", cprob.size())
            if torch.isnan(cprob).sum() > 0:
                print("cprob has nan")
            Lseg = nn.NLLLoss(ignore_index=255)(cprob,mask)

            if Reconstruct:
                Lrec = nn.MSELoss()(cprec, img)

            for param_group in optimG.param_groups:
                curr_lr = param_group['lr']
            # optimG = poly_lr_scheduler(optimG, args.g_lr, itr)
            optimG = poly_lr_step_scheduler(optimG, curr_lr, itr,steps)

            optimG.zero_grad()
            if Reconstruct:
                (Lseg+Lrec+Lncut).backward()
            else:
                Lseg.backward()
                # (Lseg+Lncut).backward()
                # Lseg.backward()
            optimG.step()

            # print("[{}][{}]Loss: {:0.4f}".format(epoch,itr,Lseg.data[0]))
            if Reconstruct:
                print("[{}][{}][{:.1E}]Loss: {:0.4f}, {:0.4f}, {:0.4f}".format(epoch,itr,curr_lr,Lseg.data,Lrec.data,Lpott.data))
            else:
                print("[{}][{}][{:.1E}]Loss: {:0.4f}".format(epoch,itr,curr_lr,Lseg.data))
        best_miou = snapshot(generator,valoader,epoch,best_miou,best_miou,args.snapshot_dir,args.prefix)
        # best_miou = snapshot(generator,valoader,epoch,best_miou,args.snapshot_dir,args.prefix)

def train_box_centroids(generator,steps,optimG,trainloader,valoader,args):
    best_miou = -1

    ada_loss = CentroidsLoss(7, norm=True)
    nt = datetime.datetime.now()
    log_name = "LOG[INFO]_SCRBOX_{}_{}_{}_{}_{}_{}.log".format(nt.year,nt.month,nt.day,nt.hour,nt.minute,nt.second)
    log_f = open(log_name, "a") 

    for epoch in range(args.start_epoch,args.max_epoch+1):
        generator.train()
        for batch_id, (img, mask, cmask, emask, img_names) in enumerate(trainloader):
            N, _, H, W = img.shape
            if args.nogpu:
                img,mask,cmask,emask = Variable(img),Variable(mask),Variable(cmask),Variable(emask)
            else:
                img,mask,cmask,emask = Variable(img.cuda()),Variable(mask.cuda()),Variable(cmask.cuda()),Variable(emask.cuda())

            cmask = cmask.unsqueeze(1)
            itr = len(trainloader)*(epoch) + batch_id
            if Reconstruct:
                cpmap, cprec = generator(img)
            else:
                cpmap, cpcen = generator(img)

            # print(cpcen.size())
            # prob = nn.Softmax2d()(cpmap)
            cprob = nn.LogSoftmax(dim=1)(cpmap)
            soft_pred = nn.Softmax2d()(cpmap)
            Lseg = nn.NLLLoss(ignore_index=255)(cprob,mask)

            if Reconstruct:
                Lrec = nn.MSELoss()(cprec, img)
            Lcen, Lreg = ada_loss(soft_pred, cpcen, cpmap, cmask) 

            for param_group in optimG.param_groups:
                curr_lr = param_group['lr']
            # optimG = poly_lr_scheduler(optimG, args.g_lr, itr)
            optimG = poly_lr_step_scheduler(optimG, curr_lr, itr, steps)

            optimG.zero_grad()
            if Reconstruct:
                (Lseg+Lrec+Lncut).backward()
            else:
                (Lseg+Lcen+Lreg).backward()
            optimG.step()

            if Reconstruct:
                print("[{}][{}][{:.1E}]Loss: {:0.4f}, {:0.4f}, {:0.4f}".format(epoch,itr,curr_lr,Lseg.data,Lrec.data,Lpott.data))
            else:
                log_str = "[{}][{}][{:.1E}]Loss: {:0.4f}, {:0.7f}, {:0.7f}".format(epoch,itr,curr_lr,Lseg.data,Lcen.data,Lreg.data)
                print(log_str)
                log_f.write(log_str+'\n')
        best_miou, log_curr = snapshot(generator,valoader,epoch,best_miou,args.snapshot_dir,args.prefix)
        log_f.write(log_curr+'\n')
    log_f.close()


def main():

    args = parse_args()

    random.seed(0)
    torch.manual_seed(0)
    if not args.nogpu:
        torch.cuda.manual_seed_all(0)

    if args.no_norm:
        imgtr = [ToTensor()]
    else:
        imgtr = [ToTensor(),NormalizeOwn()]

    if len(args.lr_step) != 0:
        steps = list(map(lambda x: int(x), args.lr_step.split(',')))

    # softmax
    labtr = [IgnoreLabelClass(),ToTensorLabel()]
    cotr = [RandomSizedCrop4((512,512))]

    print("dataset_dir: ", args.dataset_dir)

    trainset_l = BoxSet(home_dir,args.dataset_dir,img_transform=Compose(imgtr), 
                           label_transform=Compose(labtr),co_transform=Compose(cotr),
                           split=args.split,labeled=True, label_correction=True)
    trainloader_l = DataLoader(trainset_l,batch_size=args.batch_size,shuffle=True,
                               num_workers=2,drop_last=True)
    if args.split != 1:
        trainset_u = BoxSet(home_dir,args.dataset_dir,img_transform=Compose(imgtr), 
                               label_transform=Compose(labtr),co_transform=Compose(cotr),
                               split=args.split,labeled=False, label_correction=True)
        trainloader_u = DataLoader(trainset_l,batch_size=args.batch_size,shuffle=True,
                                   num_workers=2,drop_last=True)

    #########################
    # Validation Dataloader #
    ########################
    if args.val_orig:
        if args.no_norm:
            imgtr = [ZeroPadding(),ToTensor()]
        else:
            imgtr = [ZeroPadding(),ToTensor(),NormalizeOwn()]
        # softmax
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        cotr = []
    else:
        if args.no_norm:
            imgtr = [ToTensor()]
        else:
            imgtr = [ToTensor(),NormalizeOwn()]
        # softmax
        labtr = [IgnoreLabelClass(),ToTensorLabel()]
        cotr = [ResizedImage4((512,512))]

    valset = BoxSet(home_dir,args.dataset_dir,img_transform=Compose(imgtr), \
        label_transform = Compose(labtr),co_transform=Compose(cotr),train_phase=False)
    valoader = DataLoader(valset,batch_size=1)

    #############
    # GENERATOR #
    #############
    generator = unet.AttU_Net(output_ch=7, Centroids=False)
    

    if osp.isfile(args.snapshot):
        print("load checkpoint => ", args.snapshot)
        checkpoint = torch.load(args.snapshot)
        generator_dict = generator.state_dict()
        saved_net = {k.partition('module.')[2]: v for i, (k,v) in enumerate(checkpoint['state_dict'].items()) if k.partition('module.')[2] in generator_dict}
        generator_dict.update(saved_net)
        generator.load_state_dict(saved_net)
    else:
        init_weights(generator,args.init_net)

    if args.init_net != 'unet':
        optimG = optim.Adam(filter(lambda p: p.requires_grad, \
            generator.parameters()),args.g_lr, [0.5, 0.999])
    else:
        
        optimG = optim.Adam(filter(lambda p: p.requires_grad, \
            generator.parameters()),args.g_lr, [0.5, 0.999])
        """
        optimG = optim.SGD(filter(lambda p: p.requires_grad, \
            generator.parameters()),lr=args.g_lr,momentum=0.9,\
            weight_decay=0.0001,nesterov=True)
        """
    if not args.nogpu:
        generator = nn.DataParallel(generator).cuda()

    if args.mode == 'base':
        train_base(generator,optimG,trainloader_l,valoader,args)
    elif args.mode == 'label_correction':
        train_box_cluster(generator,steps,optimG,trainloader_l,valoader,args)
    else:
        print("training mode incorrect")

if __name__ == '__main__':
    main()
