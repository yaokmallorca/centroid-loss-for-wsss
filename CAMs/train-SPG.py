from update import *
from data import *
from retrain import *
import torch, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import inception_v3
from resnet import resnet_attention
from torchsummary import summary


# functions
CAM             = 1
USE_CUDA        = 1
RESUME          = 66
PRETRAINED      = 0
TRAIN           = 0
INCEPTION       = 0
RESNET          = 1


# hyperparameters
BATCH_SIZE      = 32
IMG_SIZE        = 224
LEARNING_RATE   = 0.001
EPOCH           = 100


# prepare data
normalize = transforms.Normalize(
        mean=[0.46505457, 0.4353015, 0.33521625],
        std=[0.10569176,0.10366041,0.08933138]
)

transform_train = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    # transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

train_data = customData(img_path='/media/data/seg_dataset/corrosion/JPEGImages/', 
                        txt_path='/media/data/seg_dataset/corrosion/label_cls.txt',
                        data_transforms=transform_train)
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = customData(img_path='/media/data/seg_dataset/corrosion/JPEGImages/',
                       txt_path='/media/data/seg_dataset/corrosion/label_cls_test.txt',
                       data_transforms=transform_test)
testloader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=2)


# class
classes = {0:'background', 1: 'corrosion'}


if TRAIN == True:
    # fine tuning
    if INCEPTION:
        if PRETRAINED:
            net = inception_v3(pretrained=PRETRAINED)
            for param in net.parameters():
                param.requires_grad = False
            net.fc = torch.nn.Linear(2048, 2)
        else:
            net = inception_v3(pretrained=PRETRAINED, num_classes=len(classes))
        final_conv = 'Mixed_7c'
        net.cuda()
        # print(summary(net, (3, 224, 224)))
        # input('s')
        # load checkpoint
        # if RESUME != 0:
        #     print("===> Resuming from checkpoint.")
        #     assert os.path.isfile('checkpoint/'+ str(RESUME) + '.pt'), 'Error: no checkpoint found!'
        #     net.load_state_dict(torch.load('checkpoint/' + str(RESUME) + '.pt'))
        # retrain
        criterion = torch.nn.CrossEntropyLoss()
        if PRETRAINED:
            optimizer = torch.optim.SGD(net.fc.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

        for epoch in range (1, EPOCH + 1):
            retrain(trainloader, net, USE_CUDA, epoch, criterion, optimizer)
            retest(testloader, net, USE_CUDA, criterion, epoch, RESUME)

    if RESNET:
        if PRETRAINED:
            model_file = 'model_92_sgd.pkl'
            net = resnet_attention(pretrained=PRETRAINED, model_file=model_file, num_classes=len(classes))
        else:
            net = resnet_attention(pretrained=PRETRAINED, num_classes=len(classes))
        final_conv = 'residual_block6'
        net.cuda()
        # print(summary(net, (3, 224, 224)))
        # load checkpoint
        if RESUME != 0:
            print("===> Resuming from checkpoint.")
            assert os.path.isfile('checkpoint/'+ str(RESUME) + '.pt'), 'Error: no checkpoint found!'
            net.load_state_dict(torch.load('checkpoint/' + str(RESUME) + '.pt'))
        # retrain
        criterion = torch.nn.CrossEntropyLoss()
        if PRETRAINED:
            optimizer = torch.optim.SGD(net.fc.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

        for epoch in range (1, EPOCH + 1):
            retrain(trainloader, net, USE_CUDA, epoch, criterion, optimizer)
            retest(testloader, net, USE_CUDA, criterion, epoch, RESUME)



def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

# CAM
if CAM:
    with open('/media/data/seg_dataset/corrosion/cams_list.txt') as f:
        cams_list = f.readlines()
    if INCEPTION:
        net = inception_v3(pretrained=PRETRAINED, num_classes=len(classes))
        final_conv = 'Mixed_7c'
    if RESNET:
        net = resnet_attention(pretrained=PRETRAINED, num_classes=len(classes))
        final_conv = 'residual_block6'
        prev_conv = 'residual_block5'
        prev_conv4 = 'residual_block4'
    net.eval()
    net.cuda()
    print("===> Resuming from checkpoint.")
    assert os.path.isfile('checkpoint/'+ str(RESUME) + '.pt'), 'Error: no checkpoint found!'
    net.load_state_dict(torch.load('checkpoint/' + str(RESUME) + '.pt'))
    cnt = 0
    for line in cams_list:
    # print(line)
        # line = cams_list[4]\
        # hook the feature extractor
        features_blobs = []
        net._modules.get(final_conv).register_forward_hook(hook_feature)

        img_path = line.strip().split('\t')[0]
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path)
        get_cam(net, features_blobs, img, classes, img_name, final_conv)
