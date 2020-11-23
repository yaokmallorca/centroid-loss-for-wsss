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
RESUME          = 66 # 66
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

train_data = customData(img_path='/media/data/seg_dataset/corrosion/JPEGImages_cls/', 
                        txt_path='/media/data/seg_dataset/corrosion/label_cls.txt',
                        data_transforms=transform_train)
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_data = customData(img_path='/media/data/seg_dataset/corrosion/JPEGImages_cls/',
                       txt_path='/media/data/seg_dataset/corrosion/label_cls_test.txt',
                       data_transforms=transform_test)
testloader = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=2)


# class
classes = {0:'background', 1: 'corrosion'}
OUT_PATH = '/home/yaok/software/ASSS/CAM/result/'
IMG_PATH = '/media/data/seg_dataset/corrosion/JPEGImages'

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
        print(summary(net, (3, 224, 224)))
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
        img_array = np.array(img)
        img_h, img_w, _ = img_array.shape
        heatmap = get_cam(net, features_blobs, img, classes, img_name, 'org')
        # rois = roi_generator(heatmap)
        cnt = 0
        for i in range(2): # h
            for j in range(2): # w
                features_blobs = []
                net._modules.get(final_conv).register_forward_hook(hook_feature)
                img_roi = img_array[i*250:(i+1)*250, j*250:(j+1)*250]
                pil_roi = Image.fromarray(img_roi)
                get_cam_roi(net, features_blobs, pil_roi, img_roi, classes, img_name, 'roi{}'.format(cnt))
                cnt += 1


            # print(roi)
            # xmin, ymin, xmax, ymax = roi[0], roi[1], roi[2], roi[3]
            # cv2.rectangle(img_array, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
            # roi_array = img_array[]
        # cv2.imwrite(OUT_PATH + img_name[0:-4] + '.png', img_array)
        """
        features_blobs = []
        net._modules.get(prev_conv).register_forward_hook(hook_feature)
        get_cam(net, features_blobs, img, classes, img_name, prev_conv)

        features_blobs = []
        net._modules.get(prev_conv4).register_forward_hook(hook_feature)
        # get_cam(net, features_blobs, img, classes, img_name, prev_conv4)
        get_cam_mask(net, features_blobs, img, classes, img_name)
        """
        # get_cam_grabcut(net, features_blobs, img, classes, img_name)
        input('s')