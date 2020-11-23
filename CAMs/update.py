from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2, torch
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
from region_grow import *
import matplotlib.pyplot as plt
import seaborn as sns

# dense CRF
from importlib.machinery import SourceFileLoader
crf = SourceFileLoader("crf", "/home/yaok/software/ASSS/utils/roi_crf.py").load_module()
grab = SourceFileLoader("grab", "/home/yaok/software/ASSS/utils/graphcut.py").load_module()

OUT_PATH = '/home/yaok/software/ASSS/CAM/result/'
IMG_PATH = '/media/data/seg_dataset/corrosion/JPEGImages'
# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    # 1, 2048, 7, 7
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        # print("cam: ", cam.shape)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def one_hot(target, epsilon=0.1, n_classes=2):
    h, w = target.shape
    ohot = np.zeros((n_classes, h, w))
    ohot[1,:,:] = target
    ohot[0,:,:] = 1. - target
    return ohot

def get_cam_roi(net, features_blobs, img_pil, img_roi, classes, root_img, name_ex):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.46505457, 0.4353015, 0.33521625],
        std=[0.10569176,0.10366041,0.08933138]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    # img = cv2.imread(os.path.join(IMG_PATH ,root_img))
    img = img_roi
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    CAM_norm = CAM / (np.max(CAM)-np.min(CAM))
    cv2.imwrite(OUT_PATH + root_img[0:-4] + name_ex + '.png', result)
    return CAM_norm

def get_cam(net, features_blobs, img_pil, classes, root_img, name_ex):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.46505457, 0.4353015, 0.33521625],
        std=[0.10569176,0.10366041,0.08933138]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])
    print("CAMs: ", np.shape(CAMs))

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    img = cv2.imread(os.path.join(IMG_PATH ,root_img))
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    CAM_norm = CAM / (np.max(CAM)-np.min(CAM))
    cv2.imwrite(OUT_PATH + root_img[0:-4] + name_ex + '.png', result)
    return CAM_norm
    
        # cv2.rectangle(result, (x, y), (x+w, y+h), (0,255,0), 2)
    # roi_union = get_union(rois)

    # for i in range(10):
    #     xmin, ymin, xmax, ymax = extend_roi(im_w, im_h, roi_union)
    #     cv2.rectangle(result, (xmin, ymin), (xmax, ymax), (255,0,0), 2)
    
    # cv2.imwrite(OUT_PATH + root_img[0:-4] + '.png', result) # "_" + feature_name + 

def roi_generator(heatmap):
    CAM_sha = np.zeros((heatmap.shape), np.uint8)
    CAM_sha[heatmap >= 0.7] = 1
    print(CAM_sha.shape)
    contours, _ = cv2.findContours(
                    CAM_sha.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rois = []
    im_h, im_w = heatmap.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        rois.append([x, y, x+w, y+h])
    return rois

def extend_roi(im_w, im_h, roi_base, miou=0.3):
    # im_h, im_w = img.shape
    xmin_b, ymin_b, xmax_b, ymax_b = roi_base[0], roi_base[1], roi_base[2], roi_base[3]
    xmin_e = int(xmin_b - np.random.uniform()*xmin_b)
    ymin_e = int(ymin_b - np.random.uniform()*ymin_b)
    xmax_e = int(xmax_b + np.random.uniform()*(im_w - xmax_b))
    ymax_e = int(ymax_b + np.random.uniform()*(im_h - ymax_b))
    return xmin_e, ymin_e, xmax_e, ymax_e

def get_union(rois):
    rois_array = np.array(rois)
    xmin = np.min(rois_array[:,0])
    ymin = np.min(rois_array[:,1])
    xmax = np.max(rois_array[:,2])
    ymax = np.max(rois_array[:,3])
    return [xmin, ymin, xmax, ymax]


def get_cam_mask(net, features_blobs, img_pil, classes, root_img):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.46505457, 0.4353015, 0.33521625],
        std=[0.10569176,0.10366041,0.08933138]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    print('output for the top1 prediction: %s' % classes[idx[0].item()])
    img = cv2.imread(os.path.join(IMG_PATH ,root_img))
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))

    # generate seed cues
    CAM_norm = CAM / (np.max(CAM)-np.min(CAM))
    # seeds = [Point(seed[0], seed[1]) for seed in np.argwhere(CAM_norm > 0.5)]
    # print(seeds)
    # print(img_org.shape)
    # binary_img = regionGrow(img_org, seeds, 10)

    # ax = sns.heatmap(CAM_norm)
    # fig = ax.get_figure()
    # fig.savefig(OUT_PATH + root_img[0:-4] + '_heat.png')
    # fig.clf()
    CAM_norm[CAM_norm > 0.3] = 1.
    CAM_norm[CAM_norm <= 0.3] = 0.

    CAM_onehot = one_hot(CAM_norm)
    cv2.imwrite(osp.join(OUT_PATH, root_img[0:-4]+'_bin.png'), CAM_norm*255)

    # crf_out = np.argmax(crf.img_crf(img, CAM_onehot), axis=0)
    # print(crf_out.shape)
    # cv2.imwrite(osp.join(OUT_PATH, root_img[0:-4]+'_crf.png'), crf_out*255)
    # heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    # result = heatmap * 0.3 + img * 0.5
    # cv2.imwrite(OUT_PATH + root_img, result)

def get_cam_grabcut(net, features_blobs, img_pil, classes, root_img):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.46505457, 0.4353015, 0.33521625],
        std=[0.10569176,0.10366041,0.08933138]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)

    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    print('output for the top1 prediction: %s' % classes[idx[0].item()])
    img = cv2.imread(os.path.join(IMG_PATH ,root_img))
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))

    CAM_norm = CAM / (np.max(CAM)-np.min(CAM))
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, SAL) = saliency.computeSaliency(img)
    OUT = (CAM_norm + SAL)/2.

    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(OUT_PATH + root_img[0:-4]+'.png', result)

    ax = sns.heatmap(OUT)
    plt.savefig(osp.join(OUT_PATH, root_img[0:-4]+'_grab.png'))
    plt.clf()
    CAM_hard = np.zeros(CAM.shape).astype(np.uint8)
    CAM_hard[OUT>=OUT.mean()] = 1
    CAM_hard[OUT<OUT.mean()] = 0
    cv2.imwrite(osp.join(OUT_PATH, root_img[0:-4]+'_bin.png'), CAM_hard*255)
    # generate seed cues
    CAM_onehot = one_hot(CAM_hard)

    """
    CAM_mask = np.zeros(CAM_norm.shape)
    CAM_hard = np.zeros(CAM_norm.shape).astype(np.uint8)
    CAM_hard[CAM_norm>=0.5] = 1
    CAM_hard[CAM_norm<0.5] = 0
    rois = grab.get_rois(CAM_hard)

    CAM_mask[CAM_norm>=0.5] = cv2.GC_FGD
    CAM_mask[(CAM_norm<0.5)&(CAM_norm>=0.2)] = cv2.GC_PR_FGD 
    CAM_mask[CAM_norm<0.1] = cv2.GC_BGD
    CAM_mask[(CAM_norm<0.2)&(CAM_norm>=0.1)] = cv2.GC_PR_BGD
    _, weak_label = grab.graph_cut(img, CAM_mask, rois, 5)
    cv2.imwrite(osp.join(OUT_PATH, root_img[0:-4]+'_bin.png'), weak_label*255)
    CAM_onehot = one_hot(weak_label)
    """

    # rois_crf, rois = crf.get_roi_crf(img, CAM_onehot, CAM_hard.astype(np.uint8))
    # for roi_crf, roi in zip(rois_crf, rois):
    #     # print(roi)
    #     xmin, ymin, xmax, ymax = roi[0], roi[1], roi[2], roi[3]
    #     CAM_hard[ymin:ymax, xmin:xmax] = np.argmax(roi_crf, axis=0).astype(np.uint8)
    crf_out = np.argmax(crf.img_crf(img, CAM_onehot), axis=0)
    cv2.imwrite(osp.join(OUT_PATH, root_img[0:-4]+'_CRF.png'), crf_out*255)
    

    # cv2.imwrite(osp.join(OUT_PATH, root_img[0:-4]+'_wlabel.png'), weak_label*255)
