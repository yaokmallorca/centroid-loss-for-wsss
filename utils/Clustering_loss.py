import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter
from typing import Optional

class KMeansClusteringLoss(torch.nn.Module):
    def __init__(self, class_num, norm=False, lam=1):
        super(KMeansClusteringLoss,self).__init__()
        self._C = class_num
        self._norm = norm
        self._lambda = lam

    # in_masks: n,1, h,w
    def one_hot_transform(self, in_masks, ignore=False):
        n,_,h,w = in_masks.size()
        if ignore:
            in_masks[in_masks==255] = self._C+1
            one_hot = torch.zeros(n,self._C+1,h,w)
        else:
            one_hot = torch.zeros(n,self._C,h,w)

        in_masks = in_masks.type(torch.LongTensor)
        # print("one_hot_transform: ", one_hot.size(), in_masks.size())
        one_hot = one_hot.scatter_(1, in_masks, 1)
        return one_hot[:,:,:,:]
    
    def compute_centroid(self, in_features, in_masks):
        n, c, h, w = in_features.shape
        one_hot_mask = self.one_hot_transform(in_masks).cuda()
        # print("one hot: ", one_hot_mask.size())
        features = in_features.permute(0,2,3,1).cuda()
        masks = one_hot_mask.permute(0,2,3,1).cuda()
        # print(masks.size())
        # masks[masks==255] = 7

        centroids_batch = torch.zeros(n, self._C, c).type(torch.FloatTensor).cuda()
        for i in range(n):
            centroids = torch.zeros(self._C, c).type(torch.FloatTensor)
            for m in range(0, self._C):
                category_mask = masks[i,:,:,m].type(torch.FloatTensor).unsqueeze(-1)
                feature_input = features[i,:,:,:].type(torch.FloatTensor)
                mul = category_mask*feature_input
                if len(mul[mul.sum(axis=-1)!=0]) > 0:
                    centroids[m] = torch.sum(torch.sum(mul, axis=1),axis=0)/len(mul[mul.sum(axis=-1)!=0])
                else:
                    centroids[m] = torch.sum(torch.sum(mul, axis=1),axis=0)
            centroids_batch[i] = centroids

        if self._norm:
            centroids_batch = torch.sum(centroids_batch, axis=0)/n
        return centroids_batch


    def forward(self, in_features, in_masks):
        assert (in_features.shape[0] == in_masks.shape[0]),"Batch Size Mismatch"
        assert (in_masks.shape[1] == 1),"Mask channel is not 1"
        n,c,h,w = in_features.size()
        preds = torch.argmax(in_features, dim=1).unsqueeze(1).cuda()
        # print(preds.size())
        batch_centroids = self.compute_centroid(in_features, in_masks)
        one_hot_preds = self.one_hot_transform(preds)

        # each of image in batch
        loss = torch.zeros(n,self._C).cuda()
        one_hot_preds = one_hot_preds.permute(0,2,3,1).cuda()
        features = in_features.permute(0,2,3,1).cuda()
        for i in range(n):
            if self._norm:
                img_centroids = batch_centroids
            else:
                img_centroids = batch_centroids[i]
            for m in range(0, self._C):
                centroids = img_centroids[m]
                # back_mask = one_hot_preds[i,:,:,0].type(torch.FloatTensor).unsqueeze(-1)
                pred_mask = one_hot_preds[i,:,:,m].type(torch.FloatTensor).unsqueeze(-1)
                feature_img = features[i,:,:,:].type(torch.FloatTensor)
                pred_features = pred_mask*feature_img
                # back_featrues = back_mask*feature_img
                pred_features_nonzero = pred_features[pred_features.sum(axis=-1)!=0].cuda()
                # back_featrues_nonzero = back_featrues[back_featrues.sum(axis=-1)!=0].cuda()
                # loss[i][m] = (pred_features_nonzero-centroids).norm(2,dim=0).mean()
                if len(pred_features_nonzero) > 0:
                    loss[i][m] += torch.sum((pred_features_nonzero-centroids).norm(2,dim=0))/len(pred_features_nonzero)
                    # loss[i][m] += torch.sum((pred_features_nonzero-centroids).norm(2,dim=0))/len(pred_features_nonzero)
                    # loss[i][m] += torch.sum((back_featrues_nonzero-centroids).norm(2,dim=0))/len(back_featrues_nonzero)

        # print("loss: ", loss.size())
        loss_v = loss.sum(axis=-1)/(self._C)
        # loss_v = loss_v / (loss_v.max()-loss_v.min())
        return self._lambda * loss_v.mean()

class AdaptiveCentroidsLoss_ce(torch.nn.Module):
    def __init__(self, class_num, norm=False, lam=1, threshold=0.6):
        super(AdaptiveCentroidsLoss,self).__init__()
        self._C = class_num
        self._norm = norm
        self._lambda = lam
        self._threshold = threshold
        self._criterion_cen = nn.NLLLoss()
        self._criterion_pre = nn.NLLLoss()

    # in_masks: n,1, h,w
    def one_hot_transform(self, in_masks, ignore=False):
        n,_,h,w = in_masks.size()
        if ignore:
            in_masks[in_masks==255] = self._C
            one_hot = torch.zeros(n,self._C+1,h,w)
        else:
            one_hot = torch.zeros(n,self._C,h,w)

        in_masks = in_masks.type(torch.LongTensor)
        # print("one_hot_transform: ", one_hot.size(), in_masks.size())
        one_hot = one_hot.scatter_(1, in_masks, 1)
        if ignore:
            return one_hot[:,0:-1,:,:]
        else: 
            return one_hot[:,:,:,:]

    """
    input Parameter:
    in_prob: softmask probability
    centroids: (batch_size, category)
    in_features: img RGB
    in_masks: scribble ground truth
    """
    def forward(self, in_prob, in_centroids, in_features=None, in_masks=None):
        try:
            n,c,h,w = in_features.size()
        except:
            in_features = in_prob
            n,c,h,w = in_prob.size()
        # print("in_centroids: ", in_centroids)
        centroids = in_centroids.cpu()
        centroids = centroids.view(n, -1, c)
        soft_pred = nn.Softmax2d()(in_prob)
        preds = torch.argmax(soft_pred, dim=1).unsqueeze(1) # .cuda() in_prob
        # print(preds.size())
        # batch_centroids = self.compute_centroid(in_features, in_masks)
        one_hot_preds = self.one_hot_transform(preds)
        one_hot_masks = self.one_hot_transform(in_masks, ignore=True)

        # each of image in batch
        loss_cent = torch.zeros(n,self._C) # .cuda()
        loss_pred = torch.zeros(n,self._C) # .cuda()
        # loss_mse = torch.zeros(n,self._C).cuda()
        one_hot_preds = one_hot_preds.permute(0,2,3,1)
        one_hot_masks = one_hot_masks.permute(0,2,3,1)
        features = in_features.permute(0,2,3,1)

        for i in range(n):
            feature_img = features[i,:,:,:].type(torch.FloatTensor)
            for m in range(0, self._C):
                # centroids = batch_centroids[m]
                pred_mask = one_hot_preds[i,:,:,m].type(torch.FloatTensor).unsqueeze(-1)
                gt_mask   = one_hot_masks[i,:,:,m].type(torch.FloatTensor).unsqueeze(-1)
                # feature_img = features[i,:,:,:].type(torch.FloatTensor)
                pred_features = pred_mask*feature_img # .detach()
                gt_features   = gt_mask*feature_img # .detach()

                pred_f = pred_features[pred_features.sum(axis=-1)!=0]
                gt_f   = gt_features[gt_features.sum(axis=-1)!=0]
                # print(len(pred_mask[pred_mask == 1]), len(gt_mask[gt_mask == 1]))
                if (len(pred_mask[pred_mask == 1]) > 0) and (len(gt_mask[gt_mask == 1]) > 0):
                    dist_cen = torch.zeros((len(gt_f), self._C))
                    # dist_reg = torch.zeros((len(pred_f), self._C))
                    t0 = torch.zeros(len(gt_f)).fill_(m).type(torch.LongTensor)
                    # t1 = torch.zeros(len(pred_f)).fill_(m).type(torch.LongTensor)
                    # print(centroids[i][0].size(),gt_f.size(), pred_f.size())
                    dist_cen[:,0] = torch.sum(-(gt_f-centroids[i][0])**2, dim=1)
                    dist_cen[:,1] = torch.sum(-(gt_f-centroids[i][1])**2, dim=1)
                    dist_cen[:,2] = torch.sum(-(gt_f-centroids[i][2])**2, dim=1)
                    dist_cen[:,3] = torch.sum(-(gt_f-centroids[i][3])**2, dim=1)
                    dist_cen[:,4] = torch.sum(-(gt_f-centroids[i][4])**2, dim=1)
                    dist_cen[:,5] = torch.sum(-(gt_f-centroids[i][5])**2, dim=1)
                    dist_cen[:,6] = torch.sum(-(gt_f-centroids[i][6])**2, dim=1)
                    log_dist_cen = nn.LogSoftmax(dim=1)(dist_cen)
                    loss_cent[i][m] = self._criterion_cen(log_dist_cen, t0)

                    cen = centroids[i].clone().detach()
                    dist_reg[:,0] = torch.sum(-(pred_f-centroids[i][0])**2, dim=1)
                    dist_reg[:,1] = torch.sum(-(pred_f-centroids[i][1])**2, dim=1)
                    log_dist_reg = nn.LogSoftmax(dim=1)(dist_reg)
                    loss_pred[i][m] = self._criterion_pre(log_dist_reg, t1)
                    # print(loss_cent[i][m], loss_pred[i][m])
                    # input('s')

        # print("loss: ", loss.size())
        loss_cent_v = loss_cent.sum()/(n*self._C)
        loss_pred_v = loss_pred.sum()/(n*self._C)
        # loss_v = loss_v / (loss_v.max()-loss_v.min())
        return loss_cent_v.cuda(), loss_pred_v.cuda()
        # return loss_cent_v.cuda()

class CentroidsLoss(torch.nn.Module):
    def __init__(self, class_num, lam0=1, lam1=1):
        super(CentroidsLoss,self).__init__()
        self._C = class_num
        self._lambda0 = lam0
	self._lambda1 = lam1
        self._criterion_cen = nn.NLLLoss()

    # in_masks: n,1, h,w
    def one_hot_transform(self, in_masks, ignore=False, ):
        n,_,h,w = in_masks.size()
        if ignore:
            in_masks[in_masks==255] = self._C
            one_hot = torch.zeros(n,self._C+1,h,w)
        else:
            one_hot = torch.zeros(n,self._C,h,w)

        in_masks = in_masks.type(torch.LongTensor)
        # print("one_hot_transform: ", one_hot.size(), in_masks.size())
        one_hot = one_hot.scatter_(1, in_masks, 1)
        return one_hot[:,:,:,:]

    """
    input Parameter:
    in_prob: softmask probability
    centroids: predicted centroids
    in_features: input features
    in_masks: scribble ground truth
    """
    def forward(self, in_prob, in_centroids, in_features=None, in_masks=None):
        try:
            n,c,h,w = in_features.size()
        except:
            in_features = in_prob
            n,c,h,w = in_prob.size()
        # print("in_centroids: ", in_centroids)
        centroids = in_centroids.cpu()
        centroids = centroids.view(n, -1, c)
        preds = torch.argmax(in_prob, dim=1).unsqueeze(1)
        one_hot_preds = self.one_hot_transform(preds)
        one_hot_masks = self.one_hot_transform(in_masks, ignore=True)

        loss_cent = torch.zeros(n,self._C)
        loss_pred = torch.zeros(n,self._C)
        one_hot_preds = one_hot_preds.permute(0,2,3,1)
        one_hot_masks = one_hot_masks.permute(0,2,3,1)
        features = in_features.permute(0,2,3,1)

        for i in range(n):
            feature_img = features[i,:,:,:].type(torch.FloatTensor)
            for m in range(0, self._C):
                pred_mask = one_hot_preds[i,:,:,m].type(torch.FloatTensor).unsqueeze(-1)
                gt_mask   = one_hot_masks[i,:,:,m].type(torch.FloatTensor).unsqueeze(-1)
                pred_features = pred_mask*feature_img 
                gt_features   = gt_mask*feature_img

                pred_f = pred_features[pred_features.sum(axis=-1)!=0]
                gt_f   = gt_features[gt_features.sum(axis=-1)!=0]
                if (len(pred_mask[pred_mask == 1]) > 0) and (len(gt_mask[gt_mask == 1]) > 0):
                    dist_cen = torch.zeros((len(gt_f), self._C))
                    loss_reg = torch.zeros((len(pred_f), self._C))
                    t0 = torch.zeros(len(gt_f)).fill_(m).type(torch.LongTensor)
                    # print(centroids[i][0].size(),gt_f.size(), pred_f.size())
		    for c in range(0, self.C):
                    	dist_cen[:,c] = torch.sum(-(gt_f-centroids[i][c])**2, dim=1)
  
                    log_dist_cen = nn.LogSoftmax(dim=1)(dist_cen)
                    loss_cent[i][m] = self._criterion_cen(log_dist_cen, t0)

                    cen = centroids[i].clone().detach()
                    t1 = torch.zeros(pred_f.size()).type(pred_f.type())
                    t1[:] = cen[m]
                    loss_pred[i][m] = self._criterion_pre(pred_f, t1)
		    	   
        loss_cent_v = self.lam0*loss_cent.sum()/(n*self._C)
        loss_pred_v = self.lam1*loss_pred.sum()/(n*self._C)
        return loss_cent_v.cuda(), loss_pred_v.cuda()



