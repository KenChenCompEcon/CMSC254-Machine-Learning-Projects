# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 19:06:48 2019

"""

"""
CMSC 25400 - Project "ViolaJones"
Instructor: Prof. Kondor
Author: Ken Chen
"""
import math
import os
import numpy as np
from scipy import misc
from time import time

def load_data(faces_dir, background_dir):
    '''
    faces_dir: string, file location of the faces directory
    background_dir: string, file location of the background directory
    Returns: a tuple of numpy matrices
    - a matrix of size N x 64 x 64 of the images
    - a matrix of size N x 1 of the class labels(+1 for a face, -1 for background)
    '''
    face_names = os.listdir(faces_dir)
    background_names = os.listdir(background_dir)
    img_mat = []; label = []
    for e in face_names:
        data = misc.imread(faces_dir +'/' + e, flatten = True)
        img_mat.append(data)
        label.append(1)
    for e in background_names:
        data = misc.imread(background_dir + '/' + e, flatten = True)
        img_mat.append(data)
        label.append(-1)
    return img_mat, np.array(label, dtype='int32')

def compute_integral_image(imgs):
    '''
    imgs: numpy matrix of size N x 64 x 64, where N = total number of images
    Returns: a matrix of size N x 64 x 64 of the integral image reprsentation
    '''
    for e in imgs:
        e[:] = np.cumsum(e, 1)
        e[:] = np.cumsum(e, 0)
    return imgs

def feature_list_twoRec():
    feat2_lst = []
    for r in range(4,24,4):
        for c in range(4,24,4):
            for i in range(0,64-r,min(4,r//2)):
                for j in range(0,64-c,min(4,c//2)):
                    x_ul = (i,j); x_lr = (i+r, j+c)
                    feat2_lst.append((x_ul, x_lr))
    return feat2_lst

def feature_list_threeRec():
    feat3_lst = []
    r_len = [3,6,9,12]; c_len = [6,12,18,24]
    for r,c in zip(r_len,c_len):
        for i in range(0,64-r,3):
            for j in range(0,64-c,3):
                x_ul = (i,j); x_lr = (i+r, j+c)
                feat3_lst.append((x_ul, x_lr))
    return feat3_lst

def feature_list_fourRec():
    feat4_lst = []
    for l in range(4,24,4):
        for i in range(0,64-l,min(4,l//2)):
            for j in range(0,64-l,min(4,l//2)):
                x_ul = (i,j); x_lr = (i+l, j+l)
                feat4_lst.append((x_ul, x_lr))
    return feat4_lst

def feature_list():
    '''
    Returns: list of relevant pixel locations for each feature, 
    and the number of "two, three, four rectangle" features respectively.
    The ith index of this list should be a list of the two relevant
    pixel locations needed to compute the ith feature
    '''
    f2, f3, f4 = feature_list_twoRec(), feature_list_threeRec(), feature_list_fourRec()
    feat_lst = f2 + f3 + f4
    l2, l3, l4 = len(f2), len(f3), len(f4)
    return feat_lst, l2, l3, l4

def compute_feature(int_img_rep, feat_lst, l2, l3, l4, feat_idx):
    '''
    int_img_rep: the N x 64 x 64 numpy matrix of the integral image representation
    feat_lst: list of features
    feat_idx: integer, index of a feature (in feat_lst)
    Returns: an N x 1 matrix of the feature evaluations for each image
    '''
    feature = []
    u, l = feat_lst[feat_idx]
    if feat_idx < l2:
        h, w = (l[0]-u[0])//2, l[1]-u[1]
        area = h*w
        for img in int_img_rep:
            a,b,c,d,e,f = u,(u[0],l[1]),(l[0]-h,u[1]),(l[0]-h,l[1]),(l[0],u[1]),l
            feature.append((img[f]-img[e]-2*img[d]+2*img[c]+img[b]-img[a])/area)
    elif l2 <= feat_idx < l2 + l3:
        h, w = l[0]-u[0], (l[1]-u[1])//3
        area = h*w
        for img in int_img_rep:
            a,b,c,d,e,f,g,h = \
            u,(l[0],u[1]),(u[0],u[1]+w),(l[0],u[1]+w),(u[0],l[1]-w),(l[0],l[1]-w),(u[0],l[1]),l
            feature.append(0.5*(img[h]-img[g]-3*img[f]+3*img[e]+3*img[d]-3*img[c]-img[b]+img[a])/area)
    else:
        w = (l[0]-u[0])//2
        area = w*w
        for img in int_img_rep:
            a,b,c,d,e,f,g,h,i = u,(u[0],u[1]+w),(u[0],l[1]),(u[0]+w,u[1]),(u[0]+w,u[1]+w),(u[0]+w,l[1]),(l[0],u[1]),(l[0],u[1]+w),l;
            feature.append(0.5*(img[i]-2*img[h]+img[g]-2*img[f]+4*img[e]-2*img[d]+img[c]-2*img[b]+img[a])/area)
    return feature    

def compute_feat_mat(int_img_rep, feat_lst, l2, l3, l4):
    '''
    compute the all the feature values for all the integral images, 
    and sort them at the meantime
    Returns: a #feat_lst x 2 x N matrix. Each row of the matrix contains 
    the feature values for all the images at this feature, and the image index originally
    '''
    feat_mat = []
    for feat_idx in range(len(feat_lst)):
        feature = compute_feature(int_img_rep, feat_lst, l2, l3, l4, feat_idx)
        feature_ord = sorted(enumerate(feature), key = lambda x:x[1])
        feat_mat.append([[e[0] for e in feature_ord],[e[1] for e in feature_ord]])
    return feat_mat

def compute_labels_mat(feat_mat, labels):
    '''
    Return: a #feat_lst x  N matrix. Each row contains the corresponding labels for all 
    the images at that feature.
    '''
    labels_mat = []
    for i in feat_mat:
        labels_mat.append(labels[i[0]])
    return labels_mat       

def opt_p_theta(feat_mat, labels_mat, weights, feat_idx):
    '''
    feat_mat: the #feat_lst x 2 x N matrix
    labels_mat: the #feat_lst x  N matrix
    weights: an N x 1 matrix containing the weights for each datapoint
    feat_idx: integer, index of the feature to compute on all of the images
    Returns: the optimal p and theta values for the given feat_idx and the weighted loss
    '''
    feat_ord = feat_mat[feat_idx]; label_ord = labels_mat[feat_idx]
    thresholds = [float("-inf")] + feat_ord[1]
    weights_ord = np.array(weights)[feat_ord[0]]
    l = len(feat_ord[0])
    pred_l = np.repeat(1,l); pred_r = np.repeat(-1,l)
    loss_l = np.dot(weights_ord, (pred_l-label_ord)/2)
    loss_r = np.dot(weights_ord, (label_ord-pred_r)/2)
    loss_vec_l = np.cumsum(np.append([loss_l], ((label_ord==1)*1-(label_ord==-1)*1)*weights_ord))
    loss_vec_r = np.cumsum(np.append([loss_r], ((label_ord==-1)*1-(label_ord==1)*1)*weights_ord))
    opt_idx_l, opt_idx_r = np.argmin(loss_vec_l), np.argmin(loss_vec_r)
    opt_loss_l, opt_loss_r = loss_vec_l[opt_idx_l], loss_vec_r[opt_idx_r]
    opt_p = (opt_loss_l < opt_loss_r)*1 - (opt_loss_l >= opt_loss_r)*1
    opt_idx = (opt_loss_l < opt_loss_r)*opt_idx_l + (opt_loss_l >= opt_loss_r)*opt_idx_r
    theta = thresholds[opt_idx]
    return theta, opt_p, min(opt_loss_l, opt_loss_r)

def eval_learner(int_img_rep, feat_lst, l2, l3, l4, feat_idx, p, theta):
    '''
    int_img_rep: the N x 64 x 64 numpy matrix of the integral image representation
    feat_lst: list of features
    feat_idx: integer, index of the feature for this weak learner
    p: +1 or -1, polarity
    theta: float, threshold
    Returns: N x 1 vector of label predictions for the given weak learner
    '''
    feat = compute_feature(int_img_rep, feat_lst, l2, l3, l4, feat_idx)
    pred = np.sign(np.array(feat)-theta)*p
    return pred

def error_rate(int_img_rep, labels, feat_lst, l2, l3, l4, weights, feat_idx, p, theta):
    '''
    int_img_rep: the N x 64 x 64 numpy matrix of the integral image representation
    feat_lst: list of features
    weights: an N x 1 matrix containing the weights for each datapoint
    feat_idx: integer, index of the feature for this weak learner
    p: +1 or -1, polarity
    theta: float, threshold
    Returns: the weighted error rate of this weak learner
    '''
    pred = eval_learner(int_img_rep, feat_lst, l2, l3, l4, feat_idx, p, theta)
    error_rate = np.dot((pred != labels), weights)
    return error_rate

def opt_weaklearner(feat_mat, labels_mat, weights):
    '''
    Returns: the i, p, theta, and the weighted loss values for the optimal weak learner
    '''
    opt_idx = -1; opt_p = 1; opt_theta = 0; opt_loss = 1
    for feat_idx in range(len(feat_mat)):
        theta, p, loss = opt_p_theta(feat_mat, labels_mat, weights, feat_idx)
        if loss < opt_loss:
            opt_idx = feat_idx
            opt_p = p
            opt_theta = theta
            opt_loss = loss
    return opt_idx, opt_p, opt_theta, opt_loss
    
def update_weights(weights, error_rate, y_pred, y_true):
    '''
    weights: N x 1 matrix containing the weights of each datapoint
    error_rate: the weighted error rate
    y_pred: N x 1 matrix of predicted labels
    y_true: N x 1 matrix of true labels
    Returns: N x 1 matrix of the updated weights
    '''
    alpha = math.log((1-error_rate)/error_rate)
    weights = weights * np.exp(-alpha * y_true * y_pred/2)
    weights = weights/sum(weights)
    return alpha, weights

def adaboost(int_img_rep, labels, feat_lst, l2, l3, l4, feat_mat, labels_mat):
    lp = int(np.sign(labels+1).sum()/2); ln = len(labels)-lp
    weights = np.append(np.repeat(0.5/lp, lp), np.repeat(0.5/ln, ln))
    pred_ensem = np.zeros(len(labels))
    opt_idx = []; opt_p = []; opt_theta = []; opt_alpha = []
    fpr = 1; i = 0
    while (fpr>0.3) and (i<30):
        i +=1
        idx, p, theta, loss = opt_weaklearner(feat_mat, labels_mat, weights)
        pred = eval_learner(int_img_rep, feat_lst, l2, l3, l4, idx, p, theta)
        error = error_rate(int_img_rep, labels, feat_lst, l2, l3, l4, weights, idx, p, theta)
        alpha, weights = update_weights(weights, error, pred, labels)
        pred_ensem = pred_ensem + alpha*pred
        opt_idx.append(idx); opt_p.append(p); opt_theta.append(theta); opt_alpha.append(alpha)
        false_neg = pred_ensem[(pred_ensem<0)&(labels==1)]
        if len(false_neg)>0: pred_adj = 2*(pred_ensem >= min(false_neg))-1; big_theta = min(false_neg)
        else: pred_adj = 2*(pred_ensem >= 0)-1; big_theta = 0
        fpr = 1*(pred_adj!=labels).sum()/(1*(labels==-1).sum())
        print(idx, p, theta, loss, fpr)
    return opt_idx, opt_p, opt_theta, opt_alpha, pred_adj, fpr, big_theta
    
if __name__=="__main__":
    imgs, labels = load_data("faces","background")
#    imgs = imgs[:50]+imgs[2000:2050]
#    labels = np.append(labels[:50],labels[2000:2050])
    int_img_rep = compute_integral_image(imgs)
    feat_lst,l2,l3,l4 = feature_list()
    f = [[e[0][0], e[0][1], e[1][0], e[1][1]] for e in feat_lst]
    np.savetxt('feat_lst.txt',f,fmt="%i")
    del(f)
    np.savetxt('l2l3l4.txt',[l2,l3,l4],fmt="%i")
    feat_mat = compute_feat_mat(int_img_rep, feat_lst, l2, l3, l4)
    labels_mat = compute_labels_mat(feat_mat, labels)
    idx_dict = {}; p_dict = {}; theta_dict = {}; alpha_dict = {}; big_theta_dict = {}
    count = 0
    while (count<5):
        s = time()
        idx, p, theta, alpha, pred, fpr, big_theta = adaboost(
                int_img_rep, labels, feat_lst, l2, l3, l4, feat_mat, labels_mat)
        idx_dict[count] = idx; p_dict[count] = p; theta_dict[count] = theta
        alpha_dict[count] = alpha; big_theta_dict[count] = big_theta
        print("current false positive rate: "+str(fpr))
        if count==4: print("This cascade took " + str(time() - s) +" seconds")
        if count==4 or fpr==0: break
        fil = (pred >= 0); int_img_rep = list(np.array(int_img_rep)[fil]); labels = labels[fil]
        del(feat_mat); del(labels_mat)
        feat_mat = compute_feat_mat(int_img_rep, feat_lst, l2, l3, l4)
        labels_mat = compute_labels_mat(feat_mat, labels)
        count +=1
        print("This cascade took " + str(time() - s) +" seconds")

    for i in range(count+1):
        np.savetxt("idx" + str(i)+".txt", idx_dict[i], fmt="%i")
        np.savetxt("p" + str(i)+".txt", p_dict[i], fmt="%i")
        np.savetxt("theta" + str(i)+".txt", theta_dict[i])
        np.savetxt("alpha" + str(i)+".txt", alpha_dict[i])
    
    np.savetxt("big_theta.txt", list(big_theta_dict.values()))
    
        