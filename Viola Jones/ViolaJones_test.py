# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 11:53:26 2019

@author: Ken CHEN
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from scipy import misc


def generate_coor(step_w, step_h, w, h):
    lower_left_coor = np.array([[(i, j) for j in range(4, w-63, step_w)] 
                        for i in range(4, h-63, step_h)])
    return lower_left_coor

def compute_integral_image(img):
    img = np.cumsum(np.cumsum(img, 1), 0)
    return img

def compute_feature(int_img_rep, feat_lst, l2, l3, l4, feat_idx):
    u, l = feat_lst[feat_idx]
    img = int_img_rep
    if feat_idx < l2:
        h, w = (l[0]-u[0])//2, l[1]-u[1]
        area = h*w
        a,b,c,d,e,f = u,(u[0],l[1]),(l[0]-h,u[1]),(l[0]-h,l[1]),(l[0],u[1]),l
        feature = (img[f]-img[e]-2*img[d]+2*img[c]+img[b]-img[a])/area
    elif l2 <= feat_idx < l2 + l3:
        h, w = l[0]-u[0], (l[1]-u[1])//3
        area = h*w
        a,b,c,d,e,f,g,h = \
        u,(l[0],u[1]),(u[0],u[1]+w),(l[0],u[1]+w),(u[0],l[1]-w),(l[0],l[1]-w),(u[0],l[1]),l
        feature = (0.5*(img[h]-img[g]-3*img[f]+3*img[e]+3*img[d]-3*img[c]-img[b]+img[a])/area)
    else:
        w = (l[0]-u[0])//2
        area = w*w
        a,b,c,d,e,f,g,h,i = u,(u[0],u[1]+w),(u[0],l[1]),(u[0]+w,u[1]),(u[0]+w,u[1]+w),(u[0]+w,l[1]),(l[0],u[1]),(l[0],u[1]+w),l;
        feature = (0.5*(img[i]-2*img[h]+img[g]-2*img[f]+4*img[e]-2*img[d]+img[c]-2*img[b]+img[a])/area)
    return feature    

def eval_learner(int_img_rep, feat_lst, l2, l3, l4, feat_idx, p, theta):
    feat = compute_feature(int_img_rep, feat_lst, l2, l3, l4, feat_idx)
    pred = np.sign(np.array(feat)-theta)*p
    return pred

def predict(int_img_rep, cascades, feat_lst, l2, l3, l4):
    for i in range(len(cascades)):
        cascade = cascades[i]
        l = len(cascade['idx'])
        alpha = cascade['alpha']; big_theta = cascade['big_theta']
        idx = cascade['idx']; p = cascade['p']; theta = cascade['theta']
        preds = np.array([eval_learner(int_img_rep, feat_lst, l2, l3, l4, idx[j], p[j], theta[j])
                    for j in range(l)])
        pred = np.sign(np.dot(preds, alpha)-2.5*big_theta)
        if pred == -1: break
    return pred

if __name__=="__main__":
    
    # load the picture data
    test_img = misc.imread("test_img.jpg")
    test_h = test_img.shape[0]; test_w = test_img.shape[1]
    ll_coor = generate_coor(32, 32, test_w, test_h)
    
    # load the cascades: stored as a nested dictionary
    big_theta = np.loadtxt("big_theta.txt")
    count = len(big_theta)
    cascades = {}
    for i in range(count):
        cascade = cascades[i] = {}
        cascades[i]['big_theta'] = big_theta[i]

    for i in range(count):
        cascade = cascades[i]
        cascade['idx'] = np.loadtxt("idx" + str(i)+".txt", dtype=int)
        cascade['p'] = np.loadtxt("p" + str(i)+".txt", dtype=int)
        cascade['theta'] = np.loadtxt("theta" + str(i)+".txt")
        cascade['alpha'] = np.loadtxt("alpha" + str(i)+".txt")
        for key in ['idx', 'p', 'theta', 'alpha']:
            try:
                len(cascade[key])
            except:
                cascade[key] = np.array([cascade[key]])
    
        
    # load the feature lists
    feat_lst_temp = np.loadtxt("feat_lst.txt",  dtype = int)
    feat_lst = []
    for feat in feat_lst_temp:
        feat_lst.append([(feat[i],feat[i+1]) for i in range(0,4,2)])
    
    f = np.loadtxt("l2l3l4.txt", dtype=int)
    l2, l3, l4 = f[0],f[1],f[2]
    
    faces_ll_coor = []
    i = 0
    while i < ll_coor.shape[0]:
        j = 0
        while j < ll_coor.shape[1]:
            coor = ll_coor[i,j]
            img = test_img[coor[0]:coor[0]+64, coor[1]:coor[1]+64]
            int_img_rep = compute_integral_image(img)
            pred = predict(int_img_rep, cascades, feat_lst, l2, l3, l4)
            if pred==1:
                faces_ll_coor.append(coor)
                j +=2; # skip a square, so there's no overlap by row
            j +=1;
        i+=1;

    plt.gray()
    plt.figure(figsize=(64, 48))
    fig,ax = plt.subplots(1)
    ax.imshow(test_img)
    for coor in faces_ll_coor:
        x = coor[1]; y = coor[0]
        rect = patches.Rectangle((x,y),64,64,linewidth=0.5,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()