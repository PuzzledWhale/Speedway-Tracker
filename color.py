from ultralytics import YOLO
import cv2
import math 
import torch
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_ciede2000
from scipy.stats import norm
from gaussian_weight import gaussian_weight

def ColorFeature_Tracking(frame,box,waist_height=0.4, shoulder_height=0.8, width_ratio = 0.6, sigma_level=1) :

  
  # divide upper body lower body
  x1, y1, x2, y2 = box.xyxy[0]
  x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
  box_width=x2-x1;
  box_height=y2-y1;

  x1_upper=int(x1+box_width*(1-width_ratio)/2);
  x2_upper=int(x2-box_width*(1-width_ratio)/2);
  y1_upper=int(y2-shoulder_height*box_height);
  y2_upper=int(y2-waist_height*box_height);

  upper_body=frame[y1_upper:y2_upper,x1_upper:x2_upper,:];

  x1_lower=int(x1+box_width*(1-width_ratio)/2);
  x2_lower=int(x2-box_width*(1-width_ratio)/2);
  y1_lower=int(y2-waist_height*box_height);
  y2_lower=y2;

  lower_body=frame[y1_lower:y2_lower,x1_lower:x2_lower,:];

  # (optional) add gaussian weight
  gaussian_upper=gaussian_weight(upper_body)
  gaussian_lower=gaussian_weight(lower_body)

  # reshape to data input
  reshaped_upper_body=upper_body.reshape(-1,3)
  reshaped_lower_body=lower_body.reshape(-1,3)
  reshaped_gaussian_upper=gaussian_upper.reshape(-1)
  reshaped_gaussian_lower=gaussian_lower.reshape(-1)

  #  k-means clustering set k=2 including skin color
  k=2 
  kmeans_upper = KMeans(n_clusters=k, init='k-means++',n_init="auto").fit(reshaped_upper_body,sample_weight=reshaped_gaussian_upper)
  kmeans_lower = KMeans(n_clusters=k, init='k-means++',n_init="auto").fit(reshaped_lower_body,sample_weight=reshaped_gaussian_lower)
  
  color_features=np.zeros((2*k,3))
  color_features[0:2,:]=rgb2lab(kmeans_upper.cluster_centers_/255, channel_axis=1)
  color_features[2:4,:]=rgb2lab(kmeans_lower.cluster_centers_/255, channel_axis=1)
  print('color_features')
  return color_features


def ColorDistance(source_frame,source_boxes,target_frame,target_boxes) :
  # row : source, column : target
  a=source_boxes[0]
  cost_matrix=np.zeros((len(source_boxes),len(target_boxes)))
  i=0;
  j=0;
  for source_box in source_boxes :
    for target_box in target_boxes :
      source_feature =ColorFeature_Tracking(source_frame,source_box)
      target_feature1 =ColorFeature_Tracking(target_frame,target_box)
      target_feature2 =[target_feature1[1],target_feature1[0],target_feature1[2],target_feature1[3]]
      target_feature3 =[target_feature1[0],target_feature1[1],target_feature1[3],target_feature1[2]]
      target_feature4 =[target_feature1[1],target_feature1[0],target_feature1[3],target_feature1[2]]

      dist=[]
      dist.append(np.sum(deltaE_ciede2000(source_feature,target_feature1)))
      dist.append(np.sum(deltaE_ciede2000(source_feature,target_feature2)))
      dist.append(np.sum(deltaE_ciede2000(source_feature,target_feature3)))
      dist.append(np.sum(deltaE_ciede2000(source_feature,target_feature4)))
      print('cost_matrix')
      cost_matrix[i,j]=np.min(dist)
  return cost_matrix



