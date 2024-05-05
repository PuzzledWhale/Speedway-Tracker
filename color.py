from ultralytics import YOLO
import cv2
import math 
import torch
import numpy as np
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, deltaE_ciede2000
from scipy.stats import norm
from gaussian_weight import gaussian_weight



def ColorFeature(frame,box,waist_height=0.4, shoulder_height=0.8, width_ratio = 0.6, sigma_level=1) :

  
  # divide bounding box into upper body and lower body

  x1, y1, x2, y2 = box.get_corners()
  x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
  box_width=x2-x1;
  box_height=y1-y2;
  
  

  x1_upper=int(x1+box_width*(1-width_ratio)/2);
  x2_upper=int(x2-box_width*(1-width_ratio)/2);
  y1_upper=int(y1-shoulder_height*box_height);
  y2_upper=int(y1-waist_height*box_height);

  upper_body=frame[y1_upper:y2_upper,x1_upper:x2_upper,:];

  x1_lower=int(x1+box_width*(1-width_ratio)/2);
  x2_lower=int(x2-box_width*(1-width_ratio)/2);
  y1_lower=int(y1-waist_height*box_height);
  y2_lower=y1;

  lower_body=frame[y1_lower:y2_lower,x1_lower:x2_lower,:];

  #  add gaussian weight
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
  

  # convert rgb color feature into CIE L*a*b* color feature 
  color_features=np.zeros((2*k,3))
  color_features[0:2,:]=rgb2lab(kmeans_upper.cluster_centers_/255, channel_axis=1)
  color_features[2:4,:]=rgb2lab(kmeans_lower.cluster_centers_/255, channel_axis=1)

  return color_features


def ColorDistance(frame,boxes,people) :
  # row : source(frame t+1), column : target(frame t)

  # initialize cost matrix
  cost_matrix=np.zeros((len(boxes),len(people)))
  
  for i in range(len(boxes)) :

    # obtain color feature of new bounding box (frame t+1)
    source_feature =ColorFeature(frame,boxes[i])
    print('box_'+str(i)+':',source_feature)
    for j in range(len(people)) :
      
      # load color feature of existing persons (frame t)
      target_feature1 =people[j].color

      # shuffle color feature to match color feature pair
      target_feature2 =[target_feature1[1],target_feature1[0],target_feature1[2],target_feature1[3]]
      target_feature3 =[target_feature1[0],target_feature1[1],target_feature1[3],target_feature1[2]]
      target_feature4 =[target_feature1[1],target_feature1[0],target_feature1[3],target_feature1[2]]
      if i==0 :
        print('person_'+str(j)+':',target_feature1)
      
      # gather possible combination of color feature pairs
      dist=[]
      dist.append(np.sum(deltaE_ciede2000(source_feature,target_feature1)))
      dist.append(np.sum(deltaE_ciede2000(source_feature,target_feature2)))
      dist.append(np.sum(deltaE_ciede2000(source_feature,target_feature3)))
      dist.append(np.sum(deltaE_ciede2000(source_feature,target_feature4)))

      #obtain minimum distance among possible combinations & store them on cost matrix
      cost_matrix[i,j]=np.min(dist)
      
  return cost_matrix



