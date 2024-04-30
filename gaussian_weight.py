import numpy as np
from scipy.stats import norm

def gaussian_weight(frame,sigma_level=1) :
  
  # Standard deviation is half the box width
  sigma_x = sigma_level*frame.shape[1] / 2
  sigma_y = sigma_level*frame.shape[0] / 2

  frame_width=frame.shape[1]
  frame_height=frame.shape[0]  
  # Create a coordinate grid
  x = np.linspace(0, frame_width, frame_width)
  y = np.linspace(0, frame_height, frame_height)
  X, Y = np.meshgrid(x, y)
  
  # Calculate the center of the grid
  x_center = frame_width / 2
  y_center = frame_height / 2
  
  # Compute 2D Gaussian
  gaussian = norm.pdf(X, x_center, sigma_x) * norm.pdf(Y, y_center, sigma_y)
  gaussian=gaussian/np.sum(gaussian)*frame_width*frame_height
  
  return gaussian