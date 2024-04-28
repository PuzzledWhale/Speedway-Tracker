import numpy as np
import kalman

class Person:
    def __init__(self, bounding_box, ):
        self.bounding_box = bounding_box
        self.kalman = kalman.KalmanFilter(0.1)
        
