import numpy as np
import kalman

class Person:
    def __init__(self, bounding_box, ):
        self.bounding_box = bounding_box # current bounding box this person is associated to
        self.kalman = kalman.KalmanFilter(0) # keeps track of the position, velocity, acceleration state of the person
        self.box_history = [bounding_box]
        self.frame_history = []
        self.color = ''
        
        
