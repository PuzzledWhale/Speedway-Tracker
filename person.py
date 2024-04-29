import numpy as np
from kalman import KalmanFilter
from box import Box

class Person:
    def __init__(self, bounding_box, start_time):
        self.bounding_box = bounding_box # current bounding box this person is associated to
        init_velocity = -20 if bounding_box.position[0] > 320 else 20
        self.kalman = KalmanFilter(start_time=start_time, state=np.array([bounding_box.position[0], bounding_box.position[1], init_velocity, 0, 0, 0])) # keeps track of the position, velocity, acceleration state of the person
        self.box_history = [bounding_box]
        self.frame_history = []
        self.color = ''
        
        
