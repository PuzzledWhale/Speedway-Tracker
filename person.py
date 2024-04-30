import numpy as np
from box import Box

class Person:
    def __init__(self, bounding_box, start_time, id, q = 0.1, r = 1.0):
        self.bounding_box = bounding_box # current bounding box this person is associated to
        self.prediction_box = bounding_box
        self.id = id
        self.box_history = [bounding_box]
        self.frame_history = [bounding_box.frame]
        self.color = ''

        # Initialize state transition matrix
        self.state_transition_matrix = np.array([[1, 0, 0, 0, 0, 0],
                                                [0, 1, 0, 0, 0, 0],
                                                [0, 0, 1, 0, 0, 0],
                                                [0, 0, 0, 1, 0, 0],
                                                [0, 0, 0, 0, 1, 0],
                                                [0, 0, 0, 0, 0, 1]])
        
        # Initialize measurement matrix (for direct measurement of position)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]])
        
        # Initialize process noise covariance matrix
        self.Q = q * np.eye(6)
        
        # Initialize measurement noise covariance matrix
        self.R = r * np.eye(2)
        
        # Initialize state and covariance matrix
        self.state = np.array([bounding_box.position[0], bounding_box.position[1], np.sign(320 - bounding_box.position[0]) * 10, 0, 0, 0])  # [x, y, vx, vy, ax, ay]
        self.covariance = np.eye(6) * 1000.0
        self.last_time = start_time
        self.delete = False
        
    def predict(self, new_time):
        dt = new_time - self.last_time
        self.state_transition_matrix = np.array([[1, 0, dt, 0, 0.5*dt**2, 0],
                                                 [0, 1, 0, dt, 0, 0.5*dt**2],
                                                 [0, 0, 1, 0, dt, 0],
                                                 [0, 0, 0, 1, 0, dt],
                                                 [0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 1]])
                                                 
        # Predict new state and covariance
        self.state = np.dot(self.state_transition_matrix, self.x)
        self.covariance = np.dot(np.dot(self.state_transition_matrix, self.covariance), self.state_transition_matrix.T) + self.Q
        self.last_time = new_time
        self.prediction_box = Box(self.bounding_box.frame, x1=self.state[0], y1=self.state[1], size=self.bounding_box.size)

    def update(self, new_box=None):
        if new_box:
            self.bounding_box = new_box
            self.box_history.append(new_box)
            self.frame_history.append(new_box.frame)
        else:
            self.bounding_box = self.prediction_box

        measurement = self.bounding_box.position

        # Kalman gain
        S = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R
        K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))
        
        # Update state estimate
        z = np.array(measurement).reshape(2, 1)
        y = z - np.dot(self.H, self.state)
        self.state = self.state + np.dot(K, y)
        
        # Update covariance matrix
        self.covariance = np.dot((np.eye(6) - np.dot(K, self.H)), self.covariance)
