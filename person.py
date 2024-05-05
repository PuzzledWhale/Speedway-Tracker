import numpy as np
from box import Box

class Person:
    def __init__(self, bounding_box, start_time, id, q = 0.8, r = 0.4):
        self.bounding_box = bounding_box # current bounding box this person is associated to
        self.prediction_box = bounding_box # dummy bounding box that holds state predictions
        self.id = id # id to differentiate identified pedestrians
        self.frame_history = [bounding_box.frame] # array of frames that the person was seen in or associated in
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
        self.state = np.array([bounding_box.position[0], bounding_box.position[1], 0, 0, 0, 0])  # [x, y, vx, vy, ax, ay]
        self.covariance = np.eye(6) * 1000.0

        self.last_time = start_time # holds the last time that the state was changed
        self.delete = False # if True, marks person for deletion
        # print('PERSON', self.id, 'MADE WITH INITIAL STATE', self.state) # debug
        
    def predict(self, new_time):
        # Update state transition matrix with new time difference
        dt = new_time - self.last_time
        self.last_time = new_time
        self.state_transition_matrix = np.array([[1, 0, dt, 0, 0.5*dt**2, 0],
                                                 [0, 1, 0, dt, 0, 0.5*dt**2],
                                                 [0, 0, 1, 0, dt, 0],
                                                 [0, 0, 0, 1, 0, dt],
                                                 [0, 0, 0, 0, 1, 0],
                                                 [0, 0, 0, 0, 0, 1]])
                                                 
        # Predict new state
        prev_state = self.state
        self.state = np.dot(self.state_transition_matrix, self.state.reshape(6, 1)).reshape(1,6)[0]

        # Predict new covariance matrix
        self.covariance = np.dot(np.dot(self.state_transition_matrix, self.covariance), self.state_transition_matrix.T) + self.Q
        

        # Generate new dummy bounding box for predicted state, for use in cost matrix generation
        self.prediction_box = Box(self.bounding_box.frame, 0, x1=self.state[0], y1=self.state[1], size=self.bounding_box.size)

        # print('PERSON', self.id, ' PREDICTED NEW STATE: ', self.state, 'PREVIOUS STATE WAS:', prev_state, self.state[0]) # debug

    def update(self, new_time, new_box):

        self.bounding_box = new_box
        # self.box_history.append(new_box)
        self.frame_history.append(new_box.frame)

        measurement = self.bounding_box.position

        self.last_time = new_time

        # calculate Kalman gain
        S = np.dot(np.dot(self.H, self.covariance), self.H.T) + self.R
        K = np.dot(np.dot(self.covariance, self.H.T), np.linalg.inv(S))
        
        # Update state estimate
        z = np.array(measurement).reshape(2, 1)
        y = z - np.dot(self.H, self.state.reshape(6, 1))
        self.state = self.state + np.dot(K, y).reshape(1, 6)
        # print("IN UPDATE FOR", self.id, 'UPDATED STATE IS', self.state) # for debug

        # Update covariance matrix
        self.covariance = np.dot((np.eye(6) - np.dot(K, self.H)), self.covariance)
