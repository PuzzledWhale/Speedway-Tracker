import numpy as np

class KalmanFilter():
    def __init__(self, dt, q = 0.1, r = 1.0, state = np.zeros((6,1))):
        # Initialize state transition matrix
        self.F = np.array([[1, 0, dt, 0, 0.5*dt**2, 0],
                           [0, 1, 0, dt, 0, 0.5*dt**2],
                           [0, 0, 1, 0, dt, 0],
                           [0, 0, 0, 1, 0, dt],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        
        # Initialize measurement matrix (direct measurement of position)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]])
        
        # Initialize process noise covariance matrix
        self.Q = q * np.eye(6)
        
        # Initialize measurement noise covariance matrix
        self.R = r * np.eye(2)
        
        # Initialize state and covariance matrix
        self.x = state  # [x, y, vx, vy, ax, ay]
        self.P = np.eye(6) * 1000.0

    def predict(self):
        # Predict new state and covariance
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, measurement, dt):
        self.F = np.array([[1, 0, dt, 0, 0.5*dt**2, 0],
                           [0, 1, 0, dt, 0, 0.5*dt**2],
                           [0, 0, 1, 0, dt, 0],
                           [0, 0, 0, 1, 0, dt],
                           [0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 1]])
        # Kalman gain
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state estimate
        z = np.array(measurement).reshape(2, 1)
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        
        # Update covariance matrix
        self.P = np.dot((np.eye(6) - np.dot(K, self.H)), self.P)
