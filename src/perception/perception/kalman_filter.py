import numpy as np

class KalmanFilter:
    """A simple Kalman filter for smoothing 3D joint positions."""
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2, dt=1.0/30.0):
        # State is [x, y, z, vx, vy, vz]
        self.state_dim = 6
        self.measurement_dim = 3
        
        self.x = np.zeros((self.state_dim, 1))  # Initial state
        self.P = np.eye(self.state_dim) * 500   # Initial uncertainty
        
        # State transition model
        self.F = np.eye(self.state_dim)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        # Measurement function
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * process_variance
        
        # Measurement noise covariance
        self.R = np.eye(self.measurement_dim) * measurement_variance

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(self.state_dim) - K @ self.H) @ self.P

    def process(self, measurement):
        self.predict()
        self.update(measurement)
        return self.x[:3].flatten()