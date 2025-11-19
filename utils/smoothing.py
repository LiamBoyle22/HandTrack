import numpy as np

class PositionalSmoother:
    
    def __init__(self, smoothing = 0.5):
        self.smoothing = smoothing
        self.smoothed_x = None
        self.smoothed_y = None

    def smooth(self, x, y):
        if self.smoothed_x is None:
            self.smoothed_x = x
            self.smoothed_y = y
        
        else:
            self.smoothed_x = self.smoothing * x + (1 - self.smoothing) * self.smoothed_x
            self.smoothed_y = self.smoothing * y + (1 - self.smoothing) * self.smoothed_y

        return (self.smoothed_x, self.smoothed_y)
    
    def reset(self):
        self.smoothed_x = None
        self.smoothed_y = None

class VelocityLimiter:

    def __init__(self, max_speed = 100):
        self.max_speed = max_speed
        self.prev_x = None
        self.prev_y = None

    def limit(self, x, y):
        
        if self.prev_x is None:
            self.prev_x = x
            self.prev_y = y
            return (x, y)
        
        dx = x -self.prev_x
        dy = y - self.prev_y
        distance = np.sqrt(dx * dx + dy * dy)

        if distance > self.max_speed:
            scale = self.max_speed / distance
            dx *= scale
            dy *= scale
            x = self.prev_x + dx
            y = self.prev_y + dy
        
        self.prev_x = x
        self.prev_y = y
        return (x, y)
    
    def reset(self):
        self.prev_x = None
        self.prev_y = None

ExponentialMovingAverage = PositionalSmoother
#KalmanFilter1D = None 