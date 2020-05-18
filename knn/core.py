import numpy as np
from .utils import DataSystem

class BaseKNN:

    def __init__(self, k=1, data=None, distance=2):
        self.k = k
        self.data = data

        assert self.data is not None, "Data not found!"

        """Set distance definition: 1 - L1, 2 - L2"""
        if distance == 1:
            self.distance = np.abs     # absolute value
        elif distance == 2:
            self.distance = np.square  # square root
        else:
            raise Exception("Distance not defined.")    

    def predict(self, x):
        # TODO: Extend this abstract method
        raise NotImplementedError


class VectorKNN(BaseKNN):

    def __init__(self, k, data, distance=2):
        super().__init__(k, data, distance)
    
    def predict(self, x):
        distances = np.sum(self.distance(self.data- x), axis=1)
        min_index = np.argmin(distances)
        return self.data[min_index]
    
    
class LoopKNN(BaseKNN):

    def __init__(self, k, data, distance=2):
        super().__init__(k, data, distance)
    
    def predict(self, x):
        min_distance = float('inf')
        min_index = 0
        for idx, datapoint in enumerate(data):
            current_distance = self.distance(datapoint-x)
            if current_distance < min_distance:
                min_index = idx
                min_distance = current_distance
        return self.data[min_index]

