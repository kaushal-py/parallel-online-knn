import numpy as np

class DataGenerator:


    def __init__(self, dim=2, num_points=1000):

        self.dim = dim
        self.num_points = num_points
    

    def generate(self, memory=True, path=None):

        if memory is not True:
            assert path is not None, "Specify path, if you don't want things in memory"
        
            # TODO: Disk arrays
        
        # Memory arrays
        data = np.random.rand(self.dim, self.num_points)
        return data
