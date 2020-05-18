import numpy as np

class DataSystem:


    def __init__(self, dim=2, num_points=1000):

        self.dim = dim
        self.num_points = num_points
    

    def generate(self, memory=True, path=None, distribution='normal', **kwargs):

        if memory is not True:
            assert path is not None, "Specify path, if you don't want things in memory"
        
            # TODO: Disk arrays


        else:
            # Memory arrays
            data = np.random.rand(self.num_points, self.dim)
            return data
