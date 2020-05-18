import numpy as np

class DataSystem:


    def __init__(self, dim=2, num_points=1000, distribution = 'normal'):

        self.dim = dim
        self.num_points = num_points
        seld.distribution = distribution
    

    def generate(self, memory=True, path=None, **kwargs):
        
        func = self.get_dist_func(self.distribution)
        if memory is not True:
            assert path is not None, "Specify path, if you don't want things in memory"
        
            # TODO: Disk arrays


        else:
            # Memory arrays
            data = np.random.rand(self.num_points, self.dim)
            return data
    

    def get_dist_func(self, distribution):
        func_dict = {
            'normal': np.random.normal,
            'uniform': np.random.uniform,
            'beta': np.random.beta,
            'exponential': np.random.exponential,
            'gamma': np.random.gamma,
            'geometric': np.random.geometric,
            'multinoamial': np.random.multinomial,
            'poisson': np.random.poisson,
        }
        # Get the function from function dictionary
        func = func_dict.get(distribution, lambda: "Invalid distribution")
        # Return the function
        return func
