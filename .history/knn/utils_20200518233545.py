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
    
    def get_dist_func(distribution):
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
        # Get the function from switcher dictionary
        func = func_dict.get(argument, lambda: "Invalid distribution")
        # Execute the function
        return func
