import numpy as np

class DataSystem:


    def __init__(self, dim=2, num_points=1000, distribution = 'normal'):

        self.dim = dim
        self.num_points = num_points
        self.distribution = distribution
    

    def generate(self, memory=True, path=None, **kwargs):
        
        # Get the correspoding numpy function for the distribution
        func = self.get_dist_func(self.distribution)
        func_args = self.get_func_args()

        if memory is not True:
            assert path is not None, "Specify path, if you don't want things in memory"
        
            # TODO: Disk arrays


        else:
            # Memory arrays
            data = np.random.rand(self.num_points, self.dim)
            return data
    

    def get_dist_func(self):
        func_dict = {
            'normal': np.random.normal(loc=0.0, scale=1.0),
            'uniform': np.random.uniform(low=0.0, high=1.0),
            'beta': np.random.beta(a=1, b=2),
            'exponential': np.random.exponential(scale=1.0),
            'gamma': np.random.gamma(shape=2.0, scale=1.0),
            'geometric': np.random.geometric(p=0.5),
            'poisson': np.random.poisson(lam=1.0),
        }
        # Get the function from function dictionary
        func = func_dict.get(distribution, lambda: "Invalid distribution")
        # Return the function
        return func
    

    def get_func_args(self):
        arg_dict = {
            'normal': {'loc' : 0.0, 'scale' : 1.0, 'size' : (self.num_points, self.dim)},
            'uniform': {'low' : 0.0, 'high' : 1.0, 'size' : (self.num_points, self.dim)},
            'beta': {'a': 1, 'b' : 2, 'size' : (self.num_points, self.dim)},
            'exponential': {'scale' : 1.0, 'size' : (self.num_points, self.dim)},
            'gamma': {'shape' : 2.0, 'scale' : 1.0, 'size' : (self.num_points, self.dim)},
            'geometric': {'p': 0.5, 'size' : (self.num_points, self.dim)},
            'poisson': {'lam' : 1.0, 'size' : (self.num_points, self.dim)},
        }
        # Get the function from function dictionary
        args = arg_dict.get(distribution, lambda: "Invalid distribution")
        # Return the function
        return args
