import numpy as np

class DataSystem:


    def __init__(self, dim=2, num_points=1000, distribution = 'normal'):
        """
        Parameters
            dim (int)               : Dimension of the data required
            num_points (int)        : Number of the data points required
            distribution (string)   : The distribution from which the generated data will be sampled from
                options:
                    'normal'        : Normal Distribution
                    'uniform'       : Uniform Distribution
                    'beta'          : Beta Distribution
                    'exponential'   : Exponential Distribution
                    'gamma'         : Gamma Distribution
                    'geometric'     : Geometric Distribution
                    'poisson'       : Poisson Distribution
        """
        self.dim = dim
        self.num_points = num_points
        self.distribution = distribution
    

    def generate(self, memory=True, path=None, **kwargs):
        """
        Generates numbers from a specified distribution

            Parameters:
                memory (bool)       : denoted whether the data has to be stored in memory or not
                path (string)       : Path where the data will be stored, if memory = True (Please specify path as .txt)
                **kwargs            : Parameters of the required distribution.

            Returns:
            data                    : Numpy array of size (num_points x dim)

        """
        # Get the correspoding numpy function for the distribution
        func = self.get_dist_func()
        # Get the arguments for the function as a dictionary
        func_args = self.get_func_args()
        # Update the arguments from the **kwargs passed
        func_args.update(pair for pair in kwargs.items() if pair[0] in func_args.keys())

        print(func_args)

        if memory is not True:
            assert path is not None, "Specify path, if you don't want things in memory"
        
            # TODO: Disk arrays
            # Save the array to the given path
            np.savetxt(path, func(**func_args))

        else:
            # Memory arrays
            data = func(**func_args)
            return data
    

    def get_dist_func(self):
        func_dict = {
            'normal': np.random.normal,
            'uniform': np.random.uniform,
            'beta': np.random.beta,
            'exponential': np.random.exponential,
            'gamma': np.random.gamma,
            'geometric': np.random.geometric,
            'poisson': np.random.poisson,
        }
        # Get the function from function dictionary
        func = func_dict.get(self.distribution, lambda: "Invalid distribution")
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
        # Get the arguments from argument dictionary
        args = arg_dict.get(self.distribution, lambda: "Invalid distribution")
        # Return the arguments
        return args

# data = DataSystem(10, 100, 'normal')
# print(data.generate(True, None, {'loc' : 0.0, 'scale' : 5.0, 'size' : (data.num_points, data.dim)}))
