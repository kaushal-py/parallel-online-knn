from mpi4py import MPI
import numpy as np
from utils import DataSystem, Data_Generator
# from core import BaseKNN
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# class _KDNode:

#     def __init__(self, dimensions, split_axis, data):
#         self.split_axis = split_axis
#         self.dimensions = dimensions
#         self.right_child = None
#         self.left_child = None
#         self.data = data
#         self.split_point = None

#     def split(self, level=0):

#         if level > 0:
#             self.split_point = np.median(self.data[:,self.split_axis])
#             child_axis = (self.split_axis+1)%self.dimensions
#             left_data = self.data[self.data[:,self.split_axis] <= self.split_point]
#             right_data = self.data[self.data[:,self.split_axis] > self.split_point]

#             self.right_child = _KDNode(self.dimensions, child_axis, right_data)
#             self.left_child = _KDNode(self.dimensions, child_axis, left_data)

#             self.data = None
#             self.left_child.split(level-1)
#             self.right_child.split(level-1)

#     def pre_order(self):

#         if self.split_point is not None:
#             print(self.split_axis, self.split_point)
#             self.left_child.pre_order()
#             self.right_child.pre_order()


# class KDTreeKNN(BaseKNN):

#     def __init__(self, k, data, distance=2):

#         super(KDTreeKNN, self).__init__(k, data, distance)
#         dimensions = self.data.shape[1]
#         self.tree = _KDNode(dimensions, 0, self.data)
#         self.tree.split(level=5)
#         # self.tree.pre_order()

#     def predict(self, x):

#         node = self.tree
#         while node.split_point is not None:
#             if x[node.split_axis] <= node.split_point:
#                 node = node.left_child
#             else:
#                 node = node.right_child

#         distances = np.sum(self.distance(node.data- x), axis=1)
#         min_index = np.argmin(distances)
#         return node.data[min_index]

#     def add_batch(self, new_batch):

#         for x in new_batch:

#             node = self.tree
#             while node.split_point is not None:
#                 if x[node.split_axis] <= node.split_point:
#                     node = node.left_child
#                 else:
#                     node = node.right_child

#             node.data = np.concatenate(node.data, [x])

def tree_plot():#self, ax, level=None):
    # fig, ax = plt.subplots(1)
    plt.axes()
    rect = patches.Rectangle((50,100), 10, 100, edgecolor='r', facecolor='none')
    rec = plt.Rectangle((10,10), 50, 20, fc='none',ec="blue")
    rec1 = plt.Rectangle((20,20), 25, 10, fc='none',ec="blue")
    plt.gca().add_patch(rec)
    plt.gca().add_patch(rec1)
    plt.axis('scaled')
    # ax.add_patch(rect)
    plt.show()

# tree_plot()

data_class = Data_Generator('constant_high', 100, 10, 2, 1000, 'gamma')
count = 0
for data in data_class.generator():#, loc = 0.0, scale = 1.0, size = (data_class.num_points, data_class.dim)):
    # print(data.shape)
    print(data.min(0), data.max(0))
    count+=1
    # if (count==10):
    #     print(data)
print('final count is ', count)
