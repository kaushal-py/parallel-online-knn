from mpi4py import MPI
import numpy as np
from .utils import DataSystem

class BaseKNN(object):

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

    def add_batch(self, new_batch):
        # TODO: Extend this abstract method
        raise NotImplementedError


class VectorKNN(BaseKNN):

    def __init__(self, k, data, distance=2):
        super(VectorKNN, self).__init__(k, data, distance)

    def predict(self, x):
        distances = np.sum(self.distance(self.data- x), axis=1)
        min_index = np.argmin(distances)
        return self.data[min_index]

    def add_batch(self, new_batch):
        self.data = np.concatenate((self.data, new_batch))

class ParallelVectorKNN(BaseKNN):

    def __init__(self, k, data, distance=2):
        super(ParallelVectorKNN, self).__init__(k, data, distance)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()
        self.num_points = data.shape[0]

    def predict(self, x):
        chunk_size = self.num_points // self.num_procs
        if self.rank != self.num_procs-1:
            my_data = self.data[self.rank*chunk_size:(self.rank+1)*chunk_size]
        else:
            my_data = self.data[self.rank*chunk_size:]
        local_distances = np.sum(self.distance(my_data-x), axis=1)
        min_index = np.argmin(local_distances)
        local_min = my_data[min_index]

        # Gather all data in process 0
        recv_buf = None
        if self.rank==0:
            recv_buf = np.empty((self.num_procs, self.data.shape[1]))
        self.comm.Gather(local_min, recv_buf, root=0)
        if self.rank==0:
            global_distances = np.sum(self.distance(recv_buf-x), axis=1)
            min_index = np.argmin(global_distances)
            return recv_buf[min_index]

    def add_batch(self, new_batch):
        self.data = np.concatenate((self.data, new_batch))


class LoopKNN(BaseKNN):

    def __init__(self, k, data, distance=2):
        super(LoopKNN, self).__init__(k, data, distance)

    def predict(self, x):
        min_distance = float('inf')
        min_index = 0
        for idx, datapoint in enumerate(self.data):
            current_distance = np.sum(self.distance(self.data[idx]- x))
            if current_distance < min_distance:
                min_index = idx
                min_distance = current_distance
        return self.data[min_index]

    def add_batch(self, new_batch):
        self.data = np.concatenate((self.data, new_batch))

class _KDNode:

    def __init__(self, dimensions, split_axis, data):
        self.split_axis = split_axis
        self.dimensions = dimensions
        self.right_child = None
        self.left_child = None
        self.data = data
        self.datalength = self.data.shape[0]
        self.split_point = None
        self.level = 0

    def split(self, level=1):

        if level > 0:
            self.split_point = np.median(self.data[:,self.split_axis])
            self.level = level
            self.datalength = self.data.shape[0]
            child_axis = (self.split_axis+1)%self.dimensions
            left_data = self.data[self.data[:,self.split_axis] <= self.split_point]
            right_data = self.data[self.data[:,self.split_axis] > self.split_point]

            self.right_child = _KDNode(self.dimensions, child_axis, right_data)
            self.left_child = _KDNode(self.dimensions, child_axis, left_data)

            self.data = None
            self.left_child.split(level-1)
            self.right_child.split(level-1)

    def get_subtree_data(self):

        if self.data is not None:
            return self.data
        else:
            right_data = self.right_child.get_subtree_data()
            left_data = self.left_child.get_subtree_data()
            return np.concatenate((left_data, right_data))

    def balance(self):
        self.data = self.get_subtree_data()
        self.split(self.level)

    def pre_order(self):

        if self.split_point is not None:
            print(self.split_axis, self.split_point)
            self.left_child.pre_order()
            self.right_child.pre_order()


class KDTreeKNN(BaseKNN):

    def __init__(self, k, data, distance=2, balance_distance=10):

        super(KDTreeKNN, self).__init__(k, data, distance)
        dimensions = self.data.shape[1]
        self.tree = _KDNode(dimensions, 0, self.data)
        self.tree.split(level=5)
        self.balance_distance = balance_distance
        # self.tree.pre_order()

    def predict(self, x):

        node = self.tree
        while node.split_point is not None:
            if x[node.split_axis] <= node.split_point:
                node = node.left_child
            else:
                node = node.right_child

        distances = np.sum(self.distance(node.data- x), axis=1)
        min_index = np.argmin(distances)
        return node.data[min_index]

    def add_batch(self, new_batch):

        for x in new_batch:

            node = self.tree
            while node.split_point is not None:
                if np.abs(node.left_child.datalength-node.right_child.datalength) > self.balance_distance:
                    print("Balance called")
                    node.balance()

                if x[node.split_axis] <= node.split_point:
                    node = node.left_child
                else:
                    node = node.right_child

            node.data = np.concatenate((node.data, np.array([x])))
            node.datalength += 1


class _ParallelKDNode:

    def __init__(self, dimensions, split_axis, data, comm, node_id):
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()
        self.split_axis = split_axis
        self.dimensions = dimensions
        self.right_child = None
        self.left_child = None
        self.data = data
        self.split_point = None
        self.level = 0
        self.node_id = node_id
        self.datalength = None

    def split(self, level=1):

        if level > 0:
            if self.rank == 0:
                self.split_point = np.median(self.data[:,self.split_axis])
                self.datalength = self.data.shape[0]
                self.metadata = {
                    'split_point':self.split_point,
                    'datalength':self.datalength,
                }
            else:
                self.metadata = None


            self.metadata = self.comm.bcast(self.metadata, root=0)
            if self.rank != 0:
                self.split_point = self.metadata['split_point']
                self.datalength = self.metadata['datalength']


            self.level = level
            child_axis = (self.split_axis+1)%self.dimensions
            if self.rank == 0:
                left_data = self.data[self.data[:,self.split_axis] <= self.split_point]
                right_data = self.data[self.data[:,self.split_axis] > self.split_point]
            else:
                left_data, right_data = None, None

            self.right_child = _ParallelKDNode(self.dimensions, child_axis, right_data, self.comm, self.node_id)
            self.left_child = _ParallelKDNode(self.dimensions, child_axis, left_data, self.comm, self.node_id+(2**(self.level-1)))

            self.data = None
            self.left_child.split(level-1)
            self.right_child.split(level-1)

        elif level == 0:

            if self.rank == 0:
                self.datalength = self.data.shape[0]
            self.datalength = self.comm.bcast(self.datalength, root=0)

            if self.rank == 0 and self.node_id != 0:
                self.comm.send(self.data, dest=self.node_id, tag=self.node_id)
                self.data = None
            elif self.rank != 0 and self.rank == self.node_id:
                self.data = self.comm.recv(source=0, tag=self.node_id)



    def get_subtree_data(self):

        if self.level == 0 and self.rank == 0 and self.node_id != 0:
            data = self.comm.recv(source=self.node_id)
            return data
        elif self.level == 0 and self.rank == 0 and self.node_id == 0:
            return self.data
        elif self.level == 0 and self.rank == self.node_id:
            assert self.data is not None, "Correct node id has empty data"
            print("Sent by ", self.node_id)
            self.comm.send(self.data, dest=0)
            return None
        elif self.level != 0:
            right_data = self.right_child.get_subtree_data()
            left_data = self.left_child.get_subtree_data()
            if self.rank == 0:
                assert right_data is not None and left_data is not None, "Process 0 data is none"
                return np.concatenate((left_data, right_data))
            else:
                assert right_data is None and left_data is None, "Data is not none"

                return None

    def balance(self):
        print("Balance called on ", self.level, self.node_id, "by", self.rank)
        data = self.get_subtree_data()
        if self.rank == 0:
            self.data = data
        self.split(self.level)

    def pre_order(self):

        if self.split_point is not None:
            print(self.split_axis, self.split_point)
            self.left_child.pre_order()
            self.right_child.pre_order()

class ParallelKDTreeKNN(BaseKNN):

    def __init__(self, k, data, distance=2, balance_distance=10):

        super(ParallelKDTreeKNN, self).__init__(k, data, distance)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.num_procs = self.comm.Get_size()

        dimensions = self.data.shape[1]
        if self.rank == 0:
            self.tree = _ParallelKDNode(dimensions, 0, self.data, self.comm, 0)
        else:
            self.tree = _ParallelKDNode(dimensions, 0, None, self.comm, 0)
        self.tree.split(level=int(np.log2(self.num_procs)))
        self.balance_distance = balance_distance
        # self.tree.pre_order()

    def predict(self, x):

        node = self.tree
        while node.split_point is not None:
            if x[node.split_axis] <= node.split_point:
                node = node.left_child
            else:
                node = node.right_child

        if node.data is not None:
            distances = np.sum(self.distance(node.data- x), axis=1)
            min_index = np.argmin(distances)
            return node.data[min_index]
        else:
            return None

    def add_batch(self, new_batch):

        for x in new_batch:

            # print(self.tree.datalength, "as seen by", self.rank, "for", x)
            node = self.tree
            while node.level != 0:
                if np.abs(node.left_child.datalength-node.right_child.datalength) > self.balance_distance:
                    node.balance()

                node.datalength += 1

                if x[node.split_axis] <= node.split_point:
                    node = node.left_child
                else:
                    node = node.right_child

            node.datalength += 1
            if node.data is not None:
                node.data = np.concatenate((node.data, np.array([x])))
                # print("Data inserted at node", node.node_id, self.rank)

            self.comm.Barrier()
            # print("barrier called by", self.rank, "for", x)

