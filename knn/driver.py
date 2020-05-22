'''
Driver function for testing, monitoring the KNN algorithms
'''
import time

from mpi4py import MPI
import numpy as np
import knn

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    data_loader = knn.DataSystem(2, int(1e7), 'uniform')
    data = data_loader.generate(memory=True, path=None)
    if rank==0:

        model_vector = knn.VectorKNN(1, data)
        # model_loop = knn.LoopKNN(1, data)

        start = time.monotonic()
        nearest_vector = model_vector.predict(np.array([1.0,0.0]))
        end = time.monotonic()
        print("Execution Time:", end-start)

        # start = time.monotonic()
        # nearest_loop = model_loop.predict(np.array([1.0,0.0]))
        # end = time.monotonic()
        # print("Execution Time:", end-start)

        # assert np.array_equal(nearest_loop, nearest_vector), "Outputs did not match."

        # model_kd = knn.KDTreeKNN(1, data)
        # start = time.monotonic()
        # nearest_kd = model_kd.predict(np.array([1.0,0.0]))
        # end = time.monotonic()
        # print("Execution Time:", end-start)

        # print(nearest_kd, nearest_vector)

    model_parallel_vector = knn.ParallelVectorKNN(1, data)

    start = time.monotonic()
    nearest_parallel_vector = model_parallel_vector.predict(np.array([1.0,0.0]))

    if rank==0:
        print(nearest_vector, nearest_parallel_vector)
        end = time.monotonic()
        print("Execution Time:", end-start)


if __name__ == "__main__":
    main()
