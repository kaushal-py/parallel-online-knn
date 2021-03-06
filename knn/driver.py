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

    data_loader = knn.DataSystem(2, int(1e6), 'uniform')
    data = data_loader.generate(memory=True, path=None)
    batch_generator = knn.Data_Generator(chunk_dist='constant_low', dim=2, num_points=int(1e2), distribution='uniform')
    batch_loader = batch_generator.generator()

#     if rank==0:
# 
#         model_vector = knn.VectorKNN(1, data)
#         # model_loop = knn.LoopKNN(1, data)
#         model_vector.add_batch(next(batch_loader))
# 
#         start = time.monotonic()
#         nearest_vector = model_vector.predict(np.array([1.0,0.0]))
#         end = time.monotonic()
#         print("Execution Time:", end-start)
# 
#         # start = time.monotonic()
#         # nearest_loop = model_loop.predict(np.array([1.0,0.0]))
#         # end = time.monotonic()
#         # print("Execution Time:", end-start)
# 
#         # assert np.array_equal(nearest_loop, nearest_vector), "Outputs did not match."
# 
#         model_kd = knn.KDTreeKNN(1, data, balance_distance=10)
#                 
#         data_batch = next(batch_loader)
#         data_batch /= np.max(data_batch, axis=0)
#         model_kd.add_batch(data_batch)
#         start = time.monotonic()
#         nearest_kd = model_kd.predict(np.array([1.0,0.0]))
#         end = time.monotonic()
#         print("Execution Time:", end-start)
# 
#         print(nearest_kd, nearest_vector)
# 
#     model_parallel_vector = knn.ParallelVectorKNN(1, data)
#     model_parallel_vector.add_batch(next(batch_loader))
# 
#     start = time.monotonic()
#     nearest_parallel_vector = model_parallel_vector.predict(np.array([1.0,0.0]))
# 
#     if rank==0:
#         print(nearest_vector, nearest_parallel_vector)
#         end = time.monotonic()
#         print("Execution Time:", end-start)

    print("Processes started")
    model_parallel_kdtree = knn.ParallelKDTreeKNN(1, data, balance_distance=10)
    print("Creation done")
    model_parallel_kdtree.add_batch(next(batch_loader))
    model_parallel_kdtree.add_batch(next(batch_loader))
    model_parallel_kdtree.add_batch(next(batch_loader))
    model_parallel_kdtree.add_batch(next(batch_loader))
    model_parallel_kdtree.add_batch(next(batch_loader))
    print("Data added")
    comm.Barrier()
    start = time.monotonic()
    nearest_parallel_vector = model_parallel_kdtree.predict(np.array([1.0, 0.0]))
    if nearest_parallel_vector is not None:
        print(nearest_parallel_vector, "Found by", rank)
        end = time.monotonic()
        print("time taken", end-start)

    nearest_parallel_vector = model_parallel_kdtree.predict(np.array([0.0, 0.0]))
    if nearest_parallel_vector is not None:
        print(nearest_parallel_vector, "Found by", rank)

    nearest_parallel_vector = model_parallel_kdtree.predict(np.array([1.0, 1.0]))
    if nearest_parallel_vector is not None:
        print(nearest_parallel_vector, "Found by", rank)

    nearest_parallel_vector = model_parallel_kdtree.predict(np.array([0.0, 1.0]))
    if nearest_parallel_vector is not None:
        print(nearest_parallel_vector, "Found by", rank)

if __name__ == "__main__":
    main()
