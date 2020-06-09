import time
import sys

from mpi4py import MPI
import numpy as np
import knn

def execute(k, dim, init_size, build_dist, batch_dist, query_dist, \
    batch_chunk_dist, query_chunk_dist, batch_size, query_size, \
        distance, algo_type, batch_high, batch_low, query_high, query_low):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    data_loader = knn.DataSystem(dim, init_size, build_dist)
    data = data_loader.generate(memory=True, path=None)
    # Normalizing the data
    data = data/np.max(data, axis = 0)

    batch_generator = knn.Data_Generator(chunk_dist=batch_chunk_dist, high = batch_high, low = batch_low, dim=dim, num_points=batch_size, distribution=batch_dist)
    batch_loader = batch_generator.generator()

    query_generator = knn.Data_Generator(chunk_dist=query_chunk_dist, high = query_high, low = query_low, dim=dim, num_points=query_size, distribution=query_dist)
    query_loader = query_generator.generator()

    # if algo_type == 'LKNN':
        
    # build_itr = 0
    # query_itr = 0

    # build_start = time.monotonic()
    loop_KNN = knn.LoopKNN(k, data, distance)
    # build_end = time.monotonic()
    # # print('Building Tree, LoopKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)

    # for chunk in next(batch_loader):
    #     loop_KNN.add_batch(chunk)
    #     build_itr+=1
    #     if (build_itr%100 == 0):
    #         query_start = time.monotonic()
    #         loop_KNN.predict(np.array([1.0,0.0]))
    #         query_end = time.monotonic()
    #         # print('Querying , LoopKNN, ', query_itr+1, ', Time taken, ', query_end-query_start)
    #         query_itr+=1
    
    if algo_type == 'VKNN':
        
        build_itr = 0
        query_itr = 0
        build_time = 0
        query_time = 0

        # loop_KNN = knn.LoopKNN(k, data, distance)
        build_start = time.monotonic()
        vec_knn = knn.VectorKNN(k, data, distance)
        build_end = time.monotonic()
        # print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)

        for chunk in batch_loader:
            build_itr+=1
            # print(chunk.shape)
            # loop_KNN.add_batch(chunk/np.max(chunk, axis = 0))
            build_start = time.monotonic()
            vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_end = time.monotonic()
            build_time += build_end - build_start
            print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)
            # build_itr+=1
            if (build_itr%50 == 0):
                query_data = next(query_loader)
                # print(query_data.shape)
                for query in query_data:
                    query_start = time.monotonic()
                    result = vec_knn.predict(np.array([1.0,0.0]))
                    query_end = time.monotonic()
                    query_time += query_end - query_start
                    # assert np.array_equal(loop_result, result), "Outputs did not match."
                    print('Querying , VectorKNN, ', query_itr+1, ', Time taken, ', query_end-query_start)
                query_itr+=1
        print('Build Time: ',build_time)
        print('Query Time: ',query_time)
    
    elif algo_type == 'PVKNN':
        
        build_itr = 0
        query_itr = 0
        build_time = 0
        query_time = 0

        vec_knn = knn.VectorKNN(k, data, distance)
        build_start = time.monotonic()
        par_vec_knn = knn.ParallelVectorKNN(k, data, distance)
        build_end = time.monotonic()
        # print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)

        for chunk in batch_loader:
            build_itr+=1
            # print(chunk.shape)
            vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_start = time.monotonic()
            par_vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            if rank == 0:
                build_end = time.monotonic()
                build_time += build_end - build_start
                # print('Building Tree, ParallelVectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)
            if (build_itr%50 == 0):
                query_data = next(query_loader)

                for query in query_data:
                    
                    vec_knn_result = vec_knn.predict(query)
                    query_start = time.monotonic()
                    # print(vec_knn_result)
                    result = par_vec_knn.predict(query)
                    # print(result)
                    if rank==0:
                        query_end = time.monotonic()
                        query_time += query_end - query_start
                        assert np.array_equal(vec_knn_result, result), "Outputs did not match."
                        # print('Querying , ParallelVectorKNN, ', query_itr+1, ', Time taken, ', query_end-query_start)
                query_itr+=1
        if rank == 0:
            print('Build Time: ',build_time)
            print('Query Time: ',query_time)

    elif algo_type == 'KDKNN':
        
        build_itr = 0
        query_itr = 0
        build_time = 0
        query_time = 0

        vec_knn = knn.VectorKNN(k, data, distance)
        build_start = time.monotonic()
        kdt_knn = knn.KDTreeKNN(k, data, distance, balance_distance=np.Infinity)
        build_end = time.monotonic()
        print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)

        for chunk in batch_loader:
            build_itr+=1
            # print(chunk.shape)
            vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_start = time.monotonic()
            kdt_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_end = time.monotonic()
            build_time += build_end - build_start
            print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)
            # build_itr+=1
            # if (build_itr%50 == 0):
            query_data = next(query_loader)
            # print(query_data.shape)
            for query in query_data:
                vec_knn_result = vec_knn.predict(query)
                query_start = time.monotonic()
                result = kdt_knn.predict(query)
                query_end = time.monotonic()
                query_time += query_end - query_start
                assert np.array_equal(vec_knn_result, result), "Outputs did not match."
                print('Querying , VectorKNN, ', query_itr+1, ', Time taken, ', query_end-query_start)
            query_itr+=1
        print('Build Time: ',build_time)
        print('Query Time: ',query_time)
        

if __name__ == "__main__":
    k = 1
    # dim = int(sys.argv[1])
    # build_dist = sys.argv[2]
    # batch_dist = sys.argv[3]
    # query_dist = sys.argv[4]
    # batch_chunk_dist = sys.argv[5]
    # query_chunk_dist = sys.argv[6]
    # batch_size = int(sys.argv[7])
    # query_size = int(sys.argv[8])
    # algo_type = sys.argv[9]
    # distance = 2

    dim = 2
    init_size = int(1e8)
    build_dist = 'uniform'
    batch_dist = 'gamma'
    query_dist = 'normal'
    batch_chunk_dist = 'constant_high'
    query_chunk_dist = 'constant_low'
    batch_size = int(1e3)
    query_size = 100
    batch_high = int(1e3)
    batch_low = int(1e3)
    query_high = int(10)
    query_low = int(2)
    algo_type = 'KDKNN'
    distance = 2

    # dim = [2,3,4,5,6,7,8,9,10]
    # init_size = [int(1e8), int(1e9), int(1e10)]
    # build_dist = ['normal', 'uniform', 'gamma', 'beta', 'exponential']
    # batch_dist = ['normal', 'uniform', 'gamma', 'beta', 'exponential']
    # query_dist = ['normal', 'uniform', 'gamma', 'beta', 'exponential']
    # batch_chunk_dist = ['constant_high', 'constant_low', 'random', 'crest_trough', 'low_high', 'high_low']
    # query_chunk_dist = ['constant_high', 'constant_low', 'random', 'crest_trough', 'low_high', 'high_low']
    # batch_size = int(1e6)
    # query_size = 100
    # batch_high = list(np.arange(1e4, 1e8, 1e3))
    # batch_low = int(1e3)
    # query_high = int(10)
    # query_low = int(2)
    # algo_type = 'VKNN'
    # distance = 2
    execute(k, dim, init_size, build_dist, batch_dist, query_dist, batch_chunk_dist, query_chunk_dist,\
        batch_size, query_size, distance, algo_type, batch_high, batch_low, query_high, query_low)


    # if rank==0:

    #     model_vector = knn.VectorKNN(1, data)
    #     # model_loop = knn.LoopKNN(1, data)
    #     model_vector.add_batch(next(batch_loader))

    #     start = time.monotonic()
    #     nearest_vector = model_vector.predict(np.array([1.0,0.0]))
    #     end = time.monotonic()
    #     print("Execution Time:", end-start)

    #     # start = time.monotonic()
    #     # nearest_loop = model_loop.predict(np.array([1.0,0.0]))
    #     # end = time.monotonic()
    #     # print("Execution Time:", end-start)

    #     # assert np.array_equal(nearest_loop, nearest_vector), "Outputs did not match."

    #     model_kd = knn.KDTreeKNN(1, data, balance_distance=2)
    #     model_kd.add_batch(next(batch_loader))
    #     start = time.monotonic()
    #     nearest_kd = model_kd.predict(np.array([1.0,0.0]))
    #     end = time.monotonic()
    #     print("Execution Time:", end-start)

    #     print(nearest_kd, nearest_vector)

    # model_parallel_vector = knn.ParallelVectorKNN(1, data)
    # model_parallel_vector.add_batch(next(batch_loader))

    # start = time.monotonic()
    # nearest_parallel_vector = model_parallel_vector.predict(np.array([1.0,0.0]))

    # if rank==0:
    #     print(nearest_vector, nearest_parallel_vector)
    #     end = time.monotonic()
    #     print("Execution Time:", end-start)

    # print("Processes started")
    # model_parallel_kdtree = knn.ParallelKDTreeKNN(1, data)