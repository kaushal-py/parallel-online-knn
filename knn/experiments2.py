import time
import sys

from mpi4py import MPI
import numpy as np
import knn

def execute(k, dim, init_size, build_dist, batch_dist, \
    batch_chunk_dist, query_chunk_dist, batch_size, query_size, query_dist_list, \
        distance, algo_type, batch_high, batch_low, query_high, query_low, balance_dist):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    data_loader = knn.DataSystem(dim, init_size, build_dist)
    data = data_loader.generate(memory=True, path=None)
    # Normalizing the data
    data = data/np.max(data, axis = 0)

    batch_generator = knn.Data_Generator(chunk_dist=batch_chunk_dist, high = batch_high, low = batch_low, dim=dim, num_points=batch_size, distribution=batch_dist)
    batch_loader = batch_generator.generator()

    # Building the query data with 4 points from 2 distributions
    query_data = []
    for i in range(4):
        if (i%2==0):
            query_generator1 = knn.Data_Generator(chunk_dist=query_chunk_dist, high = query_high, low = query_low, dim=dim, num_points=query_size, distribution=query_dist_list[0])
            query_loader1 = query_generator1.generator()
            query_data.append(next(query_loader1)[0])
            
        else:
            query_generator2 = knn.Data_Generator(chunk_dist=query_chunk_dist, high = query_high, low = query_low, dim=dim, num_points=query_size, distribution=query_dist_list[1])
            query_loader2 = query_generator2.generator()
            query_data.append(next(query_loader2)[0])

    if algo_type == 'VKNN':
        
        build_itr = 0
        query_itr = 0
        build_time = 0
        query_time = 0

        # loop_KNN = knn.LoopKNN(k, data, distance)
        build_start = time.monotonic()
        vec_knn = knn.VectorKNN(k, data, distance)
        build_end = time.monotonic()
        print('Time for inssertion, ', build_itr+1,' ,VectorKNN, ', build_end - build_start)

        #Querying with both distributions
        for i in range(2):
            query_start = time.monotonic()
            result = vec_knn.predict(query_data[i])
            query_end = time.monotonic()
            query_time += query_end-query_start
            print('Query Time, VectorKNN, ', query_dist_list[i], ' , ', query_end - query_start)
        
        # print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)

        for chunk in batch_loader:
            build_itr+=1
            build_start = time.monotonic()
            vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_end = time.monotonic()
            build_time += build_end - build_start
            print('Time for inssertion, ', build_itr+1,' ,VectorKNN, ', build_end - build_start)
            # build_itr+=1
            for i in range(2,4):
                query_start = time.monotonic()
                result = vec_knn.predict(query_data[i])
                query_end = time.monotonic()
                query_time += query_end-query_start
                print('Query Time, VectorKNN, ', query_dist_list[i-2], ' , ', query_end - query_start)
        print('Build Time: ',build_time)
        print('Query Time: ',query_time)
    
    elif algo_type == 'PVKNN':
        
        build_itr = 0
        query_itr = 0
        build_time = 0
        query_time = 0

        # loop_KNN = knn.LoopKNN(k, data, distance)
        
        vec_knn = knn.VectorKNN(k, data, distance)
        build_start = time.monotonic()
        par_vec_knn = knn.ParallelVectorKNN(k, data, distance)
        if rank == 0:
            build_end = time.monotonic()
            print('Time for inssertion, ', build_itr+1,' ,ParallelVectorKNN, ', build_end - build_start)

        #Querying with both distributions
        for i in range(2):
            check = vec_knn.predict(query_data[i])
            query_start = time.monotonic()
            result = par_vec_knn.predict(query_data[i])
            
            if rank == 0:
                query_end = time.monotonic()
                query_time += query_end-query_start
                assert np.array_equal(check, result), "Outputs did not match."
                print('Query Time, ParallelVectorKNN, ', query_dist_list[i], ' , ', query_end - query_start)
        
        # print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)

        for chunk in batch_loader:
            build_itr+=1
            vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_start = time.monotonic()
            par_vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_end = time.monotonic()
            if rank == 0:
                build_time += build_end - build_start
                print('Time for inssertion, ', build_itr+1,' ,ParallelVectorKNN, ', build_end - build_start)
            # build_itr+=1
            for i in range(2,4):
                check = vec_knn.predict(query_data[i])
                query_start = time.monotonic()
                result = par_vec_knn.predict(query_data[i])
                
                if rank == 0:
                    query_end = time.monotonic()
                    query_time += query_end-query_start
                    assert np.array_equal(check, result), "Outputs did not match."
                    print('Query Time, ParallelVectorKNN, ', query_dist_list[i-2], ' , ', query_end - query_start)
        if rank == 0:
            print('Build Time: ',build_time)
            print('Query Time: ',query_time)

    elif algo_type == 'KDKNN':
        
        build_itr = 0
        query_itr = 0
        build_time = 0
        query_time = 0

        # loop_KNN = knn.LoopKNN(k, data, distance)
        
        vec_knn = knn.VectorKNN(k, data, distance)
        build_start = time.monotonic()
        kdt_knn = knn.KDTreeKNN(k, data, distance, balance_dist)
        build_end = time.monotonic()
        print('Time for inssertion, ', build_itr+1,' ,ParallelVectorKNN, ', build_end - build_start)

        #Querying with both distributions
        for i in range(2):
            check = vec_knn.predict(query_data[i])
            query_start = time.monotonic()
            result = kdt_knn.predict(query_data[i])
            
            query_end = time.monotonic()
            query_time += query_end-query_start
            assert np.array_equal(check, result), "Outputs did not match."
            print('Query Time, ParallelVectorKNN, ', query_dist_list[i], ' , ', query_end - query_start)
        
        # print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)

        for chunk in batch_loader:
            build_itr+=1
            vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_start = time.monotonic()
            kdt_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_end = time.monotonic()
            build_time += build_end - build_start
            print('Time for inssertion, ', build_itr+1,' ,ParallelVectorKNN, ', build_end - build_start)
            # build_itr+=1
            for i in range(2,4):
                check = vec_knn.predict(query_data[i])
                query_start = time.monotonic()
                result = kdt_knn.predict(query_data[i])

                query_end = time.monotonic()
                query_time += query_end-query_start
                assert np.array_equal(check, result), "Outputs did not match."
                print('Query Time, ParallelVectorKNN, ', query_dist_list[i-2], ' , ', query_end - query_start)

            print('Build Time: ',build_time)
            print('Query Time: ',query_time)
    
    elif algo_type == 'ParKDKNN':
        
        build_itr = 0
        query_itr = 0
        build_time = 0
        query_time = 0

        # loop_KNN = knn.LoopKNN(k, data, distance)
        
        vec_knn = knn.VectorKNN(k, data, distance)
        build_start = time.monotonic()
        par_kdt_knn = knn.ParallelKDTreeKNN(k, data, distance, balance_dist)
        if rank == 0:
            build_end = time.monotonic()
            print('Time for inssertion, ', build_itr+1,' ,ParallelVectorKNN, ', build_end - build_start)

        #Querying with both distributions
        for i in range(2):
            check = vec_knn.predict(query_data[i])
            query_start = time.monotonic()
            result = par_kdt_knn.predict(query_data[i])
            if rank == 0:
                query_end = time.monotonic()
                query_time += query_end-query_start
                assert np.array_equal(check, result), "Outputs did not match."
                print('Query Time, ParallelVectorKNN, ', query_dist_list[i], ' , ', query_end - query_start)
        
        # print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)

        for chunk in batch_loader:
            build_itr+=1
            vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_start = time.monotonic()
            par_kdt_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_end = time.monotonic()
            if rank == 0:
                build_time += build_end - build_start
                print('Time for inssertion, ', build_itr+1,' ,ParallelVectorKNN, ', build_end - build_start)
            # build_itr+=1
            for i in range(2,4):
                check = vec_knn.predict(query_data[i])
                query_start = time.monotonic()
                result = par_kdt_knn.predict(query_data[i])
                if rank == 0:
                    query_end = time.monotonic()
                    query_time += query_end-query_start
                    assert np.array_equal(check, result), "Outputs did not match."
                    print('Query Time, ParallelVectorKNN, ', query_dist_list[i-2], ' , ', query_end - query_start)
            if rank == 0:
                print('Build Time: ',build_time)
                print('Query Time: ',query_time)
        

if __name__ == "__main__":
    k = 1

    dim = 2
    init_size = int(1e8)
    build_dist = 'uniform'
    batch_dist = 'gamma'
    query_dist_list = ['normal', 'gamma']
    batch_chunk_dist = 'constant_high'
    query_chunk_dist = 'constant_low'
    batch_size = [int(1e3)]#, int(1e3), int(1e4), int(1e5), int(1e6), int(1e7)] 
    query_size = 100
    # batch_high = int(1e4)
    batch_low = int(1e3)
    query_high = int(10)
    query_low = int(2)
    algo_type = 'PVKNN'
    distance = 2
    balance_dist = np.Infinity

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
    for num in batch_size:
        print(num)
        execute(k, dim, init_size, build_dist, batch_dist, batch_chunk_dist, query_chunk_dist,\
            num, query_size, query_dist_list,distance, algo_type, num, \
                batch_low, query_high, query_low, balance_dist)
    # print('done')