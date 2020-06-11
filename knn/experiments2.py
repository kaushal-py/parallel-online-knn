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
    # for i in range(4):
    #     if (i%2==0):
    #         query_generator1 = knn.Data_Generator(chunk_dist=query_chunk_dist, high = query_high, low = query_low, dim=dim, num_points=query_size, distribution=query_dist_list[0])
    #         query_loader1 = query_generator1.generator()
    #         query_data.append(next(query_loader1)[0])
            
    #     else:
    #         query_generator2 = knn.Data_Generator(chunk_dist=query_chunk_dist, high = query_high, low = query_low, dim=dim, num_points=query_size, distribution=query_dist_list[1])
    #         query_loader2 = query_generator2.generator()
    #         query_data.append(next(query_loader2)[0])
    query_data = [np.array([1.0, 0.0]), np.array([0.0, 0.0]), np.array([1.0, 1.0]), np.array([0.0, 1.0])]

    if algo_type == 'VKNN':
        
        build_itr = 0
        query_itr = 0
        build_time = 0
        query_time = 0

        # loop_KNN = knn.LoopKNN(k, data, distance)
        build_start = time.monotonic()
        vec_knn = knn.VectorKNN(k, data, distance)
        build_end = time.monotonic()
        print('Time for inssertion, ', build_itr+1,' ,ParVecKNN, for initial size of, ', init_size, ', is, ', build_end - build_start)

        #Querying with both distributions
        for i in range(4):
            query_start = time.monotonic()
            result = vec_knn.predict(query_data[i])
            query_end = time.monotonic()
            query_time += query_end-query_start
            print('Query Time, ParVecKNN, for, ', query_data[i], ',with balance length of, ', balance_dist, ', is, ' , query_end - query_start)
        
        # print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)

        for chunk in batch_loader:
            build_itr+=1
            build_start = time.monotonic()
            vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_end = time.monotonic()
            build_time += build_end - build_start
            print('Time for inssertion, ', build_itr+1,' ,ParVecKNN, for batch size of, ', batch_size, ', is, ', build_end - build_start)
            # build_itr+=1
            for i in range(4):
                query_start = time.monotonic()
                result = vec_knn.predict(query_data[i])
                query_end = time.monotonic()
                query_time += query_end-query_start
                print('Query Time, ParVecKNN, for, ', query_data[i], ',with balance length of, ', balance_dist, ', is, ' , query_end - query_start)
        print('Build Time: ',build_time)
        print('Query Time: ',query_time)
    
    elif algo_type == 'PVKNN':
        
        build_itr = 0
        query_itr = 0
        build_time = 0
        query_time = 0

        # loop_KNN = knn.LoopKNN(k, data, distance)
        
        # vec_knn = knn.VectorKNN(k, data, distance)
        build_start = time.monotonic()
        par_vec_knn = knn.ParallelVectorKNN(k, data, distance)
        if rank == 0:
            build_end = time.monotonic()
            print('Time for inssertion, ', build_itr+1,' ,ParVecKNN, for initial size of, ', init_size, ', is, ', build_end - build_start, ', ', comm.Get_size())

        #Querying with both distributions
        for i in range(4):
            # check = vec_knn.predict(query_data[i])
            # comm.Barrier()
            query_start = time.monotonic()
            result = par_vec_knn.predict(query_data[i])
            
            if rank == 0:
                query_end = time.monotonic()
                query_time += query_end-query_start
                # assert np.array_equal(check, result), "Outputs did not match."
                print('Query Time, ParVecKNN, for, ', query_data[i], ',with balance length of, ', balance_dist, ', is, ' , query_end - query_start, ', ', comm.Get_size())
        
        # print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)

        for chunk in batch_loader:
            build_itr+=1
            # vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_start = time.monotonic()
            par_vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_end = time.monotonic()
            if rank == 0:
                build_time += build_end - build_start
                print('Time for inssertion, ', build_itr+1,' ,ParVecKNN, for batch size of, ', batch_size, ', is, ', build_end - build_start, ', ', comm.Get_size())
            # build_itr+=1
            for i in range(4):
                # check = vec_knn.predict(query_data[i])
                # comm.Barrier()
                query_start = time.monotonic()
                result = par_vec_knn.predict(query_data[i])
                
                if rank == 0:
                    query_end = time.monotonic()
                    query_time += query_end-query_start
                    # assert np.array_equal(check, result), "Outputs did not match."
                    print('Query Time, ParVecKNN, for, ', query_data[i], ',with balance length of, ', balance_dist, ', is, ' , query_end - query_start, ', ', comm.Get_size())
        if rank == 0:
            print('Build Time: ',build_time)
            print('Query Time: ',query_time)

    elif algo_type == 'KDKNN':
        
        build_itr = 0
        query_itr = 0
        build_time = 0
        query_time = 0

        # loop_KNN = knn.LoopKNN(k, data, distance)
        
        # vec_knn = knn.VectorKNN(k, data, distance)
        build_start = time.monotonic()
        kdt_knn = knn.KDTreeKNN(k, data, distance, balance_dist)
        build_end = time.monotonic()
        print('Time for inssertion, ', build_itr+1,' ,KDKNN, for initial size of, ', init_size, ', is, ', build_end - build_start)

        #Querying with both distributions
        for j in range(10000):
            for i in range(1):
                # check = vec_knn.predict(query_data[i])
                query_start = time.monotonic()
                result = kdt_knn.predict(query_data[1])
                
                query_end = time.monotonic()
                query_time += query_end-query_start
                # assert np.array_equal(check, result), "Outputs did not match."
                # print('Query Time, KDKNN, for, ', query_data[i], ',with balance length of, ', balance_dist, ', is, ' , query_end - query_start)
        print('Total Query time after initial insertion: ', query_time)
        # print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)
        query_time = 0
        for chunk in batch_loader:
            build_itr+=1
            # vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_start = time.monotonic()
            kdt_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_end = time.monotonic()
            build_time += build_end - build_start
            print('Time for inssertion, ', build_itr+1,' ,KDKNN, for batch size of, ', batch_size, ', is, ', build_end - build_start)
            # build_itr+=1
            for j in range(10000):
                for i in range(1):
                    # check = vec_knn.predict(query_data[i])
                    query_start = time.monotonic()
                    result = kdt_knn.predict(query_data[1])

                    query_end = time.monotonic()
                    query_time += query_end-query_start
                    # assert np.array_equal(check, result), "Outputs did not match."
                    # print('Query Time, KDKNN, for, ', query_data[i], ',with balance length of, ', balance_dist, ', is, ' , query_end - query_start)
            print('Total query time after batch insertion: ', query_time)

        print('Build Time: ',build_time)
        print('Query Time: ',query_time)
    
    elif algo_type == 'ParKDKNN':
        
        build_itr = 0
        query_itr = 0
        build_time = 0
        query_time = 0

        # loop_KNN = knn.LoopKNN(k, data, distance)
        
        # vec_knn = knn.VectorKNN(k, data, distance)
        build_start = time.monotonic()
        kdt_knn = knn.ParallelKDTreeKNN(k, data, distance, balance_dist)
        comm.Barrier()
        if rank == 0:
            build_end = time.monotonic()
            print('Time for inssertion, ', build_itr+1,' ,ParKDKNN, for initial size of, ', init_size, ', is, ', build_end - build_start, ', ', comm.Get_size())
        comm.Barrier()
        #Querying with both distributions
        for i in range(4):
            # check = vec_knn.predict(query_data[i])
            query_start = time.monotonic()
            result = kdt_knn.predict(query_data[i])
            if result is not None:
                query_end = time.monotonic()
                query_time += query_end-query_start
                # assert np.array_equal(check, result), "Outputs did not match."
                print('First, Query Time, ParKDKNN, for, ', query_data[i], ',with balance length of, ', balance_dist, ', is, ' , query_end - query_start, ', ', comm.Get_size())
        comm.Barrier()
        # print('Building Tree, VectorKNN, ', build_itr+1, ', Time taken, ', build_end-build_start)

        for chunk in batch_loader:
            build_itr+=1
            # vec_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_start = time.monotonic()
            kdt_knn.add_batch(chunk/np.max(chunk, axis = 0))
            build_end = time.monotonic()
            build_time += build_end - build_start
            comm.Barrier()
            if rank == 0:
                print('Time for inssertion, ', build_itr+1,' ,ParKDKNN, for batch size of, ', batch_size, ', is, ', build_end - build_start, ', ', comm.Get_size())
            comm.Barrier()
            # build_itr+=1
            for i in range(4):
                # check = vec_knn.predict(query_data[i])
                query_start = time.monotonic()
                result = kdt_knn.predict(query_data[i])
                if result is not None:
                    query_end = time.monotonic()
                    query_time += query_end-query_start
                    # assert np.array_equal(check, result), "Outputs did not match."
                    print('Second, Query Time, ParKDKNN, for, ', query_data[i], ',with balance length of, ', balance_dist, ', is, ' , query_end - query_start, ', ', comm.Get_size())
            
            # print('Build Time: ',build_time)
            # print('Query Time: ',query_time)
        

if __name__ == "__main__":
    k = 1

    dim = 2
    # init_size = int(1e2)
    init_size = [int(1e6)]
    build_dist = 'uniform'
    batch_dist = 'gamma'
    query_dist_list = ['uniform', 'gamma']
    batch_chunk_dist = 'constant_high'
    query_chunk_dist = 'constant_low'
    batch_size = [int(1e5)]#, int(1e3), int(1e4), int(1e5)]#, int(1e6), int(1e7), int(1e8)] 
    query_size = 100
    # batch_high = int(1e4)
    batch_low = int(1e3)
    query_high = int(10)
    query_low = int(2)
    algo_type = 'KDKNN'
    distance = 2
    bal = np.inf

    for init_size in init_size:
        for num in (batch_size):
            # bal_dist = [500, 600, 800, 1000]
            # for bal in bal_dist:
            #     print(num, bal)
            print(init_size)
            execute(k, dim, init_size, build_dist, batch_dist, batch_chunk_dist, query_chunk_dist,\
                num, query_size, query_dist_list,distance, algo_type, num, \
                    batch_low, query_high, query_low, bal)
    # print('done')