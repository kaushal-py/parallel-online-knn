'''
Driver function for testing, monitoring the KNN algorithms
'''
import time

import numpy as np
import knn

def main():
    data_loader = knn.DataSystem(2, int(1e8), 'uniform')
    data = data_loader.generate(memory=True, path=None)

    model_vector = knn.VectorKNN(1, data)
    model_loop = knn.LoopKNN(1, data)

    start = time.monotonic()
    nearest_vector = model_vector.predict(np.array([1.0,0.0]))
    end = time.monotonic()
    print("Execution Time:", end-start)

    # start = time.monotonic()
    # nearest_loop = model_loop.predict(np.array([1.0,0.0]))
    # end = time.monotonic()
    # print("Execution Time:", end-start)

    # assert np.array_equal(nearest_loop, nearest_vector), "Outputs did not match."

    model_kd = knn.KDTreeKNN(1, data)
    start = time.monotonic()
    nearest_kd = model_kd.predict(np.array([1.0,0.0]))
    end = time.monotonic()
    print("Execution Time:", end-start)

    print(nearest_kd, nearest_vector)


if __name__ == "__main__":
    main()
