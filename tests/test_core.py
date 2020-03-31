import numpy as np
from .context import knn

class TestCore:

    sanity_data = np.array(
        [[0, 0],
        [10, 10],
        [0, 10],
        [10, 0]]
    )

    def test_vector_knn(self):

        model = knn.core.VectorKNN(1, self.sanity_data)
        
        test_data = np.array([1,1])
        predicted = model.predict(test_data)
        assert np.array_equal(predicted, self.sanity_data[0])

        test_data = np.array([1,8])
        predicted = model.predict(test_data)
        assert np.array_equal(predicted, self.sanity_data[2])