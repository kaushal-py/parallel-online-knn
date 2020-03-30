import pytest

from .context import knn

class TestDataGenerator:

    def test_init(self):

        d = knn.utils.DataGenerator()
        assert d.dim == 2
        assert d.num_points == 1000

        d = knn.utils.DataGenerator(10, 10)
        assert d.dim == 10
        assert d.num_points == 10
    
    def test_generate_path_specified(self):
        
        d = knn.utils.DataGenerator()
        
        with pytest.raises(AssertionError):
            d.generate(memory=False)
    
    def test_generate_data_validation(self):

        generator = knn.utils.DataGenerator(10, 20)
        data = generator.generate()
        assert data.shape == (10, 20)


    def test_generator_mem_capacity(self):

        num_points = 10**6
        generator = knn.utils.DataGenerator(100, num_points)
        data = generator.generate()
        assert data.shape == (100, num_points)
        
