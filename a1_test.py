import numpy as np
from tensorflow import keras
import unittest

import a1

class TestBasic(unittest.TestCase):
    def test_q1(self):
        image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]], \
                          [[  2,  20,  20], [250, 255, 255], [127, 127, 127]]])                          
        red_target = np.array([[1, 0, 0],
                               [0, 1, 1]])
        green_target = np.array([[0, 1, 0],
                                 [0, 1, 1]])
        blue_target = np.array([[0, 0, 1],
                                [1, 1, 1]])
        red_result = a1.light_pixels(image, 20, 'red')
        green_result = a1.light_pixels(image, 20, 'green')
        blue_result = a1.light_pixels(image, 15, 'blue')
        np.testing.assert_array_equal(red_result, red_target)
        np.testing.assert_array_equal(green_result, green_target)
        np.testing.assert_array_equal(blue_result, blue_target)

    def test_q2(self):
        image = np.array([[[250,   2,   2], [  0,   2, 255], [  0,   0, 255]], \
                          [[  2,   2,  20], [250, 255, 255], [127, 127, 127]]])                          
                        

        result_red = a1.histogram(image, 4, 'red')
        result_green = a1.histogram(image, 5, 'green')
        result_blue = a1.histogram(image, 6, 'blue')
        np.testing.assert_array_equal(result_red, 
                                      np.array([3, 1, 0, 2]))
        np.testing.assert_array_equal(result_green, 
                                      np.array([4, 0, 1, 0, 1]))
        np.testing.assert_array_equal(result_blue, 
                                      np.array([2, 0, 0, 1, 0, 3]))

    def test_q3(self):
        layer_options = [ 
            (128, 'relu', 0.2),  # 128 neurons, relu activation, 20% dropout
            (64, 'relu', 0),     # 64 neurons, relu activation, no dropout
            (32, 'sigmoid', 0.4) # 32 neurons, sigmoid activation, 40% dropout
        ]

        model = a1.build_deep_nn(28, 28, 2, layer_options)
        self.assertTrue(isinstance(model.layers[0], keras.layers.Flatten))
        self.assertEqual(model.layers[0].output.shape, (None, 1568))
        self.assertTrue(isinstance(model.layers[1], keras.layers.Dense))
        self.assertEqual(model.layers[1].output.shape, (None, 128))
        self.assertTrue(isinstance(model.layers[2], keras.layers.Dropout))
        self.assertTrue(isinstance(model.layers[3], keras.layers.Dense))
        self.assertEqual(model.layers[3].output.shape, (None, 64))
        self.assertTrue(isinstance(model.layers[4], keras.layers.Dense))
        self.assertEqual(model.layers[4].output.shape, (None, 32))
        self.assertTrue(isinstance(model.layers[5], keras.layers.Dropout))
        self.assertEqual(model.layers[1].get_config()['activation'],'relu')
        self.assertEqual(model.layers[3].get_config()['activation'],'relu')
        self.assertEqual(model.layers[4].get_config()['activation'],'sigmoid')

if __name__ == "__main__":
    unittest.main()

