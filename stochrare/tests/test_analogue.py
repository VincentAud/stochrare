# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 15:51:53 2021

@author: vincent audoly
"""


import unittest
import numpy as np
import stochrare.data.analogue as analogue

class TestAnalog(unittest.TestCase):
    def test_stoch_matrix(self):
        X = [np.random.random_sample(2) for i in range(100)]
        for K in range(2,10):
          markov = analogue.AnalogueMarkovChain('test',X,K)
          np.testing.assert_allclose(markov.transition_matrix.sum(axis=1),np.ones(100))  
    def coeff_matrix(self):
        X = [np.random.random_sample(2) for i in range(100)]
        for K in range(2,10):
            markov = analogue.AnalogueMarkovChain('test',X, K)
            np.testing.assert_allclose(np.unique(markov.transition_matrix),np.array([0,1/K,1/(K-1)]))
    def true_process(self):
        X = [np.array([i,i]) for i in range(10)]
        markov = analogue.AnalogueMarkovChain('test',X,2)
        Gth = np.array(0.5*[[0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0, 2, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
        np.testing.assert_allclose(markov.transition_matrix,Gth)
if __name__ == "__main__":
    unittest.main()