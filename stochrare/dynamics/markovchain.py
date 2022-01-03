# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:28:14 2022

@author: vince
"""
import numpy as np

class MarkovChain:
    """
    This module defines the 'MarkovChain' class representing a discrete Markovian process
    by a transition matrix and a list of its states X. For instance, 
    the elements of X can be coordinates.
    
    This class has different methods.
    simulate_trajectory: generates a realization of the process
    """
    def __init__(self, name, X, G):
        self.name = name
        self.states = X
        self.transition_matrix = G
        
            
    def simulate_trajectory(self, index, N):
        """
        Generates a realization of the Markovian process starting from the state
        indexed by 'index' in the list of states the Markov Chain.

        Parameters
        ----------
        index : integer, index of the initial state in X
        N : integer, number of steps of the simulated process

        Returns
        -------
        Y : list, list of states [Y_1, Y_2, ...... Y_N] obtained with the simulation.

        """
        k = index
        Y = [self.states[index]]
        for _ in range(N):
            k = np.random.choice(a=range(len(self.states)),p=self.transition_matrix[k,:])
            Y.append(self.states[k])
        return Y