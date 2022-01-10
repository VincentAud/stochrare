# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 15:28:14 2022

@author: vincent audoly
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
    
    def committor_function(self, A, B):
        """
        Takes two events A and B and compute a vector q representing the associated
        committor function evaluated for each possible initial state.
        The committor function q(x) for a given process and for two states A and B gives
        the probability to reach B before A.
        
        Parameters
        ----------
        A : list of index of event A
        B : list of index of event B

        Returns
        -------
        q : a vector of floats, q[i] is the probability to reach B before A starting from
            the state newly indexed by i
        ind : dictionnary, make the link between former indexes and new indexes, ind[i]=j
            means that the state indexed by i in the list of states self.states = X is indexed
            by j in vector q. Thus, the probability to reach B before A starting from state
            X[i] is q[j]

        """
        N, Na, Nb = len(self.transition_matrix), len(A), len(B)
        Gt = np.zeros((N-Na-Nb+2, N-Na-Nb+2))
        Gt[0][0] = 1
        Gt[1][1] = 1
        ind = {}
        p = 2
        for i in range(N):
            if i not in A and i not in B:
                ind[i] = p
                Gt[p][0] = sum([self.transition_matrix[i][k] for k in A])
                Gt[p][1] = sum([self.transition_matrix[i][k] for k in B])
                p += 1
        for i in range(N):
            if i not in A and i not in B:
                for j in range(N):
                    if j not in A and j not in B:
                        Gt[ind[i]][ind[j]]=self.transition_matrix[i][j]  
        vp, VP = np.linalg.eig(Gt)
        u, v = VP[:, vp == 1].T
        coeff = np.dot(np.linalg.inv(np.array([[u[0], v[0]], [u[1], v[1]]])), np.array([[0],[1]]))
        alpha, beta = coeff[0], coeff[1]
        q = alpha*u+beta*v
        return q, ind