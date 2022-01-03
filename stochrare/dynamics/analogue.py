# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 14:16:21 2021

@author: vincent audoly

This module defines the 'AnalogueMarkovChain' class. This class heritates from
the 'MarkovChain' class defined in another file. Here the transition matrix of the Markov Chain
is constructed with a method which takes a realization [X_1, X_2, ... X_n] of a process
and a parameter K (number of analogues)
and construt a transition matrix of a Markoc Chain called "analogue markov chain" which
is supposed to reproduce the dynamics of the process studied. 
"""

import numpy as np
from scipy.spatial import KDTree
from markovchain import MarkovChain


class AnalogueMarkovChain(MarkovChain):
    
    def __init__(self, name, X, K):
        tree, G = self.construct_analog_markov_chain(X, K)
        MarkovChain.__init__(self, name, X, G)
        self.K = K
        self.tree = tree
        #self.real = []
        
    @classmethod
    def construct_analog_markov_chain(cls, X, K):
        """
        Takes a list of states X = [X_1, X_2, ........X_T] and a number or analogues K
        and compute the matrix of the analogue Markov chain
        
        Parameters
        ----------
        X : list of states
        K : integer, number of analogues
        
        Returns
        -------
        tree : Tree used by the module KDTree in order to find the nearest neighbours of each
            state
        G : The transition matrix of the analogue Markov chain
        """
        N = len(X)
        G = np.zeros((N,N))
        tree = KDTree(X)
        for n in range(N):
            Xn = X[n]
            dist, ind = tree.query(Xn,K+1)
            if (N-1) not in ind[1:]:
                for j in ind[1:]:
                    G[n,j+1]=1/K
            else:
                for j in ind[1:]:
                    if j != (N-1):
                        G[n,j+1]=1/(K-1)
        return tree, G
    
        
    def simulate_trajectory(self, x0, N):
        """
        Takes an initial state and a number of steps N and compute a realization
        of the analogue process
        
        Parameters
        ----------
        x0 : Initial state.
        N : Number of steps we want to compute

        Returns
        -------
        x : The N steps of the simulated trajectory

        Notes
        -----
        We overwrite the general method for trajectory generation with a Markov chain.
        In the case of the analogue Markov chain, it may be better to use the analogue
        tree directly to avoid errors related to machine precision for coefficients 
        which are zero in the transition matrix. 
        """
        x = [x0]
        ind = self.tree.query([x0], self.K)[1][0]
        k = np.random.choice(ind)+1
        x.append(self.states[k])
        for i in range(N-1):
            #k = np.random.choice(a=range(len(X)),p=G[k,:])
            ind = self.tree.query([x[-1]], self.K +1)[1][0]
            ind = ind.tolist()
            if len(self.states)-1 in ind:
                ind.remove(len(self.states)-1)   
            k = np.random.choice(ind[1:]) + 1
            x.append(self.states[k])
        return x
        
    
        
        
          
        
        
        
