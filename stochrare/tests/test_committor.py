"""
Unit tests for the analogue module.
"""
import unittest
import numpy as np
import stochrare.data.analogue as analogue

class TestAnalogueCommittor(unittest.TestCase):

    def test_committor_gamblersruin(self):
        nmax = 10
        proba = 0.5
        # Define the "Gambler's ruin" Markov chain
        G = proba*np.eye(nmax+1, k=1)+(1-proba)*np.eye(nmax+1, k=-1)
        G[0, 0] = G[-1, -1] = 1
        G[0, 1] = G[-1, -2] = 0
        gamblersruin = analogue.MarkovChain("Gambler's Ruin", np.arange(nmax+1), G)
        # Compute the committor function by solving the eigenvalue problem
        q_analogue, ind = gamblersruin.committor_function([0], [nmax])
        q_analogue.sort() # WARNING: this is a hugly hack ! We should instead keep track of how the states are rearranged when constructing the Gtilde matrix.
        # Compare to theoretical solution
        if proba == 0.5:
            q_theory = np.arange(nmax+1)/nmax
        else:
            q_theory = np.array([(1-(1/proba-1)**k)/(1-(1/proba-1)**nmax) for k in range(nmax+1)])
        np.testing.assert_allclose(q_analogue, q_theory)

if __name__ == "__main__":
    unittest.main()
