import pytest
import sys
sys.path.append('./')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from script.Game import TicTakToe
from script.Agents import RuleBaseAgent,ESAgent
from script.Player import Player
from script.Model import Evolutionary_Model

class TestAgent:
    def test_ESAgent_network(self):
        agent = ESAgent()

        assert agent.weights.shape == (16,9)
        assert agent.bias.shape == (9,)

    def test_ESAgent_network(self):
        agent = ESAgent()
        
        board = np.array([[0,1,1],
                          [1,-1,-1],
                          [-1,-1,1]
                          ])
        move = agent.make_a_move(board)
        assert move == 0

        board = np.array([[-1,1,1],
                          [1,0,-1],
                          [-1,-1,1]
                          ])
        move = agent.make_a_move(board)
        assert move == 4
        
        board = np.array([[-1,1,1],
                          [1,-1,-1],
                          [0,-1,1]
                          ])
        move = agent.make_a_move(board)
        assert move == 6

class TestModel:
    def test_model_play(self):
        em = Evolutionary_Model(max_pop=1000,parent_percent=0.2)
        best_agent = em.play_tic_tak_toe(max_epoch=100)

        num_trials = 1000
        wins = 0
        loss = 0
        draw = 0

        for _ in range(num_trials):
            env = TicTakToe(rng=0)
            while not env.terminate:
                action = best_agent.make_a_move(env.board)
                env.step(action)
            if env.winner == Player.PLAYER1:
                wins += 1
            elif env.winner == Player.PLAYER2:
                loss += 1
            else:
                draw += 1
            env.reset()

        print("wins:",wins/num_trials,"loss",loss/num_trials,"draw",draw/num_trials)

        assert True
        