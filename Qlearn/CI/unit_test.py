import pytest
import sys
sys.path.append('./')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from script.Game import TicTakToe
from script.Agents import RuleBaseAgent,QLearningMLPAgent
from script.Player import Player
from script.Model import QLearningMLPModel

class TestModel:
    def test_model_play(self):
        hist = [[],[],[]]
        rounds = 10
        
        for i in range(rounds):
            em = QLearningMLPModel()
            agent = em.play_tic_tak_toe(max_epoch=10000)

            num_trials = 1000
            wins = 0
            loss = 0
            draw = 0

            for i in range(num_trials):
                env = TicTakToe(rng=0)
                while not env.terminate:
                    action = agent.maxQuery(env.board)
                    env.step(action)
                if env.winner == Player.PLAYER1:
                    wins += 1
                elif env.winner == Player.PLAYER2:
                    loss += 1
                else:
                    draw += 1
                env.reset()

            hist[0].append(wins)
            hist[1].append(loss)
            hist[2].append(draw)

        print("wins:",np.average(hist[0])/num_trials,"loss",np.average(hist[1])/num_trials,"draw",np.average(hist[2])/num_trials)

        assert True
        
