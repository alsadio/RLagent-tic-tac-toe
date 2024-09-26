import pytest
import sys
sys.path.append('./')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from script.Game import TicTakToe
from script.Agents import RuleBaseAgent,SarsaMLPAgent
from script.Player import Player
from script.Model import SarsaMLPModel

class TestModel:
    def test_model_play(self):
        num_test = 10
        max_epoch = 10000
        num_trials = 1000

        wins = []
        losses = []
        draws = []
        
        for i in range(num_test):
            em = SarsaMLPModel()
            agent = em.play_tic_tak_toe(max_epoch=max_epoch)

            win = 0
            loss = 0
            draw = 0
            for i in range(num_trials):
                env = TicTakToe(rng=0)
                while not env.terminate:
                    action = agent.maxQuery(env.board)
                    env.step(action)
                if env.winner == Player.PLAYER1:
                    win += 1
                elif env.winner == Player.PLAYER2:
                    loss += 1
                else:
                    draw += 1
                env.reset()
            wins.append(win)
            losses.append(loss)
            draws.append(draw)

        print("wins:",np.average(wins)/num_trials,"loss",np.average(losses)/num_trials,"draw",np.average(draws)/num_trials)

        assert True
        
        
