import sys
sys.path.append('./')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from script.Game import TicTakToe
from script.Agents import RuleBaseAgent,ESAgent
from script.Player import Player
from script.Model import Evolutionary_Model

if __name__ == "__main__":
    em = Evolutionary_Model(max_pop=100,parent_percent=0.15)
    best_agent = em.play_tic_tak_toe(max_epoch=300)
    em.plot_performance(em.fitness_history)
    print("Best reward:", em.best_reward)
    print("Best agent's weights:", best_agent.weights)
    print("Best agent's bias:", best_agent.bias)
    np.savetxt('B00895876_wts.csv', best_agent.weights, delimiter=',')
    np.savetxt('B00895876_bis.csv', best_agent.bias, delimiter=',')
    num_trials = 1000
    wins = 0
    loss = 0
    draw = 0

    for _ in range(num_trials):
        env = TicTakToe(rng=0.4)
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