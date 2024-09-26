import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt 
from script.Game import TicTakToe
from script.Agents import RuleBaseAgent,SarsaMLPAgent
from script.Player import Player

class SarsaMLPModel:
    def __init__(self,lr=0.002,alpha = 0.5,gamma = 0.9, eplison = 0.2):
        """initialize sarsa model
        setting up a Sarsa agent. Hyper-parameter should be turning by yourself

        Args:
            lr      : set value by yourself
            alpha   : set value by yourself
            gamma   : set value by yourself
            eplison : set value by yourself
        """
        self.rewards = []
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = eplison
        self.agent = SarsaMLPAgent(lr=self.lr, alpha=self.alpha, gamma=self.gamma, eplison=self.epsilon)
    def play_tic_tak_toe(self, max_epoch = 10000):
        """play tic tak toe with sarsa algorithm and return the agent you have

        you should follow the persudocode in slides but you allow to some changes as long as it return the best agent you have in the model.

        1. intilize environment
        2. first action and board
        3. do
        4.  step action to environment
        5.  new action and new board
        6.  update agent
        7   action,board <- new action,new board
        8. return agent

        Returns:
            agent
        """
        for epoch in range(max_epoch):
            env = TicTakToe(rng=0.19)
            state = env.board
            action = self.agent.epsilon_greedy(state)

            episode_reward = 0
            while not env.terminate:
                next_state, _, _, _ = env.step(action)
                reward = self.get_reward(env)
                episode_reward += reward

                if env.terminate:
                    next_action = None
                else:
                    next_action = self.agent.epsilon_greedy(next_state)

                self.agent.update_network(state, action, next_state, next_action, reward)
                state = next_state
                action = next_action

            self.rewards.append(episode_reward)

        self.plot_agent_reward(reward_history=self.rewards)
        return self.agent

    def get_reward(self, env):
        if env.terminate:
            if env.winner == Player.PLAYER1:
                return 10  # Win
            elif env.winner == Player.PLAYER2:
                return -5  # Loss
            else:
                return -1  # Draw
        return 0

    def plot_agent_reward(self, reward_history):
        plt.figure(figsize=(12, 6))
        plt.plot(np.cumsum(reward_history))
        plt.title('Agent Cumulative Reward vs. Iteration')
        plt.ylabel('Reward')
        plt.xlabel('Episode')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()

    def save_to_csv(self, b00):
        weights_df = pd.DataFrame(self.agent.weights)
        bias_df = pd.DataFrame(self.agent.bias.reshape(1, -1))
        weights_df.to_csv(f'B00{b00}_wts.csv', index=False, header=False)
        bias_df.to_csv(f'B00{b00}_bis.csv', index=False, header=False)
