import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt 
from script.Game import TicTakToe
from script.Agents import RuleBaseAgent,QLearningMLPAgent
from script.Player import Player

class QLearningMLPModel:
    def __init__(self,lr=0.002,alpha = 0.5,gamma = 0.9, eplison = 0.2, epsilon_decay=0.9999, epsilon_min=0.2):
        """initialize sarsa model
        setting up a Sarsa agent. Hyper-parameter should be turning by yourself

        Args:
            lr      : set value by yourself
            alpha   : set value by yourself
            gamma   : set value by yourself
            eplison : set value by yourself
        """
        self.agent = QLearningMLPAgent(lr, alpha, gamma, eplison, epsilon_decay, epsilon_min)
        self.env = TicTakToe(rng=0.01)


    def play_tic_tak_toe(self, max_epoch = 10000):#10000
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
        rewards = []
        for epoch in range(max_epoch):
            self.env.reset()
            total_reward = 0
            board = copy.deepcopy(self.env.board)

            while not self.env.terminate:
                action = self.agent.choose_action(board)
                if action is None:
                    break  # No moves available, end the game

                new_board, _, terminate, winner = self.env.step(action)

                # Calculate reward
                if terminate:
                    if winner == Player.PLAYER1:
                        reward = 7
                    elif winner == Player.PLAYER2:
                        reward = -5
                    else:
                        reward = -3
                elif action == 4:
                    reward = 2
                else:
                    reward = 1

                total_reward += reward

                # Choose next action for Q-learning update
                next_action = self.agent.choose_action(new_board)
                if next_action is None:
                    next_action = action  # If no next action, use current action

                # Update the agent
                self.agent.update_network(board, action, new_board, next_action, reward)

                board = copy.deepcopy(new_board)

            rewards.append(total_reward)
            self.agent.decay_epsilon()

            # Print progress
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{max_epoch}, Average Reward: {np.mean(rewards[-1000:]):.3f}")

        self.plot_agent_reward(reward_history=rewards)
        self.save_to_csv("895876")
        return self.agent

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
