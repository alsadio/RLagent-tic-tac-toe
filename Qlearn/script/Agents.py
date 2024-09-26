import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from script.Uilts import feature_construct

class QLearningMLPAgent:
    def __init__(self,lr,alpha,gamma, eplison, epsilon_decay, epsilon_min):
        """INIT function
            initilize a MLP with (16,9) weights and 9 bias

        Args:
            lr (float, optional): learning rate
            alpha (float, optional): learning rate of Q
            gamma (float, optional): future weight
            eplison (float, optional): random action chance
        """
        self.weights = np.random.randn(16, 9) * 0.01
        self.bias = np.zeros(9)
        
        self.lr = lr
        self.alpha = alpha
        self.gamma = gamma
        self.eplison = eplison
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def decay_epsilon(self):
        self.eplison = max(self.epsilon_min, self.eplison * self.epsilon_decay)
    def forward(self,board):
        """forward pass to network

        Args:
            board

        Returns:
            output of network
        """
        features = feature_construct(board)
        net = np.dot(features, self.weights) + self.bias
        output = 1 / (1 + np.exp(-net))
        return output
    def maxQuery(self,board):
        """return the action with maximum q-value

        Args:
            board

        Returns:
            action
        """
        output = self.forward(board)
        mask = (board.flatten() == 0)
        masked_output = np.where(mask, output, -np.inf)
        action = np.argmax(masked_output)
        return action
    def update_network(self,board_t,action_t,board_t_1,action_t_1,reward):
        """update network with correspond new q value

        Args:
            board_t 
            action_t 

            board_t_1
            action_t_1

            reward
        """
        yi = self.forward(board_t)[action_t]
        f_prime = yi * (1 - yi)
        q_t = self.forward(board_t)[action_t]
        q_t_1 = np.max(self.forward(board_t_1))  # Use max Q-value for next state
        features_t = feature_construct(board_t)
        td_error = reward + ((self.gamma * q_t_1) - q_t)

        err = td_error * f_prime
        self.weights[:, action_t] += self.lr * err * features_t
        self.bias[action_t] += self.lr * err

    def choose_action(self, board):
        available_moves = np.where(board.flatten() == 0)[0]
        if len(available_moves) == 0:
            return None  # No moves available
        if np.random.random() < self.eplison:
            return np.random.choice(available_moves)
        else:
            return self.maxQuery(board)

# dont change any code after this
class RuleBaseAgent:
    def __init__(self,id,rival_id,p_rnd=0.1):
        self.p_rnd = p_rnd
        self.move = -1
        self.id = id
        self.rival_id = rival_id
    
    def make_a_move(self,board):
        self.find_avaliable_position(board)
        if np.random.random() < self.p_rnd:
            self.random_move()
        elif self.make_win_move(board):
            pass
        elif self.make_block_move(board):
            pass
        elif self.make_two_open_move(board):
            pass
        else: 
            self.random_move()
        self.avaliable_moves = None
        return self.move
        
    def find_avaliable_position(self,board):
        self.avaliable_moves = [i for i in range(9) if board[i//3][i%3] == 0]

    def random_move(self):
        move = np.random.choice(self.avaliable_moves)
        # move = self.avaliable_moves[0]
        self.move = (move//3,move%3)

    def make_win_move(self,board):
        for i,row in enumerate(board):
            if row.sum() == 2 * self.id:
                for j,value in enumerate(row):
                    if value == 0:
                        self.move= (i,j)
                        return True
                    
        
        for j,col in enumerate(board.T):
            if col.sum() == 2 * self.id:
                for i,value in enumerate(col):
                    if value == 0:
                        self.move= (i,j) 
                        return True
                    
        if board.trace() == 2 * self.id:
            for i in range(3):
                if board[i][i] == 0:
                    self.move = (i,i)
                    return True
        
        if np.fliplr(board).trace() == 2 * self.id:
            for i in range(3):
                if board[i][2-i] == 0:
                    self.move = (i,2-i)
                    return True
        
        return False
    
    def make_block_move(self,board):
        for i,row in enumerate(board):
            if row.sum() == 2 * self.rival_id:
                for j,value in enumerate(row):
                    if value == 0:
                        self.move= (i,j)
                        return True
                    
        for j,col in enumerate(board.T):
            if col.sum() == 2 * self.rival_id:
                for i,value in enumerate(col):
                    if value == 0:
                        self.move= (i,j)
                        return True
    
        if board.trace() == 2 * self.rival_id:
            for i in range(3):
                if board[i][i] == 0:
                    self.move = (i,i)
                    return True
        
        if np.fliplr(board).trace() == 2 * self.rival_id:
            for i in range(3):
                if board[i][2-i] == 0:
                    self.move = (i,2-i)
                    return True
        
        return False
    
    def make_two_open_move(self,board):
        p = 0.5
        if board.trace() == self.id:
            for i in range(3):
                if board[i][i] == 0:
                    if p < np.random.random():
                        self.move = (i,i)
                        return True
                    else:
                        p = 1
        
        p = 0.5
        if np.fliplr(board).trace() == self.id:
            for i in range(3):
                if board[i][2-i] == 0:
                    if p < np.random.random():
                        self.move = (i,2-i)
                        return True
                    else:
                        p = 1
        
        p = 0.5
        for i,row in enumerate(board):
            if row.sum() == self.id:
                for j,value in enumerate(row):
                    if value == 0:
                        if p < np.random.random():
                            self.move= (i,j)
                            return True
                        else:
                            p = 1
                    
        
        p = 0.5
        for j,col in enumerate(board.T):
            if col.sum() == self.id:
                for i,value in enumerate(col):
                    if value == 0:
                        if p < np.random.random():
                            self.move= (i,j)
                            return True
                        else:
                            p = 1

        return False
    