import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from script.Uilts import feature_construct


class ESAgent:
    def __init__(self,weights = None, bias = None):
        """
            initialize agent with weight and bias

            The agent should carry a MLP network where it has 16 features input and 9 output.

            Hence, MLP has one input layer (16 weights), and input layer fully connects to one output layer (9 weights and 9 bias).

            In order to let ESAgent able to inherite network from its parent. The arguments should have default value for inilizate new population.
        Args:
            weights : weights of network or None
            bias : bias of network or None
        """
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(16, 9)

        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.random.rand(9)

    def forward(self, st):
        net = np.dot(st, self.weights) + self.bias
        yt = 1/ (1 + np.exp(-net))
        return yt
    
    def make_a_move(self,board):
        """make a move respect to current board state. You would like to use feature_construct function to convert board state to input features

        Args:
            board: tic tak toe game state

        Returns:
            move (0-8): position of board

        """
        st = feature_construct(board)

        yt = self.forward(st)

        mask = (board.flatten() == 0).astype(int)

        masked_yt = yt * mask

        move = np.argmax(masked_yt)

        return move



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
    