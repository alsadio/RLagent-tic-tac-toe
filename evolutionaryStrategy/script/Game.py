from script.Player import Player
from script.Agents import RuleBaseAgent
import copy

import numpy as np

    
class TicTakToe():
    """
        tic take tow environment. you dont need to change any code here

    """
        
    def __init__(self,rng):
        """initilize opponent rule base agent

        Args:
            rng (_type_): it the number of chance opponents make random move
        """
        self.player2 = RuleBaseAgent(id=Player.PLAYER2,rival_id=Player.PLAYER1,p_rnd=rng)
        self.reset()

    def reset(self):
        """
            reset game state
        """
        self.board = np.zeros((3,3)).astype('int')
        self.prev_board = copy.deepcopy(self.board)
        self.curr_player = Player.PLAYER1
        self.round = 1
        self.winner = 0
        self.terminate = False

    def step(self,action):
        """environment take a action from player 1 (in this case will be your agent), and it will take this move and call opponent agent also move. 

        Args:
            action      : the position of board

        Returns:
            board       : the board state
            prev_board  : the prev board state (no useful in assignment 2)
            terminate   : if game terminated
            self.winner : winner of game if terminated, Player 1 -> player 1 win, Player 2 -> player 2 win, 0 -> draw
        """
        if self.terminate:
            return self.board,self.prev_board,self.terminate,self.winner

        if not isinstance(action,list) and not isinstance(action, np.ndarray) and not isinstance(action, tuple):
            x,y = (action//3,action%3)
        elif len(action) == 2:
            x,y = action
        else:
            print("invalid input")
            assert RuntimeError
        
        self.move(x,y)
        self.update_round()

        if not self.terminate:
            self.prev_board = copy.deepcopy(self.board)
            action = self.player2.make_a_move(self.board)
            x,y = action
            self.move(x,y)
            self.update_round()

        return self.board,self.prev_board,self.terminate,self.winner
    
    def switch_player(self):
        if self.curr_player == Player.PLAYER1:
            self.curr_player = Player.PLAYER2
        else:
            self.curr_player = Player.PLAYER1

    def move(self,x,y):
        if x < 0 or x > 2 or y < 0 or y > 2:
            print("out of boundary.")
            return False
        elif self.board[x][y] != 0:
            print("occupied.")
            return False
        else:
            self.board[x][y] = self.curr_player
            self.last_move = (x,y)

            return True

    def is_win(self):
        x,y = self.last_move
        rows_check = self.board[x].sum() == 3 * self.curr_player
        cols_check = self.board[:,y].sum() == 3 * self.curr_player

        left_diagonal_check = False
        if x == y:
            left_diagonal_check = np.trace(self.board) == 3 * self.curr_player

        right_diagonal_check = False
        if x + y == 2:
            right_diagonal_check = np.trace(np.fliplr(self.board)) == 3 * self.curr_player

        return rows_check or cols_check or left_diagonal_check or left_diagonal_check or right_diagonal_check
    
    def update_round(self):
        if self.is_win():
            self.winner = self.curr_player
            self.terminate = True
        
        self.round += 1
        if self.round > 9:
            self.terminate = True
        else:
            self.switch_player()

    def show(self):
        """
            print the board state
        """
        print(self.board)
        