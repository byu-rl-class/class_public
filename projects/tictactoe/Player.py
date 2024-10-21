# Player.py

import random
import math
from TicTacToe import TicTacToe
import numpy as np


class Player:
    def __init__(self, letter):
        # letter is 'X' or 'O'
        self.letter = letter
        self.opponent_letter = 'O' if self.letter == 'X' else 'X'

    def get_move(self, game):
        pass


class RandomPlayer(Player):
    def get_move(self, game):
        # Randomly choose a valid move
        square = random.choice(game.empty_squares())
        return square


class HumanPlayer(Player):
    def get_move(self, game):
        # Ask user for input
        valid_square = False
        val = None
        while not valid_square:
            square = input(f"Your turn ({self.letter}). Input move (0-8): ")
            try:
                val = int(square)
                if val not in game.empty_squares():
                    raise ValueError
                valid_square = True
            except ValueError:
                print("Invalid move. Try again.")
        return val


class OptPlayer(Player):
    def get_move(self, game):
        # Use minimax algorithm to choose the best move
        if len(game.empty_squares()) == 9:
            # If first move, pick a corner
            square = random.choice([0, 2, 6, 8])
        else:
            # Get the best move
            square = self.minimax(game, self.letter)['position']
        return square

    def minimax(self, state, player):
        max_player = self.letter  # Yourself
        other_player = self.opponent_letter

        # Base case: check for previous move winner
        if state.current_winner == other_player:
            return {'position': None,
                    'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else
                            -1 * (state.num_empty_squares() + 1)}
        elif not state.empty_squares():
            return {'position': None, 'score': 0}

        # Initialize dictionaries
        if player == max_player:
            best = {'position': None, 'score': -math.inf}  # Maximize the max_player
        else:
            best = {'position': None, 'score': math.inf}   # Minimize the other player

        for possible_move in state.empty_squares():
            # Make a move
            state.make_move(possible_move, player)
            sim_score = self.minimax(state, other_player)  # Simulate the game

            # Undo the move
            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move

            # Update the dictionaries
            if player == max_player:
                if sim_score['score'] > best['score']:
                    best = sim_score  # Replace best
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score  # Replace best
        return best


class RLPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
        self.V = {}                      # State-value function
        self.policy = {}                 # Policy mapping state to action
        self.states = []                 # List of all states encountered
        self.gamma = 1                   # Discount factor
        self.epsilon = .0001             # Convergence constant
       
    ############################################################################
    # this is the function you will implement policy iteration with Monte-Carlo
    # simulation for the policy evaluation step
    ##########################################################
    def train(self):
        pass

    def get_state(self, game):
        # Convert the board to a tuple (immutable and hashable)
        return tuple(game.board)

    def get_move(self, game):
        state = self.get_state(game)
        available_moves = game.empty_squares()

        # Ensure the policy has an action for the current state
        if state not in self.policy:
            self.policy[state] = random.choice(available_moves)

        return self.policy[state]
        
    def generate_all_states(self, game, player_turn):
        # Generate all possible states where it's RLPlayer's turn
        state = self.get_state(game)

        # Only add the state if it's RLPlayer's turn
        if player_turn == self.letter and state not in self.states:
            self.states.append(state)

        # Check for terminal state
        if game.current_winner or game.is_full():
            return

        for action in game.empty_squares():
            # Make a copy of the game
            game_copy = TicTacToe()
            game_copy.board = game.board.copy()
            game_copy.current_winner = game.current_winner

            # Make a move
            game_copy.make_move(action, player_turn)

            # Switch player turn
            next_player_turn = 'O' if player_turn == 'X' else 'X'

            # Recursively generate states
            self.generate_all_states(game_copy, next_player_turn)

