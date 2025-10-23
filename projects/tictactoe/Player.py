# Player.py
import random
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

    def __init__(self, letter):
        super().__init__(letter)
        print('finding optimal policy...')
        game = TicTacToe()
        self.policy = {}
        self.explore('X', game)
        print('done')

    def explore(self, player, game):
        res = game.result()
        if res == 'not done':
            best_move = ''
            best_score = -float('inf') if player == self.letter else float('inf')
            for move in game.empty_squares():
                new_game = TicTacToe()
                new_game.board = game.board.copy()
                new_game.make_move(move, player)
                if player == self.letter:
                    new_player = self.opponent_letter
                else:
                    new_player = self.letter
                score = self.explore(new_player, new_game)
                if player == self.letter and score > best_score:
                    best_score = score
                    best_move = move
                elif player == self.opponent_letter and score < best_score:
                    best_score = score
                    best_move = move
            if player == self.letter:
                state = tuple(game.board)
                self.policy[state] = best_move
            return best_score

        elif res == self.letter:
            return 1
        elif res == self.opponent_letter:
            return -1
        else:
            return 0

    def get_move(self, game):
        state = tuple(game.board)
        return self.policy[state]


class RLPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)
        self.V = {}                      # State-value function
        self.policy = {}                 # Policy mapping state to action

    ############################################################################
    # this is the function you will implement policy iteration with Monte-Carlo
    # simulation for the policy evaluation step
    ##########################################################
    def train(self, eta_type = 'standard', N = 1000, gamma = 0.1, epsilon = .001, opponent_policy = 'random', max_iterations = 100):
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
