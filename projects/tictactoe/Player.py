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
        self.states = []                 # List of all states encountered
        self.gamma = 0.1                 # Discount factor
        self.epsilon = .001               # Convergence constant
        self.P = None                    # P matrix used in direct computation policy evaluation
        self.g = None                    # g vector used in direct computation policy evaluation

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

    def policy_evaluation_direct(self):
        # use the direct method for policy evaluation V = (I - \gamma*P)^{-1} * g
        self.generate_P()
        self.generate_g()
        num_states = len(self.states)
        vals = np.linalg.inv(np.eye(num_states) - (self.gamma * self.P)) @ self.g
        for i, state in enumerate(self.states):
            self.V[state] = vals[i]

    def generate_P(self):
        num_states = len(self.states)
        self.P = np.zeros((num_states, num_states))
        for i, state in enumerate(self.states):
            game = TicTacToe()
            game.board = list(state)
            if state == ('O', ' ', 'X', 'O', ' ', 'X', ' ', ' ', 'X'):
                pass
            if game.result() != 'not done':
                self.P[i, i] = 1
                continue
            if state not in self.policy:
                self.policy[state] = random.choice(game.empty_squares())
            action = self.policy[state]

            game.make_move(action, self.letter)
            if game.result() != 'not done':
                self.P[i, i] = 1
                continue

            opponent_possible_actions = game.empty_squares()
            for opponent_action in opponent_possible_actions:
                game_copy = TicTacToe()
                game_copy.board = game.board.copy()
                game_copy.make_move(opponent_action, self.opponent_letter)
                new_state = tuple(game_copy.board)
                j = self.states.index(new_state)
                self.P[j, i] = 1 / len(opponent_possible_actions)

    def generate_g(self):
        num_states = len(self.states)
        self.g = np.zeros((num_states, 1))
        for i, state in enumerate(self.states):
            game = TicTacToe()
            game.board = list(state)
            res = game.result()
            if res == self.opponent_letter:
                self.g[i, 0] = -1
                continue
            elif res == 'T':
                continue

            action = self.policy[state]
            game.make_move(action, self.letter)
            if game.result() == self.letter:
                self.g[i, 0] = 1

    def policy_evaluation_mc(self, N, T):
        for state in self.states:
            self.V[state] = 0
            for n in range(N):
                game = TicTacToe()
                game.board = list(state)
                g_n = 0
                res = 'not done'
                player = self.letter
                for t in range(T):
                    # if state is a terminal state, then get reward
                    if res == 'not done':
                        res = game.result()
                    if res == self.letter:
                        g_n += self.gamma**t
                        continue
                    if res == self.opponent_letter:
                        g_n -= self.gamma**t
                        continue
                    if res == 'T':
                        continue

                    # if not, then do a move
                    if player == self.letter:
                        new_state = tuple(game.board)
                        action = self.policy[new_state]
                    else:
                        action = random.choice(game.empty_squares())
                    game.make_move(action, player)
                    if player == self.letter:
                        player = self.opponent_letter
                    else:
                        player = self.letter
                self.V[state] = (1 - self.eta(n)) * self.V[state] + self.eta(n) * g_n

    def eta(self, n):
        return 1/(n+1)

    def policy_improvement(self):
        # Policy Improvement
        for state in self.V:
            game = TicTacToe()
            game.board = list(state)
            res = game.result()
            if res != 'not done':
                continue  # Skip terminal states

            if state == ('O', ' ', 'X', ' ', ' ', ' ', ' ', ' ', 'X'):
                pass

            best_value = float('-inf')
            best_action = None
            for action in game.empty_squares():
                # Simulate taking the action
                game_copy = TicTacToe()
                game_copy.board = list(state)
                game_copy.make_move(action, self.letter)
                new_state = tuple(game_copy.board)

                if game_copy.result() == self.letter:
                    best_action = action
                    break

                opponent_possible_moves = game_copy.empty_squares()
                v_sum = 0
                for opponent_move in opponent_possible_moves:
                    new_state2 = new_state[:opponent_move] + (self.opponent_letter,) + new_state[opponent_move+1:]
                    v_sum += self.V[new_state2]
                value = self.gamma * v_sum / len(opponent_possible_moves)

                if value > best_value:
                    best_value = value
                    best_action = action

            if best_action is not None:
                self.policy[state] = best_action

    def train(self, N=10, T=10):
        # Initialize states encountered
        self.states = []
        # Generate all possible states (for Tic-Tac-Toe, this is feasible)
        print("generating all states...")
        self.generate_all_states(TicTacToe(), 'X')

        i = 0
        self.v_for_plotting = []

        # randomly choose policy
        for state in self.states:
            game = TicTacToe()
            game.board = list(state)
            res = game.result()
            if res == 'not done':
                self.policy[state] = random.choice(game.empty_squares())

        print('starting to train player...')
        while True:
            V_last = self.V.copy()
            # self.policy_evaluation_mc(N, T)
            self.policy_evaluation_direct()
            done = True
            num_changes_v = 0
            for state in self.V:
                if state not in V_last:
                    # print('missing state {}'.format(state))
                    num_changes_v += 1
                    done = False
                    continue
                if abs(self.V[state] - V_last[state]) >= self.epsilon:
                    done = False
                    num_changes_v += 1
                    # break
            if done:
                break

            policy_last = self.policy.copy()
            self.policy_improvement()
            num_changes_p = 0
            what_changed = {}
            for state in self.policy:
                if self.policy[state] != policy_last[state]:
                    num_changes_p += 1
                    what_changed[state] = (self.policy[state], policy_last[state])
            i += 1
            if i % 1 == 0:
                print('on iteration {}, made {} changes to policy, made {} changes to value, state has value {}'\
                      .format(i, num_changes_p, num_changes_v, self.V[('X', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ')]))

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

