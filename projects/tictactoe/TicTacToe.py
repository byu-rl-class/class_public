# TicTacToe.py

import time

class TicTacToe:
    def __init__(self):
        """
        Initializes the TicTacToe game. 
        The board is a list of 9 empty strings, and the current winner is None.
        """
        self.board = [' ' for _ in range(9)]
        self.current_winner = None  # Keep track of the winner!

    def make_move(self, square, letter):
        # If the move is valid, make the move (assign square to letter)
        # Then return True. If invalid, return False.
        if self.is_valid_move(square, letter):
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def is_valid_move(self, square, letter):
        if self.board[square] == ' ':
            return True
        return False

    def get_valid_moves(self, letter):
        return [i for i, x in enumerate(self.board) if x == ' ' and self.is_valid_move(i, letter)]

    def result(self):
        # Determine the result of the current board state
        # 'X' - player 'X' won
        # 'O' - player 'O' won
        # 'T' - tie
        # 'not done' - game is not over
        winner = self.check_winner()
        if winner:
            return winner
        elif self.is_full():
            return 'T'  # Tie
        else:
            return 'not done'  # Game is not over

    def check_winner(self):
        # Check the current board for a winner without relying on current_winner
        for letter in ['X', 'O']:
            # Check rows
            for row in range(3):
                if all([self.board[row * 3 + i] == letter for i in range(3)]):
                    return letter
            # Check columns
            for col in range(3):
                if all([self.board[col + i * 3] == letter for i in range(3)]):
                    return letter
            # Check diagonals
            if all([self.board[i] == letter for i in [0, 4, 8]]):
                return letter
            if all([self.board[i] == letter for i in [2, 4, 6]]):
                return letter
        return None  # No winner

    def winner(self, square, letter):
        # Check if the player has won
        row_ind = square // 3
        row = self.board[row_ind*3:(row_ind+1)*3]
        if all([s == letter for s in row]):
            return True

        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        if all([s == letter for s in column]):
            return True

        # Check diagonals
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0,4,8]]
            if all([s == letter for s in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2,4,6]]
            if all([s == letter for s in diagonal2]):
                return True

        return False

    def empty_squares(self):
        # Return a list of indices of empty squares
        return [i for i, x in enumerate(self.board) if x == ' ']

    def num_empty_squares(self):
        return len(self.empty_squares())

    def is_full(self):
        return ' ' not in self.board

    def print_board(self):
        # Print the board
        flag = False
        for row in [self.board[i*3:(i+1)*3] for i in range(3)]:
            if flag:
                print(' ---+---+--- ')
            else:
                flag = True
            print('  ' + ' | '.join(row) + '  ')

    def reset(self):
        # Reset the board
        self.board = [' ' for _ in range(9)]
        self.current_winner = None

    def play(self, player_x, player_o, print_game=True):
        # Simulate a game between two players
        if print_game:
            self.print_board_nums()

        letter = 'X'  # Starting letter
        while self.num_empty_squares() > 0 and self.current_winner is None:
            if letter == 'O':
                square = player_o.get_move(self)
            else:
                square = player_x.get_move(self)

            # Make move
            if self.make_move(square, letter):
                if print_game:
                    print(f"{letter} makes a move to square {square}")
                    self.print_board()
                    print('')  # Empty line for better readability

                if self.current_winner:
                    if print_game:
                        print(f"{letter} wins!")
                    return letter  # Return the winner

                # Switch player
                letter = 'O' if letter == 'X' else 'X'

            else:
                if print_game:
                    print("Invalid move. Try again.")

            # Tiny pause to make things easier to read
            if print_game:
                time.sleep(0.5)

        if print_game:
            print("It's a tie!")
        return None  # Return None if it's a tie

    def print_board_nums(self):
        # Prints the board with numbers (for reference)
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        flag = False
        for row in number_board:
            if flag:
                print(' ---+---+--- ')
            else:
                flag = True
            print('  ' + ' | '.join(row) + '  ')

