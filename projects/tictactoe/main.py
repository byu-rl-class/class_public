from Player import HumanPlayer, OptPlayer, RandomPlayer, RLPlayer
from TicTacToe import TicTacToe

game = TicTacToe()
p1 = HumanPlayer('X')
p2 = OptPlayer('O')

# uncomment this when you're ready to train and play against the RL player
# p2 = RLPlayer('O')
# p2.train()
game.play(p1, p2)
