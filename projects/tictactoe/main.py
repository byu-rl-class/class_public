from Player import HumanPlayer, OptPlayer, RandomPlayer, RLPlayer
from TicTacToe import TicTacToe

game = TicTacToe()

# you can use this to test if the environment is working correctly
p1 = HumanPlayer('X')
p2 = OptPlayer('O')

result = game.play(p1, p2, print_game=True)
print(result)

# uncomment this when you're ready to train and play against the RL player
# p2 = OptPlayer('X')
# p1 = RLPlayer('O')
# p1.train(eta_type = 'standard', N = 500, gamma = .1, epsilon = .001, opponent_policy = 'random', max_iterations = 100)

# def evaluate_performance(player, opponent, num_games=100):
#         """Evaluate performance against an opponent"""
#         wins = 0
#         losses = 0
#         ties = 0
        
#         for _ in range(num_games):
#             game = TicTacToe()
#             if player.letter == 'X':
#                 result = game.play(player, opponent, print_game=False)
#             else:
#                 result = game.play(opponent, player, print_game=False)
            
#             if result == player.letter:
#                 wins += 1
#             elif result == opponent.letter:
#                 losses += 1
#             else:
#                 ties += 1
                
#         return {
#             'wins': wins,
#             'losses': losses,
#             'ties': ties,
#             'win_rate': wins / num_games,
#             'loss_rate': losses / num_games,
#             'tie_rate': ties / num_games
#         }

# print(evaluate_performance(p1, p2))