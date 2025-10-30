import numpy as np
import matplotlib.pyplot as plt
import random

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridWorld:
    def __init__(self):
        self.height = 4
        self.width = 12
        self.start_state = (0, 0)
        self.goal_state = (3, 11)
        #self.cliff = [(2, i) for i in range(1, 7)]
        self.cliff_start = (1, 2)
        self.cliff_height = 2
        self.cliff_width = 6 
        self.reset()
        self.prob1 = 0.8  # Probability that the correct action is taken
        self.prob2 = 0.1  # Probability it veers one way
        self.prob3 = 0.1  # Probability it veers the other way

        # Define the cliff, which is a list of tuples (x, y)
        self.cliff = []
        x_start, y_start = self.cliff_start
        for i in range(self.cliff_height):
            for j in range(self.cliff_width):
                x = x_start + i
                y = y_start + j
                if 0 <= x < self.height and 0 <= y < self.width:
                    self.cliff.append((x, y))
        # self.cliff.append((2, 8)) # TODO: Check why this was here (I commented it out - Tanner)

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        """
        Take an action in the environment with stochastic effects.
        
        Args:
            intended_action: The intended direction (0=Up, 1=Right, 2=Down, 3=Left)
        
        Returns:
            next_state (tuple): The next (x, y) location
            reward (int): The reward for the action
            done (bool): True if goal state is reached
        """
        # Define actions perpendicular to the intended action
        perpendicular_actions = {
            UP: [LEFT, RIGHT],
            RIGHT: [UP, DOWN],
            DOWN: [LEFT, RIGHT],
            LEFT: [UP, DOWN]
        }

        # Determine the actual action taken based on probabilities
        action_probs = [self.prob1, self.prob2, self.prob3]
        actual_actions = [action] + perpendicular_actions[action]
        actual_action = random.choices(actual_actions, weights=action_probs, k=1)[0]

        x, y = self.state
        if actual_action == UP:
            x = max(x - 1, 0)
        elif action == RIGHT:
            y = min(y + 1, self.width - 1)
        elif action == DOWN:
            x = min(x + 1, self.height - 1)
        elif action == LEFT:
            y = max(y - 1, 0)

        next_state = (x, y)
        reward = -1
        done = False

        if next_state in self.cliff:
            # Stepping into the cliff keeps the agent in the same state
            next_state = self.state
        elif next_state == self.goal_state:
            done = True

        self.state = next_state
        return next_state, reward, done

    def get_actions(self):
        return [UP, RIGHT, DOWN, LEFT]

    def render(self):
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.cliff:
            grid[x][y] = 'C'
        sx, sy = self.state
        gx, gy = self.goal_state
        grid[sx][sy] = 'A'
        grid[gx][gy] = 'G'
        for row in grid:
            print(' '.join(row))
        print()