import numpy as np
import matplotlib.pyplot as plt
import random

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

        self.cliff = []
        x_start, y_start = self.cliff_start
        for i in range(self.cliff_height):
            for j in range(self.cliff_width):
                x = x_start + i
                y = y_start + j
                if 0 <= x < self.height and 0 <= y < self.width:
                    self.cliff.append((x, y))
        self.cliff.append((2, 8))

    def reset(self):
        self.state = self.start_state
        return self.state

    def step(self, action):
        # Define perpendicular actions
        perpendicular_actions = {
            0: [3, 1],  # Up -> Left, Right
            1: [0, 2],  # Right -> Up, Down
            2: [3, 1],  # Down -> Left, Right
            3: [0, 2]   # Left -> Up, Down
        }

        # Determine the actual action taken based on probabilities
        action_probs = [self.prob1, self.prob2, self.prob3]
        actual_actions = [action] + perpendicular_actions[action]
        actual_action = random.choices(actual_actions, weights=action_probs, k=1)[0]

        x, y = self.state
        if actual_action == 0:  # Up
            x = max(x - 1, 0)
        elif actual_action == 1:  # Right
            y = min(y + 1, self.width - 1)
        elif actual_action == 2:  # Down
            x = min(x + 1, self.height - 1)
        elif actual_action == 3:  # Left
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
        return [0, 1, 2, 3]  # Up, Right, Down, Left