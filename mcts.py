import numpy as np
import random

from go import Go

def get_move_distribution(state, temperature):
    # stub
    return random.randint(0,361)

class MCTS(object):
    class Node(object):
        visits_count = np.ones(361)
        action_value = np.ones(361)
        mean_action_value = np.ones(361)
        move_probability = np.ones(361)
        children = list()
        


    def __init__(self, state):
        self.current_state = state
        self.root_node = None

if __name__ == "__main__":
    env = Go()
    state = env.reset()
    simulations_per_move = 1600
    total_moves = 1600
    high_temperature_moves = 30
    high_temperature = 1
    normal_temperature = 0


    for move in range(high_temperature_moves):
        action = get_move_distribution(state,high_temperature)

        # 'seed' the game by making random moves
        pass

    for move in range(total_moves - high_temperature_moves):
        # play the game using MCTS with lookahead
        pass
    else:
        # game ended before winner is decided, use scoring
        pass
    