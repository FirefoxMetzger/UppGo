import gym
from gym.spaces import Discrete, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvas

# game gives reward of -1 or +1
# ends when both players pass their turn, after a number of times or player resigns


def get_neighboors(y, x, board_shape):
    neighboors = list()

    if y > 0:
        neighboors.append((y-1, x))
    if y < board_shape[0] - 1:
        neighboors.append((y+1, x))
    if x > 0:
        neighboors.append((y, x-1))
    if x < board_shape[1] - 1:
        neighboors.append((y, x+1))

    return neighboors


def test_group(board, opponent_board, y, x, current_group):
    """ Assume the current group is captured. Find it via flood fill
    and if an empty neighboor is encountered, break (group is alive).

    board - 19x19 array of player's stones
    opponent_board - 19x19 array of opponent's stones
    x,y - position to test
    current_group - tested stones in player's color

    """

    pos = (y, x)

    if board[pos] and not current_group[pos]:
        has_liberties = False
        current_group[pos] = 1.0

        neighboors = get_neighboors(y, x, board.shape)

        for yn, xn in neighboors:
            has_liberties = test_group(board, opponent_board, yn, xn, 
                                       current_group)
            if has_liberties:
                return True
        return False
    return not opponent_board[pos]


class Go(gym.Env):
    """A simple Go environment that takes moves for each player in alternating order.
    """
    board_size = (19, 19)

    def __init__(self):
        self.is_turn_white = False
        self.black_history = list()
        self.white_history = list()
        self.action_history = list()

    # Set this in SOME subclasses
    metadata = {'render.modes': ["human", "rgb_array", "ansi"]}
    reward_range = (-1, 1)
    spec = None

    # Set these in ALL subclasses
    action_space = gym.spaces.Discrete(362)
    observation_space = None  # a 19x19x17 numpy array

    def step(self, action):
        """ Place a stone on the board in the color of the current player.
        Args:
            action (object): raveled index of the board position [19,19] or 361 for pass
        Returns:
            observation (object): 19x19x17 numpy array (using AlphaGo encoding)
            reward (float) : 0
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): {}
        """

        white_board_state = self.white_history[-1].copy()
        black_board_state = self.black_history[-1].copy()

        if not action == 361:  # 361 is the pass action
            (y, x) = np.unravel_index(action, self.board_size)
            if white_board_state[y, x] or black_board_state[y, x]:
                self.render()
                if self.is_turn_white:
                    print("Desired Move: %s: (%d,%d)" % ("white", y, x))
                else:
                    print("Desired Move: %s: (%d,%d)" % ("black", y, x))
                raise Exception("Can't move on top of another stone")

            if self.is_turn_white:
                white_board_state[y, x] = 1.0
                self.capture_pieces(white_board_state, black_board_state, y, x)
                self.is_turn_white = False
            else:
                black_board_state[y, x] = 1.0
                self.capture_pieces(black_board_state, white_board_state, y, x)
                self.is_turn_white = True

        self.action_history.append(action)
        self.white_history.append(white_board_state)
        self.black_history.append(black_board_state)

        observation = self.get_state()
        return observation, 0, False, {}

    def reset(self, initial_stones=None):
        self.action_history = list()
        self.black_history = list()
        self.white_history = list()

        white_board = np.zeros(self.board_size, dtype=bool)
        black_board = np.zeros(self.board_size, dtype=bool)

        if initial_stones:
            positions = np.unravel_index(initial_stones, self.board_size)
            for y, x in positions:
                black_board[x, y] = True
            self.is_turn_white = True

        for _ in range(8):
            self.black_history.append(black_board.copy())
            self.white_history.append(white_board.copy())
            self.action_history.append(361)

        return self.get_state()

    def render(self, mode='human'):
        """Renders the environment.
        Supported Modes:
        - human: Print the current board state in the current terminal
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).
        """

        if mode == "human" or mode == "ansi":
            white_board = self.white_history[-1]
            black_board = self.black_history[-1]

            for y in range(black_board.shape[0]):
                if y < 10:
                    row = "0"+str(y)
                else:
                    row = str(y)
                for x in range(black_board.shape[1]):
                    if white_board[y, x]:
                        row += "W"
                    elif black_board[y, x]:
                        row += "B"
                    else:
                        row += "-"
                print(row)
        elif mode == "rgb_array":
            # draw the grid
            fig = plt.figure()
            canvas = FigureCanvas(fig)
            ax = fig.gca()
            ax.set_facecolor((0.7843, 0.6314, 0.3961))
            ax.grid()
            ax.set_axisbelow(True)
            ax.set_xlim(left=-1, right=20)
            ax.set_xticks(range(0, 20))
            ax.set_ylim((-1, 20))
            ax.set_yticks(range(0, 20))
            for x in range(self.board_size[1]):
                for y in range(self.board_size[0]):
                    if self.white_history[-1][y, x]:
                        circ = plt.Circle((x, y), radius=0.49, color=(1, 1, 1))
                        ax.add_artist(circ)
                        print("white stone")
                    if self.black_history[-1][y, x]:
                        circ = plt.Circle((x, y), radius=0.49, color=(0, 0, 0))
                        ax.add_artist(circ)
                        print("black stone")
            canvas.draw()
            width, height = fig.get_size_inches() * fig.get_dpi()
            img = np.fromstring(canvas.tostring_rgb(),
                                dtype='uint8').reshape((int(height), 
                                                        int(width), 3))
            
            return img
        else:
            raise NotImplementedError

    def close(self):
        return

    def seed(self, seed=None):
        # this is a deterministic environment
        return

    def get_state(self):
        return self.get_history_state(len(self.black_history))

    def get_history_step(self, idx):
        state = self.get_history_state(idx)
        action = self.action_history[idx]

        return state, action

    def get_history_state(self, idx):
        is_turn_black = True if idx % 2 != 0 else False
        white_history = self.white_history[idx-8:idx]
        black_history = self.black_history[idx-8:idx]

        state = np.empty((19, 19, 17))
        if not is_turn_black:
            state[:, :, 0:8] = np.stack(white_history, axis=2)
            state[:, :, 8:16] = np.stack(black_history, axis=2)
        else:  # black move
            state[:, :, 0:8] = np.stack(black_history, axis=2)
            state[:, :, 8:16] = np.stack(white_history, axis=2)
        state[:, :, 16] = is_turn_black

        return state

    def capture_pieces(self, board, opponent_board, y, x):
        """Remove all pieces from the board that have
        no liberties. This function modifies the input variables in place.

        black_board is a 19x19 np.array with value 1.0 if a black stone is
        present and 0.0 otherwise.

        white_board is a 19x19 np.array similar to black_board.

        active_player - the player that made a move
        (x,y) - position of the move

        """
        neighboors = get_neighboors(y, x, self.board_size)
        original_pos = (y, x)

        # only test adjacent stones in opponent's color
        for pos in neighboors:
            if not opponent_board[pos]:
                continue

            current_group = np.zeros_like(board)
            has_liberties = test_group(opponent_board, board, *pos,
                                       current_group)

            if not has_liberties:
                opponent_board[current_group] = 0.0

        current_group = np.zeros_like(board)
        has_liberties = test_group(board, opponent_board, *original_pos, 
                                   current_group)

        if not has_liberties:
            raise Exception("Suicidal moves are illegal in Japanese and Chinese rules!")

if __name__ == "__main__":
    # play against yourself
    board = Go()
