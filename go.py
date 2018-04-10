import gym
import numpy as np

# game gives reward of -1 or +1
# ends when both players pass their turn, after a number of times or player resigns

def get_neighboors(y,x,board_shape):
    neighboors = list()

    if y > 0:
        neighboors.append((y-1,x))
    if y < board_shape[0] - 1:
        neighboors.append((y+1,x))
    if x > 0:
        neighboors.append((y,x-1))
    if x < board_shape[1] - 1:
        neighboors.append((y,x+1))

    return neighboors

def test_group(board,opponent_board,y,x, current_group):
    """ Assume the current group is captured. Find it via flood fill
    and if an empty neighboor is encountered, break (group is alive).

    board - 19x19 array of player's stones
    opponent_board - 19x19 array of opponent's stones
    x,y - position to test
    current_group - tested stones in player's color

    """

    pos = (y,x)

    if board[pos] and not current_group[pos]:
        has_liberties = False
        current_group[pos] = 1.0

        neighboors = get_neighboors(y,x,board.shape)

        for yn, xn in neighboors:
            has_liberties = test_group(board,opponent_board,yn,xn,current_group)
            if has_liberties:
                return True
        return False
    return not opponent_board[pos]

class Go(gym.Env):
    """A simple Go environment that takes moves for each player in alternating order.
    - There is no komi
    - white makes the first move
    - it does not guard against repeated board states
    """

    """The main OpenAI Gym class. It encapsulates an environment with
    arbitrary behind-the-scenes dynamics. An environment can be
    partially or fully observed.
    The main API methods that users of this class need to know are:
        render
        close
    And set the following attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
    """
    def __init__(self):
        self.turn = "black"
        self.black_history = list()
        self.white_history = list()
        self.white_reward = 0
        self.black_reward = 0
        self.move_history = list()


    # Set this in SOME subclasses
    metadata = {'render.modes': ["human"]}
    reward_range = (-1, 1)
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        """ Place a stone on the board in the color of the current player.
        Args:
            action (object): raveled index of the board position [19,19] or 361 for pass
        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        white_board_state = self.white_history[-1].copy()
        black_board_state = self.black_history[-1].copy()

        if not action == 361: # 361 is the pass action
            (y,x) = np.unravel_index(action, (19,19))
            if white_board_state[y,x] or black_board_state[y,x]:
                self.render()
                print("Desired Move: %s: (%d,%d)" % (self.turn, y,x))
                raise Exception("Can't move on top of another stone")

            if self.turn == "white":
                white_board_state[y,x] = 1.0
                self.capture_pieces(black_board_state, white_board_state, y,x)
                self.turn = "black"
            else: 
                black_board_state[y,x] = 1.0
                self.capture_pieces(black_board_state, white_board_state, y,x)
                self.turn = "white"

        #self.capture_pieces(black_board_state,white_board_state)

        self.move_history.append(action)
        self.white_history.append(white_board_state)
        self.black_history.append(black_board_state)
        
        observation = self.get_state()
        reward = self.white_reward if self.turn == "white" else self.black_reward
        return observation, reward, False, None

    def reset(self, root=None):
        white_board = np.zeros((19,19), dtype=bool)
        black_board = np.zeros((19,19), dtype=bool)
        self.move_history = list()

        if not root: # reset in self-play mode -- unknown result
            self.white_reward = 0
            self.black_reward = 0
        
        else: # reset in supervised mode -- env used for observation generation
            positions = "abcdefghijklmnopqrs"

            # get the game winner
            result = root.properties["RE"][0]
            if result[0] == "W":
                self.white_reward = 1
                self.black_reward = -1
            elif result[0] == "B":
                self.white_reward = -1
                self.black_reward = 1
            else:
                raise Exception("Failed to determine winner")

            # set initial stones for black, if any
            try:
                komi_positions = root.properties["AB"]
                for position in komi_positions:
                    x = positions.index(position[0])
                    y = positions.index(position[1])
                    black_board[y,x] = 1.0
            except KeyError:
                pass # no stones for black, "equal match"

        self.black_history = list()
        for _ in range(8):
            self.black_history.append(black_board.copy())

        self.white_history = list()
        for _ in range(8):
            self.white_history.append(white_board.copy())

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
        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        Example:
        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}
            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode is 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """

        if mode == "human":
            white_board = self.white_history[-1]
            black_board = self.black_history[-1]

            for y in range(black_board.shape[0]):
                if y < 10:
                    row = "0"+str(y)
                else:
                    row = str(y)
                for x in range(black_board.shape[1]):
                    if white_board[y,x]:
                        row += "W"
                    elif black_board[y,x]:
                        row += "B"
                    else:
                        row +="-"
                print(row)
        else:
            raise NotImplementedError

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

    def seed(self, seed=None):
        # this is a deterministic environment
        return

    def get_state(self):
        turn = self.turn
        white_history = self.white_history[-8:]
        black_history = self.black_history[-8:]

        state = np.empty((19,19,17))
        if turn == "white":
            state[:,:,0:8] = np.stack(white_history,axis=2)
            state[:,:,8:16] = np.stack(black_history,axis=2)
            state[:,:,16] = 1.0
        else: # black move
            state[:,:,0:8] = np.stack(black_history,axis=2)
            state[:,:,8:16] = np.stack(white_history,axis=2)
            state[:,:,16] = 0.0

        return state

    def get_history_state(self, idx):
        turn = "white" if idx % 2 == 0 else "black"
        white_history = self.white_history[idx:idx+8]
        black_history = self.black_history[idx:idx+8]

        state = np.empty((19,19,17))
        if turn == "white":
            state[:,:,0:8] = np.stack(white_history,axis=2)
            state[:,:,8:16] = np.stack(black_history,axis=2)
            state[:,:,16] = 1.0
        else: # black move
            state[:,:,0:8] = np.stack(black_history,axis=2)
            state[:,:,8:16] = np.stack(white_history,axis=2)
            state[:,:,16] = 0.0

        return state

    def get_history_step(self,idx):
        state = self.get_history_state(idx)
        reward = self.white_reward if idx % 2 == 0 else self.black_reward
        action = self.move_history[idx]

        return state, action, reward

    def capture_pieces(self, black_board, white_board, y,x):
        """Remove all pieces from the board that have 
        no liberties. This function modifies the input variables in place.

        black_board is a 19x19 np.array with value 1.0 if a black stone is
        present and 0.0 otherwise.

        white_board is a 19x19 np.array similar to black_board.

        active_player - the player that made a move
        (x,y) - position of the move

        """

        # only test neighboors of current move (other's will have unchanged
        # liberties)
        neighboors = get_neighboors(y,x,black_board.shape)

        if self.turn == "white":
            board = white_board
            opponent_board = black_board
        else:
            board = black_board
            opponent_board = white_board

        # to test suicidal moves
        original_pos = (y,x)

        # only test adjacent stones in opponent's color
        for pos in neighboors:
            if not opponent_board[pos]:
                continue

            current_group = np.zeros_like(board)
            has_liberties = test_group(opponent_board, board, *pos, current_group)

            if not has_liberties:
                opponent_board[current_group] = 0.0

        current_group = np.zeros_like(board)
        has_liberties = test_group(board, opponent_board, *original_pos, current_group)
        if not has_liberties:
            # actually my replays are not only played with Japanese or Chinese rules
            board[current_group] = 0.0
            #raise Exception("Suicidal moves are illegal in Japanese and Chinese rules!")