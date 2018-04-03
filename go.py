import gym
import numpy as np

# game gives reward of -1 or +1
# ends when both players pass their turn, after a number of times or player resigns

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
        step
        reset
        render
        close
        seed
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
    metadata = {'render.modes': []}
    reward_range = (-1, 1)
    spec = None

    # Set these in ALL subclasses
    action_space = None
    observation_space = None

    def step(self, action):
        """ Place a stone on the board in the color of the current player.
        Args:
            action (object): a 2-tupel of integers in range [0,18]
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
            if white_board_state[x,y] or black_board_state[x,y]:
                raise Exception("Can't move on top of another stone")

            if self.turn == "white":
                white_board_state[y,x] = 1.0
                self.turn = "black"
            else: 
                black_board_state[y,x] = 1.0
                self.turn = "white"

            

        self.move_history.append(action)
        self.white_history.append(white_board_state)
        self.black_history.append(black_board_state)
        
        observation = self.get_state()
        reward = self.white_reward if self.turn == "white" else self.black_reward
        return observation, reward, False, None

    def reset(self, root=None):
        white_board = np.zeros((19,19))
        black_board = np.zeros((19,19))
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
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:
        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
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
        raise NotImplementedError

    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        logger.warn("Could not seed environment %s", self)
        return

    @property
    def unwrapped(self):
        """Completely unwrap this env.
        Returns:
            gym.Env: The base non-wrapped gym.Env instance
        """
        return self

    def __str__(self):
        if self.spec is None:
            return '<{} instance>'.format(type(self).__name__)
        else:
            return '<{}<{}>>'.format(type(self).__name__, self.spec.id)

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

    def capture_pieces(self, black_board, white_board):
        """Remove all pieces from the board that have 
        no liberties.
        """

        has_stone = np.logical_or(black_board,white_board).flatten()
        white_liberties = np.zeros(range(19*19))
        black_liberties = np.zeros(range(19*19))

        search_index = 0
        liberties_end = 0
        indexes = np.array(range(19*19))

        while search_index < 19*19:
            idx = indexes[search_index]
            x, y = np.unravel_index(idx,(19,19))
            new_liberty = False

            if not has_stone[search_index]:
                # empty field -- is liberty for both
                white_liberties[idx] = 1.0
                black_liberties[idx] = 1.0
                new_liberty = True
            elif black_board[y,x] and self.has_liberty(idx, has_stone, black_liberties):
                black_liberties[idx] = 1.0
                new_liberty = True
            if white_board[y,x] and self.has_liberty(idx,has_stone,white_liberties):
                white_liberties[idx] = 1.0
                new_liberty = True
            
            if new_liberty:
                indexes[liberties_end], indexes[search_index] = indexes[search_index], indexes[liberties_end]
                liberties_end += 1
                search_index = liberties_end
            else:
                search_index += 1

    def has_liberty(self,idx, has_stone, known_liberties):
        """Checks the local surrounding for liberties.
        If False, it means we can't tell from the current local surounding if
        the stone has a liberty or not
        """
        (y,x) = np.unravel_index(idx,(19,19))
        local_area = [
            (min(x+1,18),y),
            (max(x-1,0),y),
            (x,min(y+1,18)),
            (x,max(y-1,0))
        ]
        local_area = np.ravel_multi_index(local_area,(19,19))

        local_area_free = list() # adjacent empty field
        liberty = list() # adjacent stone that is known to have liberties

        for elem in local_area:
            local_area_free.append(has_stone[elem])
            liberty.append(known_liberties[elem])
        
        if any(local_area_free) or any(liberty):
            return True

        return False
