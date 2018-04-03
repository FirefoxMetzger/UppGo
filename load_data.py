import sgf
import glob
import numpy as np
from go import Go
from shutil import copy, rmtree
import os

from keras.utils import Sequence

def calculate_split_data(data, test_percent, validation_percent):
    training_percent = 1-test_percent-validation_percent
    total_games = len(data)

    num_validation = int(np.ceil(total_games * validation_percent))
    num_test = int(np.ceil(total_games * test_percent))
    num_training = total_games - num_validation - num_test

    order = np.random.permutation(total_games)
    training_games = order[:num_training]
    validation_games = order[num_training:num_training+num_validation]
    test_games = order[num_training+num_validation:]

    training = [data[i] for i in training_games]
    validation = [data[i] for i in validation_games]
    test = [data[i] for i in test_games]

    return training, validation, test

def generate_split(location, test_percent=0.05, validation_percent=0.05, delete_existing=True):
    replays = glob.glob(location)
    data_split = calculate_split_data(replays, test_percent, validation_percent)

    dataset_locations = [
        "replays/training_set/",
        "replays/validation_set/",
        "replays/test_set/"
    ]

    for idx in range(3):
        save_path = dataset_locations[idx]
        rmtree(save_path)
        os.mkdir(save_path)
        for replay in data_split[idx]:
                copy(replay, save_path)

def check_replay_integrity(game):
    try:
        game.root.properties["RE"]
    except KeyError:
        return False
    if game.rest is None:
        return False
 
    return True

def loadSingleDataFile(replay):
    positions = "abcdefghijklmnopqrs"

    with open(replay, "r") as f:
        collection = sgf.parse(f.read())

    if not collection:
        print("No games in replay %s" % replay)
        return list()

    game_data = list()
    for game in collection:
        if not check_replay_integrity(game):
            print("Corrupt Replay at %s" % replay)
            continue

        sim = Go()
        sim.reset(game.root)
        for move in game.rest:
            try:
                stone_position = move.properties["B"][0]
            except KeyError:
                stone_position = move.properties["W"][0]
            
            if stone_position == "":
                action = 361
            else:
                x = positions.index(stone_position[0])
                y = positions.index(stone_position[1])
                action = np.ravel_multi_index((y,x),(19,19))
            
            sim.step(action)
        game_data.append(sim)

    return game_data

def loadData(location):
    replays = glob.glob(location)

    game_data = list()
    for replay in replays:
        replay_data = loadSingleDataFile(replay)
        for game in replay_data:
            game_data.append(game)

    return game_data

class SupervisedGoBatches(Sequence):
    def __init__(self, initial_games, batch_size):
        self.batch_size = batch_size
        self.new_games = list()
        self.current_games = initial_games

        moves_per_game = list()
        for game in initial_games:
            moves_per_game.append(len(game.move_history))
        self.moves_per_game = np.array(moves_per_game)
        self.total_examples = np.sum(moves_per_game)
        self.accumulated_position = np.cumsum(np.hstack((0,moves_per_game)))

        # the first 1 million prime numbers
        primes = np.loadtxt("primes1.txt").flatten()
        
        #smallest prime bigger then total_examples
        modulus_idx = np.argmax(primes > self.total_examples)
        print("Entering %d residue class ring." % modulus_idx)
        self.modulus = primes[modulus_idx]

        self.step_size = np.random.random_integers(0,self.modulus)
        self.start = np.random.random_integers(0,self.modulus)
        self.current_element = self.start

        self.batch_indexes = [self.start]
        
    def __getitem__(self, batch_idx):
        if batch_idx > len(self):
            raise IndexError("%i is out of bounds for %i batches" % (batch_idx, len(self)))

        states = np.zeros((self.batch_size,19,19,17))
        actions = np.zeros(self.batch_size, dtype=int)
        rewards = np.zeros(self.batch_size)

        current_element = self.begining_of_batch(batch_idx)

        for idx in range(self.batch_size):
            game_idx = np.argmax(self.accumulated_position > current_element) - 1
            move_idx = current_element - self.accumulated_position[game_idx]

            state, action, reward = self.current_games[game_idx].get_history_step(move_idx)
            states[idx,:,:,:] = state
            actions[idx] = action
            rewards[idx] = reward

            current_element = self.next_element(current_element)

        if batch_idx == len(self.batch_indexes) - 1:
            self.batch_indexes.append(current_element)

        # encode actions as 1-hot
        hot_actions = np.zeros((actions.shape[0], 362))
        hot_actions[np.arange(actions.shape[0],dtype=int),actions] = 1

        return states, {"policy":hot_actions, "value":rewards}

    def begining_of_batch(self, batch_idx):
        while len(self.batch_indexes) - 1 < batch_idx:
            current = self.batch_indexes[-1]
            for _ in range(self.batch_size):
                current = self.next_element(current)
            self.batch_indexes.append(self.current_element)
        
        return self.batch_indexes[batch_idx]

    def __len__(self):
        return int(np.floor(self.total_examples/self.batch_size))

    def next_element(self, start_element):
        candidate = (start_element + self.step_size) % self.modulus
        while candidate > self.total_examples:
            candidate = (candidate + self.step_size) % self.modulus
        
        return int(candidate)

    def on_epoch_end(self):
        self.reset_generator()

    def reset_generator(self):
        self.step_size = np.random.random_integers(0,self.modulus)
        self.start = np.random.random_integers(0,self.modulus)
        self.batch_indexes = [self.start]

if __name__ == "__main__":
    all_data_path = "replays/all_replays/*.sgf"
    training_path = "replays/training_set/*.sgf"
    validation_path = "replays/validation_set/*.sgf"
    test_path = "replays/test_set/*.sgf"

    generate_split(all_data_path,test_percent=0.01,validation_percent=0.005)

    training_data = loadData(training_path)
    validation_data = loadData(validation_path)
    test_data = loadData(test_path)

    generator = SupervisedGoBatches(training_data, 512)
    print("There are %d training batches" % len(generator))

    prev_state_batch, prev_labels_batch = generator[0]
    for state_batch, labels_batch in generator:
        same = False
        for state, prev_state in zip(state_batch, prev_state_batch):
            if np.all(state == prev_state):
                same = True
                break
        else:
            for label, prev_label in zip(labels_batch, prev_labels_batch):
                print(label["policy"])
                if np.all(label["policy"] == prev_label["policy"]):
                    same = True
                    break
                if np.all(label["value"] == prev_label["value"]):
                    same = True
                    break
        prev_labels_batch = labels_batch
        prev_state_batch = state_batch
        if same:
            print("Same Batch Found Twice")

    generator = SupervisedGoBatches(validation_data, 512)
    print("There are %d validation batches" % len(generator))
    
    generator = SupervisedGoBatches(test_data, 512)
    print("There are %d test batches" % len(generator))