import sgf
import glob
import numpy as np
from go import Go
from shutil import copy
import random
 
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


def generate_split(location, test_percent=0.05, validation_percent=0.05):
    replays = glob.glob(location)
    training, validation, test = calculate_split_data(replays, test_percent, validation_percent)

    for replay in training:
            copy(replay, "replays/training_set/")
    for replay in validation:
            copy(replay, "replays/validation_set/")
    for replay in test:
            copy(replay, "replays/test_set/")


def loadData(location):
    replays = glob.glob(location)
    positions = "abcdefghijklmnopqrs"

    game_data = list()
    for replay in replays:
        with open(replay, "r") as f:
            collection = sgf.parse(f.read())

        for game in collection:
            result = game.root.properties["RE"]

            initial_stones = []
            for key, value in game.root.properties:
                if key == "AB":
                    initial_stones.append(value[0])

            sim = Go()
            if initial_stones:
                sim.reset(initial_stones=initial_stones)
            else:
                sim.reset()

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
                    action = np.ravel_multi_index((y, x), (19, 19))

                sim.step(action)
                img = sim.render('rgb_array')
            game_data.append({"game": sim, "result": result})
    return game_data


class ReplayQueue(Sequence):
    def __init__(self, queue_size=2500):
        self.queue_size = queue_size
        self.replays = list()
        self.results = list()

    def insert(self, replay_file, result_file):
        replay = np.load(replay_location, mmap_mode="r")
        result = np.load(result)
        self.replays.append(replay)
        self.results.append(result)
        if len(self.replays) > self.queue_size:
            self.replays.pop(0)
            self.results.pop(0)

    def __getitem__(self, idx):
        if len(self) <= self.batch_size:
            acc_moves = np.cumsum([0] + self.moves_per_game())
            replay_idx = np.argmax(acc_moves > idx) - 1
            move_idx = idx - self.accumulated_position[replay_idx]

            example = self.replays[replay_idx][:::move_idx]
            label = self.results[replay_idx]
            return (example, label)
        else:
            raise IndexError("Requested Element at the end of batch")

    def __len__(self):
        return sum(self.moves_per_game())

    def __iter__(self):
        return self

    def moves_per_game(self):
        return [replay.shape[4] for replay in self.replays]


def create_dataset(training_path, test_path, validation_path):
    dataset = {
        "training":   {"location": training_path},
        "test":       {"location": test_path},
        "validation": {"location": validation_path}
    }

    for key, storage in dataset.items():
        print("--- Loading %s Data ---" % key)
        label_location = locations / "results" / "*.npy"
        labels = glob.glob(label_location)

        example_location = locations / "replays" / "*.npy"
        examples = glob.glob(example_location)

        feeder = ReplayQueue()
        for replay, result in zip(examples, labels):
            feeder.insert(replay, result)
        storage["feeder"] = feeder

    return dataset


def main():
    all_data_path = "replays/all_replays/*.sgf"
    training_path = "replays/training_set/*.sgf"
    validation_path = "replays/validation_set/*.sgf"
    test_path = "replays/test_set/*.sgf"

    # generate_split(all_data_path,test_percent=0.01,validation_percent=0.005)

    training_data = loadData(training_path)
    validation_data = loadData(validation_path)
    test_data = loadData(test_path)

    generator = SupervisedGoBatches(training_data, 512)
    print("There are %d training batches" % len(generator))

    generator = SupervisedGoBatches(validation_data, 512)
    print("There are %d validation batches" % len(generator))

    generator = SupervisedGoBatches(test_data, 512)
    print("There are %d test batches" % len(generator))


if __name__ == "__main__":
    main()
