import sgf
import glob
import numpy as np
from go import Go
from shutil import copy
import random
from pathlib import Path
from os import makedirs
from shutil import rmtree
from keras.utils import Sequence


def generate_numpy_dataset(source, location):
    for key, data in source.items():
        print("--- Converting %s Set ---" % key)
        examples_location = location / key / "examples"
        labels_location = location / key / "labels"
        actions_location = location / key / "actions"
        makedirs(examples_location, exist_ok=True)
        makedirs(labels_location, exist_ok=True)
        makedirs(actions_location, exist_ok=True)

        for idx, replay in enumerate(data):
            with open(replay, "r") as f:
                collection = sgf.parse(f.read())

            examples, actions, labels = sgf_to_npy(collection)
            np.save(examples_location / ("%d.npy" % idx), examples)
            np.save(labels_location / ("%d.npy" % idx), labels)
            np.save(actions_location / ("%d.npy" % idx), actions)


def sgf_to_npy(sgf_collection):
    positions = "abcdefghijklmnopqrs"
    game = sgf_collection[0]
    black_win = True if game.root.properties["RE"][0] == "B" else False

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

    examples = np.zeros((19, 19, 17, len(sim)), dtype=bool)
    actions = np.zeros((362, len(sim)))
    for idx in range(len(sim)):
        state, action_idx = sim.get_history_step(idx)
        examples[:, :, :, idx] = state
        actions[action_idx, idx] = 1

    results = np.ones(len(sim))
    if black_win:
        results[1::2] = -1
    else:
        results[::2] = -1

    return examples, actions, results


class ReplayQueue(Sequence):
    def __init__(self, queue_size=2500):
        self.queue_size = queue_size
        self.replays = list()
        self.results = list()

    def insert(self, replay_file, result_file):
        replay = np.load(replay_location, mmap_mode="r")
        result = np.load(result, mmap_mode="r")
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

            example = self.replays[replay_idx][:, :, :, move_idx]
            label = self.results[replay_idx][move_idx]
            return (example, label)
        else:
            raise IndexError("Requested Element at the end of batch")

    def __len__(self):
        return sum(self.moves_per_game())

    def __iter__(self):
        return self

    def moves_per_game(self):
        return [replay.shape[3] for replay in self.replays]


def create_dataset(training_path, test_path, validation_path):
    dataset = {
        "training":   {"location": training_path},
        "test":       {"location": test_path},
        "validation": {"location": validation_path}
    }

    for key, storage in dataset.items():
        print("--- Loading %s Data ---" % key)
        label_location = storage["location"] / "examples"
        example_location = storage["location"] / "labels"

        num_examples = len(glob.glob(example_location / "*.npy"))
        data_pairs = [(example_location / ("%d.npy" % idx),
                      label_location / ("%d.npy" % idx))
                      for idx in range(num_examples)]

        feeder = ReplayQueue()
        for replay, result in data_pairs:
            feeder.insert(replay, result)
        storage["feeder"] = feeder

    return dataset


def main():
    try:
        rmtree("numpy")
    except FileNotFoundError:
        print("Numpy folder did not exist in working directory")
    makedirs("numpy")

    all_data_path = "replays/all_replays/*.sgf"
    data = glob.glob(all_data_path)

    # filter replays
    random.shuffle(data)

    # split data
    data_split = {
        "training": data[350:],
        "test": data[100:450],
        "validation": data[:100],
    }

    # create numpy dataset
    generate_numpy_dataset(data_split, Path("numpy"))


if __name__ == "__main__":
    main()
