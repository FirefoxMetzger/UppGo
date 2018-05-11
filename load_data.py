import sgf
import glob
import numpy as np
from go import Go
from shutil import copy, rmtree
import random
from pathlib import Path
from os import makedirs
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

        print("%d Replays created." % idx)


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


def replay_useful(replay):
    with open(replay, "r") as f:
        collection = sgf.parse(f.read())

    try:
        sgf_to_npy(collection)
    except:
        return False
    return True


class ReplayQueue(Sequence):
    def __init__(self, queue_size=2500):
        self.queue_size = queue_size
        self.replays = list()
        self.results = list()
        self.actions = list()

    def insert(self, example_file, action_file, label_file):
        replay = np.load(label_file, mmap_mode="r")
        result = np.load(example_file, mmap_mode="r")
        action = np.load(action_file, mmap_mode="r")
        self.replays.append(replay)
        self.results.append(result)
        self.actions.append(action)
        if len(self.replays) > self.queue_size:
            self.replays.pop(0)
            self.results.pop(0)
            self.actions.pop(0)

    def __getitem__(self, idx):
        if len(self) <= self.batch_size:
            acc_moves = np.cumsum([0] + self.moves_per_game())
            replay_idx = np.argmax(acc_moves > idx) - 1
            move_idx = idx - self.accumulated_position[replay_idx]

            example = self.replays[replay_idx][:, :, :, move_idx]
            label = self.results[replay_idx][move_idx]
            action = self.actions[replay_idx][:, move_idx]
            return (example, action, label)
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
        example_location = storage["location"] / "labels"
        action_location = storage["location"] / "actions"
        label_location = storage["location"] / "examples"

        num_examples = len(glob.glob(str(example_location / "*.npy")))
        data_triplets = [(example_location / ("%d.npy" % idx),
                         action_location / ("%d.npy" % idx),
                         label_location / ("%d.npy" % idx))
                         for idx in range(num_examples)]

        feeder = ReplayQueue()
        for replay, action, result in data_triplets:
            feeder.insert(str(replay), str(action), str(result))
        storage["feeder"] = feeder

    return dataset


if __name__ == "__main__":
    rmtree("numpy", ignore_errors=True)
    makedirs("numpy", exist_ok=True)

    all_data_path = "replays/all_replays/*.sgf"
    data = glob.glob(all_data_path)

    # filter replays
    useful_dir = Path("replays") / "useful"
    faulty_dir = Path("replays") / "faulty"
    rmtree(str(useful_dir), ignore_errors=True)
    rmtree(str(faulty_dir), ignore_errors=True)
    makedirs(useful_dir, exist_ok=True)
    makedirs(faulty_dir, exist_ok=True)

    for replay in data:
        if replay_useful(replay):
            copy(replay, str(useful_dir))
        else:
            copy(replay, str(faulty_dir))
    data = glob.glob(str(useful_dir / "*.sgf"))
    random.shuffle(data)

    # split data
    data_split = {
        "training": data[350:],
        "test": data[100:450],
        "validation": data[:100],
    }

    # create numpy dataset
    generate_numpy_dataset(data_split, Path("numpy"))
    create_dataset(Path("numpy/training"),
                   Path("numpy/test"),
                   Path("numpy/validation"))
