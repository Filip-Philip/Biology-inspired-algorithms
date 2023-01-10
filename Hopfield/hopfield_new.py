# https://ml-jku.github.io/hopfield-layers/
from os import path
from random import Random
from sys import argv
import numpy as np
from PIL import Image
import glob
import decimal

STATUS_THINKING = "thinking"
STATUS_FOUND = "found"
STATUS_OSCILLATION = "oscilattion"


class Hopfield:
    def __init__(self, N):
        self.status = STATUS_THINKING
        self.state = np.empty(N) * np.NaN
        self.last_state = self.state
        self.patterns = np.empty((N, 0))
        self.pattern_names = np.empty(0, dtype=object)

    def add_pattern(self, pattern, pattern_name):
        self.patterns = np.column_stack((self.patterns, pattern))
        self.pattern_names = np.append(self.pattern_names, pattern_name)

    def set_state(self, state):
        self.status = STATUS_THINKING
        self.state = state

    def update(self, fast=True):
        n = len(self.state)

        new_state = np.empty(n)

        for l in range(n):
            # (l+)
            curr_plus = self.state.copy()
            curr_plus[l] = 1
            # (l-)
            curr_minus = self.state.copy()
            curr_minus[l] = -1

            if fast:
                component = np.sum(
                    np.exp(np.dot(self.patterns.T, curr_plus))
                    - np.exp(np.dot(self.patterns.T, curr_minus))
                )
            else:
                component = decimal.Decimal(0)
                for i in range(self.patterns.shape[1]):
                    component += np.exp(
                        decimal.Decimal(np.dot(self.patterns[:, i], curr_plus))
                    )
                    component -= np.exp(
                        decimal.Decimal(np.dot(self.patterns[:, i], curr_minus))
                    )

            new_state[l] = np.sign(component)

        if np.array_equal(new_state, self.state):
            self.status = STATUS_FOUND
        elif np.array_equal(new_state, self.last_state):
            self.status = STATUS_OSCILLATION

        self.last_state = self.state
        self.state = new_state

        return self.state

    def identify(self):
        for i in range(self.patterns.shape[1]):
            if np.array_equal(self.state, self.patterns[:, i]):
                return self.pattern_names[i]

    def load_network(self):
        file = np.load("model.npz", allow_pickle=True)
        self.patterns = file["patterns"]
        self.pattern_names = file["pattern_names"]
        self.status = STATUS_THINKING
        self.state = np.empty(self.patterns.shape[0]) * np.NaN
        self.last_state = self.state

    def save_network(self):
        np.savez("model", patterns=self.patterns, pattern_names=self.pattern_names)


IMG_SIZE = 64


def img_to_array(img):
    data = np.array(img.resize((IMG_SIZE, IMG_SIZE)).convert("L"))
    data = np.sign(data - 127.5)
    return data.flatten()


def array_to_img(arr):
    return Image.fromarray(127.5 + arr.reshape((IMG_SIZE, IMG_SIZE)) * 127.5)


if argv[1] == "train":
    hopfield = None

    images = glob.glob(argv[2])
    for image_path in images:
        print(f"Train on '{image_path}'...")
        with Image.open(image_path) as img:
            data = img_to_array(img)

            if hopfield == None:
                hopfield = Hopfield(len(data))

            hopfield.add_pattern(data, image_path)

    hopfield.save_network()
elif argv[1] == "test":
    hopfield = Hopfield(0)
    hopfield.load_network()

    correct = 0
    oscillations = 0
    total = 0

    images = glob.glob(argv[2])
    for image_path in images:
        with Image.open(image_path) as img:
            data = img_to_array(img)

            hopfield.set_state(data)
            while hopfield.status == STATUS_THINKING:
                res_data = hopfield.update(fast=False)

            if hopfield.status == STATUS_FOUND:
                correct_name = path.basename(image_path)
                identified_name = path.basename(hopfield.identify())
                print("Identified", correct_name, "as:", identified_name)
                if correct_name == identified_name:
                    correct += 1
            else:
                print(hopfield.status)
                oscillations += 1

            total += 1
    
    print("Total:", total)
    print("Correct:", correct)
    print("Oscillations:", oscillations)
else:
    print("Wrong mode")