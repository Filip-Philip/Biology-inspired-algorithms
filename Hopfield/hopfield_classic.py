import numpy as np
from PIL import Image
from sys import argv
import glob

STATUS_THINKING = "thinking"
STATUS_FOUND = "found"
STATUS_OSCILLATION = "oscilattion"


class Hopfield:
    def __init__(self, N):
        self.weights = np.zeros((N, N))

        self.status = STATUS_THINKING
        self.state = np.empty(N) * np.NaN
        self.last_state = self.state

    def add_pattern(self, pattern):
        self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)

    def set_state(self, state):
        self.status = STATUS_THINKING
        self.state = state

    def update(self):
        new_state = sign_0(np.dot(self.weights, self.state))

        if np.array_equal(new_state, self.state):
            self.status = STATUS_FOUND
        elif np.array_equal(new_state, self.last_state):
            self.status = STATUS_OSCILLATION

        self.last_state = self.state
        self.state = new_state

        return new_state

    def load_network(self):
        file = np.load("model.npz")
        self.weights = file["arr_0"]
        self.status = STATUS_THINKING
        self.state = np.empty(self.weights.shape[0]) * np.NaN
        self.last_state = self.state

    def save_network(self):
        np.savez("model", self.weights)


def sign_0(array):  # x=0 -> sign_0(x) = 1
    return np.where(array >= -1e-15, 1, -1)


IMG_SIZE = 153

if argv[1] == "train":
    hopfield = None

    images = glob.glob(argv[2])
    for image in images:
        print(f"Train on '{image}'...")
        with Image.open(image) as img:
            data = np.array(img.convert("L"))
            data = np.sign(data - 127.5)
            data = data.flatten()

            res = Image.fromarray(127.5 + data.reshape((IMG_SIZE, IMG_SIZE)) * 127.5)
            res.show()

            if hopfield == None:
                hopfield = Hopfield(len(data))

            hopfield.add_pattern(data)

    hopfield.save_network()
elif argv[1] == "test":
    hopfield = Hopfield(0)
    hopfield.load_network()

    with Image.open(argv[2]) as img:
        data = np.array(img.convert("L"))
        data = np.sign(data - 127.5)
        data = data.flatten()

        hopfield.set_state(data)
        while hopfield.status == STATUS_THINKING:
            res_data = hopfield.update()
            res = Image.fromarray(127.5 + res_data.reshape((IMG_SIZE, IMG_SIZE)) * 127.5)
            res.show()

        print(hopfield.status)
        res = Image.fromarray(127.5 + res_data.reshape((IMG_SIZE, IMG_SIZE)) * 127.5)
        res.show()
else:
    print("Wrong argument")
