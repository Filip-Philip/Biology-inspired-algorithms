from math import sqrt
from os import path
import os
from random import Random
from sys import argv
from PIL import Image
import numpy as np
import glob

os.makedirs(argv[3], exist_ok=True)
noise = float(argv[1])

images = glob.glob(argv[2])
for image_path in images:
    with Image.open(image_path) as img:
        data = np.array(img)

        for _ in range(int(noise * data.size)):
            x = Random().randint(0, data.shape[0] - 1)
            y = Random().randint(0, data.shape[1] - 1)
            data[x, y] = np.abs(data[x, y] * 2 + 100) % 256

        img2 = Image.fromarray(data)

        img2.save(argv[3] + "\\" + path.basename(image_path))
