#!/usr/bin/env python3

from PIL import Image
from keras.models import load_model

import numpy as np


SIZE = 512


def generate_patches(image, patch):
    for y in range(0, SIZE, patch):
        for x in range(0, SIZE, patch):
            yield image.crop((x, y, x + patch, y + patch))


def get_predicts(patches, image, model, delta, patch):
    ls = [np.asarray(p.crop((patch - delta, 0, patch, patch))) for p in patches]
    rs = [np.asarray(p.crop((0, 0, delta, patch))) for p in patches]

    print('Before square...')
    hors = [np.hstack([l, r]) for l in ls for r in rs]
    print('After square...')
    hors = np.array(hors) / 255
    hors = np.expand_dims(hors, axis=-1)

    print('Hors: {}'.format(hors.shape))

    scores = model.predict(hors, batch_size=256)
    print('Scores: {}'.format(scores.shape))
    return scores.reshape((len(patches), len(patches)))


def get_horizontal_predicts(image, model, delta, patch):
    patches = list(generate_patches(image=image, patch=patch))
    return get_predicts(patches=patches,
                        image=image,
                        model=model,
                        delta=delta,
                        patch=patch)


def get_vertical_predicts(image, model, delta, patch):
    patches = [patch.transpose(Image.ROTATE_90) for patch in generate_patches(image=image, patch=patch)]
    return get_predicts(patches=patches,
                        image=image,
                        model=model,
                        delta=delta,
                        patch=patch)
