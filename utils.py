import keras
import random
import numpy as np
import os
import pandas as pd
from pathlib import Path


def load_training_labels(root_dir='./', num_samples=None, balanced=False):
    training_labels = pd.read_csv(os.path.join(root_dir, 'train_labels.csv'))
    training_labels['label'] = training_labels['label'].astype('bool')
    if num_samples is None:
        return training_labels.sample(frac=1).reset_index(drop=True)
    
    if balanced:
        pos = training_labels[training_labels['label']].sample(num_samples // 2)
        neg = training_labels[~training_labels['label']].sample(num_samples // 2)
        training_labels = pd.concat([pos, neg]).sample(frac=1).reset_index(drop=True)
    else:
        training_labels = training_labels.sample(num_samples).reset_index(drop=True)

    return training_labels


def get_all_training_images(training_labels, root_dir='./'):
    return np.array([load_training_image(id, root_dir) for id in training_labels['id']])


def load_image_file(filename):
    img = keras.utils.load_img(filename, target_size=(96, 96, 3))
    return keras.utils.img_to_array(img) 


def load_training_image(id, root_dir='./'):
    return load_image_file(os.path.join(root_dir, 'train', f'{id}.tif'))


def load_test_image(id, root_dir='./'):
    return load_image_file(os.path.join(root_dir, 'test', f'{id}.tif'))


def batch_image_generator(training_labels, batch_size):
    inp = list(training_labels.index)
    while True:
        random.shuffle(inp)
        for i in range(0, len(inp), batch_size):
            inds = inp[i:i+batch_size]
            ids = training_labels.loc[inds]['id']
            batch_labels = training_labels.loc[inds]['label']
            batch_images = np.array([load_training_image(id) for id in ids])
            yield batch_images, batch_labels


def stochastic_batch_image_generator(training_labels, batch_size):
    while True:
        batch_labels = training_labels.sample(batch_size)
        ids = batch_labels['id']
        batch_images = np.array([load_training_image(id) for id in ids])
        yield batch_images, batch_labels['label']


def generate_submission(model, root_dir, output_file='submission.csv'):
    test_image_files = [f for f in os.listdir(os.path.join(root_dir, "test")) if f.endswith(".tif")]
    test_images = np.array([load_test_image(Path(f).stem, root_dir) for f in test_image_files])
    test_predictions = model.predict(test_images)
    test_predictions = (test_predictions > 0.5).astype("int32")
    # Create a submission DataFrame
    submission = pd.DataFrame({"id": [Path(f).stem for f in test_image_files], "label": test_predictions.flatten()})
    # Save the submission DataFrame to a CSV file
    submission.to_csv(output_file, index=False)
