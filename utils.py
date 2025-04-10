import keras
import random
import numpy as np
import os
import pandas as pd
from pathlib import Path
import tensorflow as tf


def load_training_labels(data_dir='./', num_samples=None, balanced=False):
    training_labels = pd.read_csv(os.path.join(data_dir, 'train_labels.csv'))
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


def get_training_images(training_labels, data_dir='./', color_mode='rgb', crop=1.0):
    images = np.array(
        [tf.image.central_crop(
            keras.utils.img_to_array(keras.utils.load_img(os.path.join(data_dir, 'train', f'{id}.tif'), color_mode=color_mode)),
            crop).numpy()
         for id in training_labels['id']])
    return images


def batch_image_generator(training_labels, batch_size):
    inp = list(training_labels.index)
    while True:
        random.shuffle(inp)
        for i in range(0, len(inp), batch_size):
            inds = inp[i:i+batch_size]
            batch_labels = training_labels.loc[inds].reset_index(drop=True)
            batch_images = get_training_images(batch_labels)
            yield batch_images, batch_labels['label']


def stochastic_batch_image_generator(training_labels, batch_size):
    while True:
        batch_labels = training_labels.sample(batch_size).reset_index(drop=True)
        ids = batch_labels['id']
        batch_images = get_training_images(batch_labels)
        yield batch_images, batch_labels['label']


def get_test_images(data_dir='./', color_mode='rgb', crop=1.0):
    test_image_files = [f for f in os.listdir(os.path.join(data_dir, "test")) if f.endswith(".tif")]
    test_ids = [Path(f).stem for f in test_image_files]
    test_images = np.array(
        [tf.image.central_crop(
            keras.utils.img_to_array(keras.utils.load_img(os.path.join(data_dir, 'test', f), color_mode=color_mode)),
            crop).numpy()
         for f in test_image_files])
    return test_images, test_ids


def generate_submission(model, test_images, test_ids, color_mode='rgb', output_file='submission.csv'):
    test_predictions = model.predict(test_images)
    submission_probs = pd.DataFrame({"id": test_ids, "label": test_predictions.flatten()})
    #binary_predictions = (test_predictions > 0.5).astype("int32")
    #submission_binary = pd.DataFrame({"id": test_ids, "label": binary_predictions.flatten()})
    submission_probs.to_csv(output_file, index=False)
    #submission_binary.to_csv(output_file.replace('.csv', '_binary.csv'), index=False)