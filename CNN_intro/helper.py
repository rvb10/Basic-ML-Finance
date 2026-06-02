import math
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import pickle
import tensorflow as tf
import importlib.util
import unittest
import json
import pdb
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# Custom assert_equal function to replace nose.tools.assert_equal
def assert_equal(a, b, msg=None):
    unittest.TestCase().assertEqual(a, b, msg)

class Helper():
    def __init__(self, data_dir="./Data", model_dir="models", useKerasExt=True):
        # Data directory
        self.DATA_DIR = data_dir
        self.model_dir = model_dir
        self.useKerasExt = useKerasExt

        if not os.path.isdir(self.DATA_DIR):
            self.DATA_DIR = "../resource/asnlib/publicdata/ships_in_satellite_images/data"

        self.dataset = "shipsnet.json"

    def getData(self):
        data, labels = self.json_to_numpy(os.path.join(self.DATA_DIR, self.dataset))
        return data, labels

    def showData(self, data, labels, num_cols=5, cmap=None):
        num_rows = math.ceil(data.shape[0] / num_cols)
        fig = plt.figure(figsize=(10, 10))

        for i in range(data.shape[0]):
            img, img_label = data[i], labels[i]
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            ax.set_axis_off()
            ax.set_title(img_label)
            plt.imshow(img, cmap=cmap)

        fig.tight_layout()
        return fig

    def modelPath(self, modelName):
        return os.path.join(".", self.model_dir, modelName)

    def y_OHE(self, y):
        """Determine if y is One-Hot Encoded."""
        return (y.ndim > 1) and (y.shape[-1] > 1)

    def saveModel(self, model, modelName):
        model_path = self.modelPath(modelName)
        os.makedirs(self.model_dir, exist_ok=True)

        if self.useKerasExt:
            model.save(model_path + '.keras')
            print(f"Model saved in {model_path}.keras; submit with your assignment.")
        else:
            json_config = model.to_json()
            with open(os.path.join(model_path, 'config.json'), 'w') as json_file:
                json_file.write(json_config)
            model.save_weights(os.path.join(model_path, 'weights.h5'))
            print(f"Model saved in directory {model_path}; create an archive and submit.")

    def loadModel(self, modelName, loss="binary_crossentropy", metrics=['accuracy']):
        model_path = self.modelPath(modelName)
        if self.useKerasExt:
            return tf.keras.models.load_model(model_path + '.keras')
        else:
            with open(os.path.join(model_path, 'config.json')) as json_file:
                json_config = json_file.read()
            model = tf.keras.models.model_from_json(json_config)
            model.compile(loss=loss, metrics=metrics)
            model.load_weights(os.path.join(model_path, 'weights.h5'))
            return model

    def saveHistory(self, history, model_name):
        history_path = self.modelPath(model_name)
        os.makedirs(history_path, exist_ok=True)
        with open(os.path.join(history_path, 'history'), 'wb') as f:
            pickle.dump(history.history, f)

    def loadHistory(self, model_name):
        history_path = self.modelPath(model_name)
        with open(os.path.join(history_path, 'history'), 'rb') as f:
            return pickle.load(f)

    def json_to_numpy(self, json_file):
        with open(json_file) as f:
            dataset = json.load(f)

        data = np.array(dataset['data']).astype('uint8')
        labels = np.array(dataset['labels']).astype('uint8')

        data = data.reshape([-1, 3, 80, 80]).transpose([0, 2, 3, 1])
        return data, labels

    modelName = "Ships_in_satellite_images"
    es_callback = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=2, restore_best_weights=True)
    ckpt_ext = ".keras"

    callbacks = [
        es_callback,
        ModelCheckpoint(filepath=modelName + ckpt_ext, monitor='accuracy', save_best_only=True)
    ]
    max_epochs = 30

    def train(self, model, X, y, model_name, epochs=max_epochs):
        model.summary()
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
        
        print("Train set size:", X_train.shape[0], ", Validation set size:", X_valid.shape[0])
        history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), callbacks=self.callbacks)
        fig, axs = self.plotTrain(history, model_name)
        return history, fig, axs

    def acc_key(self, history=None, model=None):
        """Returns the key for accuracy metric based on TensorFlow version."""
        if model:
            return "accuracy" if "accuracy" in model.metrics_names else "acc"
        return "accuracy" if "accuracy" in history.history.keys() else "acc"

    def plotTrain(self, history, model_name="???"):
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        acc_string = self.acc_key(history=history)

        axs[0].plot(history.history['loss'])
        axs[0].plot(history.history['val_loss'])
        axs[0].set_title(model_name + " model loss")
        axs[0].set_ylabel('loss')
        axs[0].set_xlabel('epoch')
        axs[0].legend(['train', 'validation'], loc='upper left')

        axs[1].plot(history.history[acc_string])
        axs[1].plot(history.history['val_' + acc_string])
        axs[1].set_title(model_name + " model accuracy")
        axs[1].set_ylabel('accuracy')
        axs[1].set_xlabel('epoch')
        axs[1].legend(['train', 'validation'], loc='upper left')

        return fig, axs

    def model_interpretation(self, clf):
        dim = round(clf.coef_[0].shape[-1] ** 0.5)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(1, 1, 1)
        scale = np.abs(clf.coef_[0]).max()
        ax.imshow(clf.coef_[0].reshape(dim, dim), interpolation='nearest', cmap="gray", vmin=-scale, vmax=scale)
        ax.set_xticks(())
        ax.set_yticks(())
        fig.suptitle('Parameters')

