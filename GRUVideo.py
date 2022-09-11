from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import imageio
import cv2
import os

metadata = pd.read_csv("C:\\Users\\joshu\\Client Work\\echocardiogram\\FileList.csv")
metadata
IMG_SIZE = (112,112,3)
BATCH_SIZE = 64
EPOCHS = 10
NUM_FEATURES = 2048

MAX_SEQ_LENGTH = 250
metadata = metadata[metadata["NumberOfFrames"] <= MAX_SEQ_LENGTH]
metadata.shape

def load_video(path, max_frames=0):
    capture = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame = frame[:,:,[2,1,0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        capture.release()
    return np.array(frames)

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE)
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input(IMG_SIZE)
    preprocessed = preprocess_input(inputs)
    outputs = feature_extractor(preprocessed)

    return keras.Model(inputs, outputs, name="feature_extractor")

class PrepareVideosGen(tf.keras.utils.Sequence):
    def __init__(self, df, X_col, y_col, batch_size, input_size=(112, 112, 3), shuffle=False):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle

        self.n = len(self.df)

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["FileName"].values.tolist()
    ef = df["EF"].values

    frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )

    for idx, path in enumerate(video_paths):
        frames = load_video(os.path.join(root_dir, path) + ".avi")
        frames = frames[None, ...]

        temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH, ), dtype="bool")
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )

        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i,j,:] = feature_extractor.predict(
                    batch[None,j,:]
                )
            temp_frame_mask[i,:length] = 1

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()
    return (frame_features, frame_masks), ef

feature_extractor = build_feature_extractor()
train_data, train_values = prepare_all_videos(metadata.iloc[0:1000], "C:\\Users\\joshu\\Client Work\\echocardiogram\\Videos")
test_data, test_values = prepare_all_videos(metadata.iloc[9000:9445], "C:\\Users\\joshu\\Client Work\\echocardiogram\\Videos")
print("Done!")

def get_sequence_model():
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(1, activation="linear")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)
    rnn_model.compile(
        loss=tf.keras.losses.MeanSquaredError(), optimizer="adam", metrics=[tf.keras.losses.MeanSquaredError()]
    )
    return rnn_model

def run_experiment():
    filepath = "echocardiogram/tmp"
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_values,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint]
    )

    seq_model.load_weights(filepath)
    _, mse = seq_model.evaluate([test_data[0], test_data[1]], test_values)
    print(f"Test MSE: {round(mse, 2)}")

    return history, seq_model

_, sequnce_model = run_experiment()
