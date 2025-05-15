import os
import sys
import pydantic_argparse
from params import TrainParams
from utils import save_pkl, load_pkl, xxd_c_dump
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline      # for warping
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from pathlib import Path  # Import Path module for path manipulation

if sys.platform == "darwin":
    Adam = tf.keras.optimizers.legacy.Adam
else:
    Adam = tf.keras.optimizers.Adam

# Normalize relative to the column mean
def normalize_window(df):
    for col in [['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']]:
        df[col] = (df[col] - df[col].mean()) / df[col].std()


## This example using cubic spline is not the best approach to generate random curves. 
## You can use other approaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1], 1)) * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    cg_x = CubicSpline(xx[:, 3], yy[:, 3])
    cg_y = CubicSpline(xx[:, 4], yy[:, 4])
    cg_z = CubicSpline(xx[:, 5], yy[:, 5])
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range), cg_x(x_range), cg_y(x_range), cg_z(x_range)]).transpose()

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma)  # Regard these samples around 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)       # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0] - 1) / tt_cum[-1, 0],
               (X.shape[0] - 1) / tt_cum[-1, 1],
               (X.shape[0] - 1) / tt_cum[-1, 2],
               (X.shape[0] - 1) / tt_cum[-1, 3],
               (X.shape[0] - 1) / tt_cum[-1, 4],
               (X.shape[0] - 1) / tt_cum[-1, 5]
               ]
    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
    tt_cum[:, 3] = tt_cum[:, 3] * t_scale[3]
    tt_cum[:, 4] = tt_cum[:, 4] * t_scale[4]
    tt_cum[:, 5] = tt_cum[:, 5] * t_scale[5]
    return tt_cum

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
    X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
    X_new[:, 2] = np.interp(x_range, tt_new[:, 2], X[:, 2])
    X_new[:, 3] = np.interp(x_range, tt_new[:, 3], X[:, 3])
    X_new[:, 4] = np.interp(x_range, tt_new[:, 4], X[:, 4])
    X_new[:, 5] = np.interp(x_range, tt_new[:, 5], X[:, 5])
    return X_new

def augment(X, labels, jitter_sigma=0.01, scaling_sigma=0.05):
    X_new = np.zeros(X.shape)

    for i, orig in enumerate(X):
        jitterNoise = np.random.normal(loc=0, scale=jitter_sigma, size=orig.shape)
        scaleNoise = np.random.normal(loc=1.0, scale=scaling_sigma, size=orig.shape)
        X_new[i] = orig * scaleNoise + jitterNoise
        X_new[i] = DA_TimeWarp(X_new[i])

    return X_new

def userBatches(file, windowSize, stride, windows, vectorizedActivities, lb, labels): #processing Raw Data
    print(file)

    # window array (empty) input, list of csv files
    # for loop Iterate through all CSV files of labeled data
        #  Reads CSV files
        # for each activity, do window extraction
            # create Window np array (subwindow)
            # append subwindow to window
    for label in labels:
        df = pd.read_csv(file)
        # print(f"Columns in file {file}: {df.columns.tolist()}")  # Print the column names to check
        df.columns = df.columns.str.strip()
        filter = df.where(df['label'].str.strip() == label).dropna().sort_values('timestamp').iloc[:, [7, 8, 9, 13, 14 ,15]] #acc_x, y, z, gyro_x, y, z
        for i in range(0, filter.shape[0] - windowSize, stride):
            newWindow = filter.iloc[i:i + windowSize, :].astype(float)
            normalize_window(newWindow)
            newWindow = newWindow.to_numpy().tolist()
            ctgrs = lb.transform([label]).tolist()[0]
            vectorizedActivities.append(ctgrs)
            windows.append(newWindow)


def get_dataset(params: TrainParams, fine_tune=False):
    aug_file = os.path.join(params.dataset_dir, params.augmented_dataset)

    if aug_file and os.path.isfile(aug_file):
        print("Loading augmented dataset from " + str(aug_file))
        ds = load_pkl(aug_file)
        return ds["X"], ds["y"], ds["XT"], ds["yt"]

    processed_file = os.path.join(params.dataset_dir, params.processed_dataset)

    if not os.path.isfile(processed_file):
        print("Creating processed dataset. This may take a few minutes...")


        dataDir = [
            entry for entry in os.scandir(os.path.join(params.dataset_dir, "raw_data"))
            if entry.name != ".DS_Store"
        ]

        # when the button data is extracted, the labels are mapped
        labels = ['standing_still', 'walking_forward', 'running_forward', 'climb_up', 'climb_down']
        lb = LabelBinarizer()
        lb.fit(labels)
        
        X = []
        Y = []

        for file in dataDir:
            userBatches(file, params.num_time_steps, params.sample_step, X, Y, lb, labels)

        # Convert and reshape
        X = np.asarray(X, dtype=np.float32).reshape(-1, params.num_time_steps, params.num_features)
        Y = np.asarray(Y, dtype=np.float32)

        # Perform a 70/30 split
        aug_X, testX, aug_y, testy = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

        # Optional reshaping and type conversion
        aug_X = np.asarray(aug_X, dtype=np.float32).reshape(-1, params.num_time_steps, params.num_features)
        aug_y = np.asarray(aug_y, dtype=np.float32)
        testX = np.asarray(testX, dtype=np.float32).reshape(-1, params.num_time_steps, params.num_features)
        testy = np.asarray(testy, dtype=np.float32)

        save_pkl(processed_file, X=aug_X, y=aug_y, XT=testX, yt=testy)

        if params.augmentations == 0:
            return aug_X, aug_y, testX, testy
    else:
        print("Loading processed dataset from " + str(processed_file))
        ds = load_pkl(processed_file)
        aug_X = ds["X"]
        aug_y = ds["y"]
        testX = ds["XT"]
        testy = ds["yt"]

        if params.augmentations == 0:
            return aug_X, aug_y, testX, testy

    print("Augmenting baseline by %dx" % params.augmentations)
    orig_X = aug_X
    orig_testX = testX
    orig_y = aug_y
    orig_testy = testy

    for i in range(params.augmentations):
        aug_X = np.concatenate([aug_X, augment(orig_X, orig_y)])
        aug_y = np.concatenate([aug_y, orig_y])
        print("Augmentation pass %d complete" % i)

    if params.save_processed_dataset:
        print("Saving augmented dataset to " + str(aug_file))
        save_pkl(aug_file, X=aug_X, y=aug_y, XT=orig_testX, yt=orig_testy)

    return aug_X, aug_y, orig_testX, orig_testy