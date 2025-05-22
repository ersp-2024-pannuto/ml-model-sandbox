import os
import sys
import pydantic_argparse
from params import TrainParams
from utils import save_pkl, load_pkl, xxd_c_dump
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline      # for warping
from sklearn.model_selection import train_test_split,StratifiedKFold
import pickle


from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from pathlib import Path  # Import Path module for path manipulation

if sys.platform == "darwin":
    Adam = tf.keras.optimizers.legacy.Adam
else:
    Adam = tf.keras.optimizers.Adam

columns_to_keep = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']

# Normalize relative to the column mean
def normalize_window(df):
    cols = columns_to_keep
    df[cols] = (df[cols] - df[cols].mean()) / df[cols].std()
    return df

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
    X_new = np.zeros_like(X)
    x_range = np.arange(X.shape[0])

    for d in range(X.shape[1]):
        X_new[:, d] = np.interp(x_range, tt_new[:, d], X[:, d])

    return X_new

def augment(X, labels, jitter_sigma=0.01, scaling_sigma=0.05, label_binarizer=None):
    if label_binarizer is not None and 'standing_still' in label_binarizer.classes_:
        standing_idx = np.where(label_binarizer.classes_ == 'standing_still')[0][0]
    else:
        standing_idx = None  # Fallback if 'standing_still' not found

    X_new = np.zeros(X.shape)

    for i, orig in enumerate(X):
        is_standing = (standing_idx is not None and labels[i][standing_idx] == 1)

        if not is_standing:
            jitterNoise = np.random.normal(loc=0, scale=jitter_sigma, size=orig.shape)
            scaleNoise = np.random.normal(loc=1.0, scale=scaling_sigma, size=orig.shape)
            X_new[i] = orig * scaleNoise + jitterNoise
            X_new[i] = DA_TimeWarp(X_new[i])
        else:
            X_new[i] = DA_TimeWarp(orig)
    return X_new

def userBatches(file, windowSize, stride, windows, vectorizedActivities, lb, labels, down_sample_hz = -1): #processing Raw Data
    print(file, labels)

    # window array (empty) input, list of csv files
    # for loop Iterate through all CSV files of labeled data
        #  Reads CSV files
        # for each activity, do window extraction
            # create Window np array (subwindow)
            # append subwindow to window
    for label in labels:
        df = pd.read_csv(file)
        if down_sample_hz!=-1:
            down_sample_spacing = 100//down_sample_hz
            df = df.iloc[::down_sample_spacing].reset_index(drop=True)
        # print(f"Columns in file {file}: {df.columns.tolist()}")  # Print the column names to check
        df.columns = df.columns.str.strip()
        filtered = (
            df.where(df['label'].str.strip() == label)
              .dropna()
              .sort_values('timestamp')
              [columns_to_keep]
        )
        #print(f"File: {file}, Label: {label}, Rows selected: {len(filtered)}", lb.transform([label]).tolist())
        for i in range(0, filtered.shape[0] - windowSize, stride):
            newWindow = filtered.iloc[i:i + windowSize, :].astype(float)
            #newWindow = normalize_window(newWindow)
            newWindow = newWindow.to_numpy().tolist()
            ctgrs = lb.transform([label]).tolist()[0]
            vectorizedActivities.append(ctgrs)
            windows.append(newWindow)

def get_kfold_dataset(params: TrainParams):
    processed_file = os.path.join(params.dataset_dir, params.kfold_processed_dataset)
    augmented_file = os.path.join(params.dataset_dir, params.kfold_augmented_dataset)

    lb = LabelBinarizer()
    lb.fit(params.labels)

    if not os.path.isfile(processed_file):
        print("Creating processed dataset. This may take a few minutes...")
        data_files = [
            entry for entry in os.scandir(os.path.join(params.dataset_dir, "raw_data"))
            if entry.name != ".DS_Store"
        ]
        X = []
        Y = []
        for f in data_files:
            userBatches(f, params.num_time_steps, params.sample_step, X, Y, lb, params.labels, down_sample_hz = params.down_sample_hz)
        # Convert and reshape
        X = np.asarray(X, dtype=np.float32).reshape(-1, params.num_time_steps, params.num_features)
        Y = np.asarray(Y, dtype=np.float32)
        #save everything without split in the first two fields
        save_pkl(processed_file, X=X, y=Y, XT=[], yt=[])
    else:
        ds = load_pkl(processed_file)
        X = ds["X"]
        Y= ds["y"]
    
    if not os.path.isfile(augmented_file):
        orig_X, orig_Y = X, Y

        # Step 1: Augment original dataset
        X_aug_all = X.copy()
        Y_aug_all = Y.copy()

        for i in range(params.augmentations):
            X_aug_all = np.concatenate([X_aug_all, augment(orig_X, orig_Y, label_binarizer = lb)])
            Y_aug_all = np.concatenate([Y_aug_all, orig_Y])
            print("Augmentation pass %d complete" % i)

        save_pkl(augmented_file, X=X_aug_all, y=Y_aug_all, XT=[], yt=[])
    else:
        ds = load_pkl(augmented_file)
        X_aug_all, Y_aug_all = ds["X"], ds["y"]

    fold_file = os.path.join(params.dataset_dir, f"augmented_once_{params.n_folds}_folds.pkl")

    if not os.path.isfile(fold_file):
        if len(Y.shape) > 1:
            Y_indices = np.argmax(Y, axis=1)
        else:
            Y_indices = Y

        # Step 2: Stratified split only on original data
        skf = StratifiedKFold(n_splits=params.n_folds, shuffle=True)
        fold_data = []

        for fold, (train_orig_idx, val_orig_idx) in enumerate(skf.split(X, Y_indices)):
            print(f"\nFold {fold + 1}")

            N = len(X)
            train_full_idx = train_orig_idx.copy()
            for i in range(1, params.augmentations + 1):
                train_full_idx = np.concatenate([train_full_idx, train_orig_idx + i * N])

            X_train, Y_train = X_aug_all[train_full_idx], Y_aug_all[train_full_idx]
            X_val, Y_val = X[val_orig_idx], Y[val_orig_idx]  # only unaugmented

            fold_data.append((X_train, Y_train, X_val, Y_val))

        # Save all folds
        with open(fold_file, "wb") as f:
            pickle.dump(fold_data, f)
    else:
        with open(fold_file, "rb") as f:
            fold_data = pickle.load(f)

    return fold_data


def get_dataset(params: TrainParams, fine_tune=False):
    aug_file = os.path.join(params.dataset_dir, params.augmented_dataset)
    # when the button data is extracted, the labels are mapped
    labels = params.labels
    lb = LabelBinarizer()
    lb.fit(labels)
    # the labels will be sorted
    print(lb.classes_)

    if aug_file and os.path.isfile(aug_file):
        print("Loading augmented dataset from " + str(aug_file))
        ds = load_pkl(aug_file)
        return ds["X"], ds["y"], ds["XT"], ds["yt"]

    processed_file = os.path.join(params.dataset_dir, params.processed_dataset)

    if not os.path.isfile(processed_file):
        print("Creating processed dataset. This may take a few minutes...")


        data_files = [
            entry for entry in os.scandir(os.path.join(params.dataset_dir, "raw_data"))
            if entry.name != ".DS_Store"
        ]

        if params.split_method == 1:
            #random split
            print("random split")
            X = []
            Y = []
            for f in data_files:
                userBatches(f, params.num_time_steps, params.sample_step, X, Y, lb, labels, down_sample_hz = params.down_sample_hz)

            # Convert and reshape
            X = np.asarray(X, dtype=np.float32).reshape(-1, params.num_time_steps, params.num_features)
            Y = np.asarray(Y, dtype=np.float32)

            if len(Y.shape) > 1:
                Y_indices = np.argmax(Y, axis=1)
            else:
                Y_indices = Y

            # Perform a 70/30 split
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y_indices)
        elif params.split_method == 2:
            # user based split
            user_to_files = {}

            for file_entry in data_files:
                df = pd.read_csv(file_entry.path)
                person_id = df['person'].iloc[0]  # assuming one person per file
                user_to_files.setdefault(person_id, []).append(file_entry)

            # Step 3: split users into train/test
            all_users = list(user_to_files.keys())
            train_users, test_users = train_test_split(all_users, test_size=0.3)
            print("train_users",train_users, "test_users",test_users)
            #train_users = [1,3,5,7] 
            #test_users = [2,4,6]

            # Step 5: load batches from train/test files
            X_train, Y_train, X_test, Y_test = [], [], [], []

            for user in train_users:
                for file_entry in user_to_files[user]:
                    userBatches(file_entry, params.num_time_steps, params.sample_step, X_train, Y_train, lb, labels, down_sample_hz = params.down_sample_hz)

            for user in test_users:
                for file_entry in user_to_files[user]:
                    userBatches(file_entry, params.num_time_steps, params.sample_step, X_test, Y_test, lb, labels, down_sample_hz = params.down_sample_hz)

        # Optional reshaping and type conversion
        X_train = np.asarray(X_train, dtype=np.float32).reshape(-1, params.num_time_steps, params.num_features)
        Y_train = np.asarray(Y_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32).reshape(-1, params.num_time_steps, params.num_features)
        Y_test = np.asarray(Y_test, dtype=np.float32)
        
        
        # Convert one-hot labels to integer class indices
        if len(Y_train.shape) > 1:
            train_label_indices = np.argmax(Y_train, axis=1)
        else:
            train_label_indices = Y_train

        if len(Y_test.shape) > 1:
            test_label_indices = np.argmax(Y_test, axis=1)
        else:
            test_label_indices = Y_test

        # Count labels
        import collections
        train_label_counts = collections.Counter(train_label_indices)
        test_label_counts = collections.Counter(test_label_indices)

        # Output info
        print(f"Number of train samples: {len(train_label_indices)}")
        print(f"Train label distribution: {dict(train_label_counts)}")

        print(f"Number of test samples: {len(test_label_indices)}")
        print(f"Test label distribution: {dict(test_label_counts)}")

        save_pkl(processed_file, X=X_train, y=Y_train, XT=X_test, yt=Y_test)

        if params.augmentations == 0:
            return X_train, Y_train, X_test, Y_test
    else:
        print("Loading processed dataset from " + str(processed_file))
        ds = load_pkl(processed_file)
        X_train = ds["X"]
        Y_train = ds["y"]
        X_test = ds["XT"]
        Y_test = ds["yt"]

        if params.augmentations == 0:
            return  X_train, Y_train, X_test, Y_test

    print("Augmenting baseline by %dx" % params.augmentations)
    orig_X = X_train
    orig_y = Y_train
    #print(X_train)

    for i in range(params.augmentations):
        X_train = np.concatenate([X_train, augment(orig_X, orig_y, label_binarizer = lb)])
        Y_train = np.concatenate([Y_train, orig_y])
        print("Augmentation pass %d complete" % i)

    if params.save_processed_dataset:
        print("Saving augmented dataset to " + str(aug_file))
        save_pkl(aug_file, X=X_train, y=Y_train, XT=X_test, yt=Y_test)

    return X_train, Y_train, X_test, Y_test