import os
import sys
import pydantic_argparse
from params import TrainParams
from utils import save_pkl, load_pkl, xxd_c_dump, set_random_seed
from data import get_dataset
from model import load_existing_model, define_model

import tensorflow as tf
import numpy as np
from sklearn import metrics
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow_model_optimization as tfmot

import matplotlib.pyplot as plt
from keras import regularizers as reg
import collections

RANDOM_SEED = 42

if sys.platform == "darwin":
    Adam = tf.keras.optimizers.legacy.Adam
else:
    Adam = tf.keras.optimizers.Adam

def create_parser():
    """Create CLI argument parser
    Returns:
        ArgumentParser: Arg parser
    """
    return pydantic_argparse.ArgumentParser(
        model=TrainParams,
        prog="Human Activity Recognition Train Command",
        description="Train HAR model",
    )

def decay(epoch):
    if epoch < 5:
        return 1e-3
    if epoch < 15:
        return 1e-4
    if epoch < 30:
        return 1e-5
    return 1e-6

def decay_ft(epoch):
    return 1e-6


import datetime

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

def plot_training_results(params, model, history, test_data, test_labels, output_dir="plots"):
    """
    Plot training results and save them as image files with unique filenames.

    Args:
        model: Trained model.
        history: Keras training history object.
        test_data: Numpy array of test inputs.
        test_labels: One-hot encoded test labels.
        output_dir: Directory to save the plots.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Label configuration (expected classes)
    #EXPECTED_LABELS = ['standing_still', 'walking_forward', 'running_forward', 'climb_up', 'climb_down']
    display_mapping = {"standing_still":"STAND",
                       "walking_forward":"WALK",
                       "running_forward":"RUN",
                       "climb_up":"CLIMB UP",
                       "climb_down":"CLIMB DOWN"}
    #DISPLAY_LABELS = ['STAND', 'WALK', 'RUN', 'CLIMB UP', 'CLIMB DOWN']
    DISPLAY_LABELS = [display_mapping[x] for x in params.labels]

    # Predictions and true labels
    y_pred_test = model.predict(test_data, verbose=0)
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(test_labels, axis=1)

    # Confusion matrix (full version)
    full_matrix = metrics.confusion_matrix(max_y_test, max_y_pred_test)

    # Try plotting confusion matrix using the display labels
    try:
        matrix = metrics.confusion_matrix(
            max_y_test,
            max_y_pred_test,
            labels=range(len(DISPLAY_LABELS))
        )
        plt.figure(figsize=(6, 6))  # Adjusted size for clarity
        sns.heatmap(matrix, cmap='Blues', linecolor='white', linewidths=1,
                    xticklabels=DISPLAY_LABELS, yticklabels=DISPLAY_LABELS, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_{timestamp}.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        # In case of error, fallback to full matrix without specific labels
        labels = [f"Class {i}" for i in range(full_matrix.shape[0])]
        plt.figure(figsize=(6, 6))
        sns.heatmap(full_matrix, cmap='Blues', linecolor='white', linewidths=1,
                    xticklabels=labels, yticklabels=labels, annot=True, fmt='d')
        plt.title('Full Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"full_confusion_matrix_{timestamp}.png"))
        plt.close()

    # Accuracy plot
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(output_dir, f"accuracy_{timestamp}.png"))
    plt.close()

    # Loss plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(os.path.join(output_dir, f"loss_{timestamp}.png"))
    plt.close()

    # MAE plot
    if 'mean_absolute_error' in history.history:
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('Mean Absolute Error (MAE)')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(os.path.join(output_dir, f"mae_{timestamp}.png"))
        plt.close()



def train_model(params: TrainParams, train_data, train_labels, test_data, test_labels, fine_tune = False):
    # Initialize Hyperparameters
    verbose = 1

    if fine_tune:
        epochs = params.ft_epochs
    else:
        epochs = params.epochs

    batch_size = params.batch_size
    
    # window size
    n_timesteps = train_data.shape[1]
    n_features = train_data.shape[2]
    # number of labels
    n_outputs = train_labels.shape[1]

    print('[INFO] n_timesteps : ', n_timesteps)
    print('[INFO] n_features : ', n_features)
    print('[INFO] n_outputs : ', n_outputs)

    # Model Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="auto",
            restore_best_weights=True,
        )

    checkpoint_weight_path = os.path.join(str(params.job_dir),"model.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_weight_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        verbose=1,
    )

    tf_logger = tf.keras.callbacks.CSVLogger(os.path.join(str(params.job_dir),"history.csv"))

    if fine_tune:
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(decay_ft)
    else:
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(decay)

    model_callbacks = [early_stopping, checkpoint, tf_logger, lr_scheduler]

    # Model
    if fine_tune:
        model = load_existing_model(os.path.join(params.trained_model_dir, f"{params.model_name}.keras"))
    else:
        model = define_model(n_timesteps, n_features, n_outputs)

    model.summary()

    # Convert one-hot labels to integer class indices
    if len(train_labels.shape) > 1:
        train_label_indices = np.argmax(train_labels, axis=1)
    else:
        train_label_indices = train_labels

    if len(test_labels.shape) > 1:
        test_label_indices = np.argmax(test_labels, axis=1)
    else:
        test_label_indices = test_labels

    # Count labels
    train_label_counts = collections.Counter(train_label_indices)
    test_label_counts = collections.Counter(test_label_indices)

    # Output info
    print(f"Number of train samples: {len(train_label_indices)}")
    print(f"Train label distribution: {dict(train_label_counts)}")

    print(f"Number of test samples: {len(test_label_indices)}")
    print(f"Test label distribution: {dict(test_label_counts)}")

    # fit network
    history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels),
                        epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=model_callbacks,)

    # evaluate model
    (loss, accuracy, mae) = model.evaluate(test_data, test_labels, batch_size=batch_size, verbose=verbose)
    
    if not fine_tune:
        model.save(os.path.join(params.trained_model_dir, f"{params.model_name}.keras"))
    else:
        model.save(os.path.join(params.trained_model_dir, f"{params.ft_model_name}.keras"))

    print("[INFO] loss={:.4f}, accuracy: {:.4f}%, mean absolute error={:.4f}%".format(loss, accuracy * 100, mae))

    return model, history

if __name__ == "__main__":
    parser = create_parser()
    params = parser.parse_typed_args()
    #set_random_seed(params.seed)

    # Load dataset
    aug_train_data, aug_train_labels, aug_test_data, aug_test_labels = get_dataset(params, False)

    # Load Fine-tune data
    ft_aug_train_data, ft_aug_train_labels, ft_aug_test_data, ft_aug_test_labels = get_dataset(params, True)

    # default
    #model_name = params.base_model_name
    # Train model
    if params.train_model:
        model, history = train_model(params, aug_train_data, aug_train_labels, aug_test_data, aug_test_labels, fine_tune=False)
        if params.show_training_plot:
            plot_training_results(params, model, history, test_data=aug_test_data, test_labels=aug_test_labels)
    else:
        model = load_existing_model(os.path.join(params.trained_model_dir, f"{params.model_name}.keras"))

    # Fine-tune model
    if params.fine_tune_model:
        #model_name = params.ft_model_name
        model, history = train_model(params, ft_aug_train_data, ft_aug_train_labels, ft_aug_test_data, ft_aug_test_labels, fine_tune=True)
        if params.show_training_plot:
            plot_training_results(model, history, test_data=ft_aug_test_data, test_labels=ft_aug_test_labels)

"""
    # Quantize and convert
    tflite_filename = os.path.join(params.trained_model_dir,f"{params.model_name}.tflite")
    tflm_filename = os.path.join(params.trained_model_dir,f"{params.model_name}.cc")

    X_rep = train_test_split(aug_train_data, aug_train_labels, test_size=0.1, random_state=params.seed)[1]

    def representative_dataset():
        yield [X_rep.astype('float32')]    # TODO get better dataset, but aug_data is too large

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_type = tf.int8
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_quant_model = converter.convert()
    print("[INFO] Size of Quantized Model: " + str(len(tflite_quant_model)))

    # save binary
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_quant_model)

    # Generate the .cc file from tflite
    xxd_c_dump(
        src_path=tflite_filename,
        dst_path=tflm_filename,
        var_name='har_model',
        chunk_len=12,
        is_header=True,
    )

    if params.fine_tune_model:
        print("evaluating tflite model on fine-tune data")
        eval_tflite(tflite_filename, tflm_filename, test_data = ft_aug_test_data, test_labels = ft_aug_test_labels)
    else:
        print("evaluating tflite model on base data")
        eval_tflite(tflite_filename, tflm_filename, test_data = aug_test_data, test_labels = aug_test_labels)
"""
