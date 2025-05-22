import os
import tempfile
from enum import Enum, IntEnum
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field

class TrainParams(BaseModel):
    seed: int = Field(42, description="Random Seed")
    num_time_steps: int = Field(200, description="Number of Timestep Windows")
    num_features: int = Field(6, description="Number of Features")
    sample_step: int = Field(20, description="Timestep Window Slide")
    trained_model_dir: Path = Field("trained_models/", description="Directory where trained models are stored")
    job_dir: Path = Field("artifacts/", description="Directory where artifacts are stored")
    dataset_dir: Path =Field("datasets_audio/", description="Directory where datasets reside")
    processed_dataset: str = Field("processed_dataset.pkl", description="Name of processed baseline dataset")
    augmented_dataset: str = Field("augmented_dataset.pkl", description="Name of processed augmented dataset")
    kfold_processed_dataset: str = Field("kfold_processed_dataset.pkl", description="Name of processed baseline dataset")
    kfold_augmented_dataset: str = Field("kfold_augmented_dataset.pkl", description="Name of processed augmented dataset")
    processed_ft_dataset: str = Field("processed_ft_dataset.pkl", description="Name of processed baseline dataset")
    augmented_ft_dataset: str = Field("augmented_ft_dataset.pkl", description="Name of processed augmented dataset")
    batch_size: int = Field(32, description="Batch Size")
    augmentations: int = Field(5, description="Number of augmentation passes")
    save_processed_dataset: bool = Field(True, description="Save processed datasets as pkls")
    epochs: int =Field(10, description="Number of training epochs")
    ft_epochs: int =Field(5, description="Number of fine-tuning epochs")
    model_name: str = Field("model", description="Name of trained model")
    training_dataset_percent: int = Field(65, description="Percent of records used for training")
    show_training_plot: bool = Field(True, description="Show training statistics plots")
    train_model: bool = Field(True, description="Train the model, otherwise load existing model")
    fine_tune_model: bool = Field(False, description="Fine-tune the model, otherwise load existing model")
    split_method: int = Field(2, description="How to split: 1, random; 2, user-based")
    n_folds: int = Field(5, description="Number of Folds")
    down_sample_hz: int = Field(-1, discription="target down sampling rate")
    labels: List[str] = Field(
            default_factory=lambda: ['standing_still', 'walking_forward', 'running_forward', 'climb_up', 'climb_down'],
            description="List of labels")
