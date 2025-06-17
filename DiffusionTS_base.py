import os
import torch
import numpy as np
from pathlib import Path

from engine.solver import Trainer
from Utils.io_utils import load_yaml_config, instantiate_from_config


from minerva.data.data_modules.har import MultiModalHARSeriesDataModule
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse
from DiffusionTS_utils import get_datamodule, normalize_data, denormalize_data, create_dl_info, create_trainer, visualize, get_labels_dict, data_to_2d, obtain_embedding_from_model


def __main__(dataset_name = "UCI", preffix = "DAGHAR_UCI", num_checkpoints = 10, max_epochs = 12000, class_id = None, milestone = 10, scaler = "standard"):
    if class_id is not None:
        folder = f"{preffix}-trained_on-class_{class_id}-{scaler}_scaled-{max_epochs}_epochs"
    else:
        folder = f"{preffix}-trained_on-all_classes-{scaler}_scaled-{max_epochs}_epochs"
    os.makedirs(f"./Figs/{preffix}", exist_ok=True)
    os.makedirs(f"./Figs/{preffix}/{milestone}", exist_ok=True)
    data_module = get_datamodule(dataset_name)

    # Load the training data
    train_x, train_y = data_module.train_dataloader().dataset[:]
    
    # Normalize the training data
    train_x_accel = train_x[:, :3, :]
    train_x_gyro = train_x[:, 3:, :]

    train_x_accel_scaled, scaler_accel = normalize_data(train_x_accel, scaler)
    train_x_gyro_scaled, scaler_gyro = normalize_data(train_x_gyro, scaler)

    # Concatenate the scaled data
    train_x_scaled = np.concatenate((train_x_accel_scaled, train_x_gyro_scaled), axis=1)
    train_x_scaled = np.transpose(train_x_scaled, (0, 2, 1))

    # Filtering the data per class
    if class_id is not None:
        # Filter the data for the specific class
        data_x = train_x_scaled[train_y == class_id]
    else:
        # Use all data
        data_x = train_x_scaled

    # Create a DataLoader for the filtered data
    data_dl_info = create_dl_info(data_x)
    # Create the trainer
    trainer = create_trainer(folder=folder, dl_info=data_dl_info, max_epochs=max_epochs, num_checkpoints=num_checkpoints)

    try:
        trainer.load(milestone=milestone, verbose=True)
    except Exception as e:
        trainer.train()
    
    dataset = data_dl_info["dataset"]
    seq_length, feature_dim = 60, 6
    

    fake_data = trainer.sample(num=len(dataset), size_every=2001, shape=[seq_length, feature_dim])

    # Unnormalize the fake data
    fake_data_accel = denormalize_data(fake_data[:, :, :3], scaler_accel)
    fake_data_gyro = denormalize_data(fake_data[:, :, 3:], scaler_gyro)
    # Concatenate the two modalities
    fake_data_unnormalized = np.concatenate((fake_data_accel, fake_data_gyro), axis=2)
    # Transpose the data to match the original shape
    fake_data_unnormalized = np.transpose(fake_data_unnormalized, (0, 2, 1))
    # Save the fake data
    synthetic_data_path = Path("./Executed/SyntheticData/") / preffix / str(milestone) / folder
    os.makedirs(synthetic_data_path, exist_ok=True)
    np.save(synthetic_data_path / f"fake_data.npy", fake_data_unnormalized)
    # CUDA_VISIBLE_DEVICES=7 python DiffusionTS_base.py --max_epochs 480000 --preffix D_UCI --milestone 10 --scaler minmax

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", help="Max epochs", type=int, default=12000)
    parser.add_argument("--num_checkpoints", help="Number of checkpoints", type=int, default=10)
    parser.add_argument("--preffix", help="Some unique id. For example: daghar_uci", type=str, required=True)
    parser.add_argument("--dataset_name", help="Dataset name", type=str, default="UCI")
    parser.add_argument("--class_id", help="Class ID", type=int)
    parser.add_argument("--milestone", help="Milestone", type=int, default=10)
    parser.add_argument("--scaler", help="Scaler", type=str, default="standard", choices=["standard", "minmax"])
    args = parser.parse_args()
    print(args)
    __main__(
        dataset_name=args.dataset_name,
        num_checkpoints=args.num_checkpoints,
        preffix=args.preffix,
        max_epochs=args.max_epochs,
        class_id=args.class_id,
        milestone=args.milestone,
        scaler=args.scaler)