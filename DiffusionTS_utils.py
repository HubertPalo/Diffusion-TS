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


def get_datamodule(dataset="UCI", batch_size=64, data_percentage=1.0):
    data_module = MultiModalHARSeriesDataModule(
        data_path=f"/workspaces/HIAAC-KR-Dev-Container/shared_data/daghar/standardized_view/{dataset}/",
        feature_prefixes=["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
        label="standard activity code",
        features_as_channels=True,
        cast_to="float32",
        batch_size=batch_size,
        data_percentage=data_percentage,
    )
    data_module.setup(stage="fit")
    return data_module

def normalize_data(data, scaler="standard"):
    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler((-1,1))
    }
    scaler = scalers[scaler]
    data_flat = data.reshape(-1, 1)
    data_flat_scaled = scaler.fit_transform(data_flat)
    data_scaled = data_flat_scaled.reshape(data.shape)
    return data_scaled, scaler

def denormalize_data(data, scaler):
    data_flat = data.reshape(-1, 1)
    data_flat_denormed = scaler.inverse_transform(data_flat)
    data_denormed = data_flat_denormed.reshape(data.shape)
    return data_denormed

def create_dl_info(data):
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=True
    )
    dl_info = {
        'dataloader': data_loader,
        'test_dataloader': data_loader,
        'dataset': data
    }
    return dl_info

def create_trainer(config_path='./exec_configs/daghar.yaml', folder='daghar_uci_class_0', max_epochs=12000, num_checkpoints=10, dl_info=None):
    class Args_Example:
        def __init__(self) -> None:
            self.config_path = config_path
            self.save_dir = f"./Executed/{folder}"
            self.gpu = 0
            os.makedirs(self.save_dir, exist_ok=True)

    args =  Args_Example()
    configs = load_yaml_config(args.config_path)
    configs["solver"]["results_folder"] = f"./Executed/Checkpoints/{folder}"
    configs["solver"]["max_epochs"] = max_epochs
    configs["solver"]["save_cycle"] = max_epochs // num_checkpoints
    

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        raise RuntimeError("CPU is not supported, please use GPU.")

    model = instantiate_from_config(configs['model']).to(device)
    trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)
    return trainer


def visualize(data, title, filename, parent_folder, ylim=None):
    # Create a folder
    os.makedirs(f"./Figs", exist_ok=True)
    # Create subplots to accomodate 40 samples
    fig, axs = plt.subplots(5, 4, figsize=(10, 8))
    # Plot original data samples
    for i in range(5):
        for j in range(4):
            fig.suptitle(title)
            axs[i, j].plot(data[i * 4 + j])
            if ylim is not None:
                axs[i, j].set_ylim(ylim)
    plt.tight_layout()
    extra = '_y_limited' if ylim is not None else ''
    plt.savefig(f"./Figs/{parent_folder}/{filename + extra}.png")
    plt.close()
    
def get_labels_dict():
    return {
        2: "WALK",
        3: "STAIR UP",
        4: "STAIR DOWN",
        0: "SIT",
        1: "STAND",
    }

def data_to_2d(original_data, fake_data=None):
    # Transform the original data to 2d using tsne
    tsne = TSNE(n_components=2, random_state=42)
    if fake_data is not None:
        mix_data = np.concatenate((original_data, fake_data), axis=0)
        mix_data_2d = tsne.fit_transform(mix_data.reshape(mix_data.shape[0], -1))
        ori_data_tsne = mix_data_2d[:original_data.shape[0]]
        fake_data_tsne = mix_data_2d[original_data.shape[0]:]
        return ori_data_tsne, fake_data_tsne
    else:
        data_2d = tsne.fit_transform(original_data.reshape(original_data.shape[0], -1))
        return data_2d

def obtain_embedding_from_model(model, data):
    model.eval()
    timestep = 0
    timestep_tensor = torch.ones(data.shape[0], dtype=torch.long)
    timestep_tensor = timestep_tensor.to("cuda") * timestep
    with torch.no_grad():
        input_data = torch.from_numpy(data).to("cuda")
        embedding = model.model.emb(input_data)
        embedding = model.model.pos_enc(embedding)
        embedding = model.model.encoder(embedding, timestep_tensor)
    return embedding.cpu().numpy()

