import os
import torch
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

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
    # Read the original data
    data_module = get_datamodule(dataset_name)
    # Load the training data
    train_x, train_y = data_module.train_dataloader().dataset[:]
    original_classes = np.unique(train_y)
    train_data_x = []
    train_data_y = []
    # Filter the data for the specific class
    classes_to_replace = original_classes if class_id is None else [class_id]
    for class_to_replace in classes_to_replace:
        # Load the synthetic data
        synthetic_data_path = Path("./Executed/SyntheticData/") / preffix / str(milestone) / f"{preffix}-trained_on-class_{class_to_replace}-{scaler}_scaled-{max_epochs}_epochs"
        synthetic_data = np.load(synthetic_data_path / "fake_data.npy")
        # Pick random samples from the synthetic data
        random_synthetic_samples = random.sample(list(synthetic_data), len(train_y[train_y == class_to_replace]))
        train_data_x.append(random_synthetic_samples)
        train_data_y.append([class_to_replace] * len(random_synthetic_samples))
    # Concatenate the original data with the synthetic data
    original_data_left = train_x[~np.isin(train_y, classes_to_replace)]
    original_labels_left = train_y[~np.isin(train_y, classes_to_replace)]
    train_x = np.concatenate((original_data_left, *train_data_x))
    train_y = np.concatenate((original_labels_left, *train_data_y))
    # Load the test data
    data_module.setup(stage="test")
    test_x, test_y = data_module.test_dataloader().dataset[:]
    # Define the classifiers
    classifiers_class = {
        "Random Forest": RandomForestClassifier,
        "SVM": SVC
    }
    classifiers_metrics = {}
    # Train and evaluate each classifier
    for clf_name, clf_class in classifiers_class.items():
        classifier = clf_class()
        classifier.fit(train_x.reshape(len(train_x), -1), train_y)
        y_pred = classifier.predict(test_x.reshape(len(test_x), -1))
        report = classification_report(test_y, y_pred, output_dict=True, zero_division=0)
        classifiers_metrics[clf_name] = {
            "accuracy": report["accuracy"],
            "precision": report["weighted avg"]["precision"],
            "recall": report["weighted avg"]["recall"],
            "f1-score": report["weighted avg"]["f1-score"]
        }
        print(f"Classifier: {clf_name}, Metrics: {classifiers_metrics[clf_name]}")
        # Print the confusion matrix
        conf_matrix = confusion_matrix(test_y, y_pred) 
        print(f"Confusion Matrix:\n{conf_matrix}")

    

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