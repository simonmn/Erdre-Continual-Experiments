import os
from pathlib import Path
from typing import List
import numpy as np
import shutil
import yaml

from src.profiling import profiling
from src.clean import clean
from src.featurize import featurize
from src.split import split
from src.scale import scale
from src.sequentialize import sequentialize
from src.combine import combine
from src.train import train
from src.evaluate import evaluate

from continual_config import (
    DATA_COMBINED_PATH,
    DATA_FEATURIZED_PATH,
    DATA_PATH_RAW,
    DATA_SCALED_PATH,
    DATA_SEQUENTIALIZED_PATH,
    DATA_SPLIT_PATH,
    INPUT_SCALER_PATH,
    INTERVALS_PLOT_PATH,
    METRICS_FILE_PATH,
    OUTPUT_SCALER_PATH,
    PLOTS_PATH,
    PREDICTION_PLOT_PATH,
    PREDICTIONS_FILE_PATH,
    PREDICTIONS_PATH,
    PROFILE_PATH,
    FEATURES_PATH,
    OUTPUT_FEATURES_PATH,
    INPUT_FEATURES_PATH,
    REMOVABLE_FEATURES,
    DATA_CLEANED_PATH,
    SCALER_PATH,
    TRAININGLOSS_PLOT_PATH,
)


def create_base_model(model_name: str, dataset: str, replay_portion: float):
    clean_folders()
    update_dataset_param(dataset)

    profiling(DATA_PATH_RAW=DATA_PATH_RAW, PROFILE_PATH=PROFILE_PATH)

    clean(DATA_PATH_RAW=DATA_PATH_RAW,
          inference_df=None,
          FEATURES_PATH=FEATURES_PATH,
          PROFILE_PATH=PROFILE_PATH,
          REMOVABLE_FEATURES=REMOVABLE_FEATURES,
          DATA_CLEANED_PATH=DATA_CLEANED_PATH,
          OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH
          )

    featurize(DATA_CLEANED_PATH=DATA_CLEANED_PATH,
              inference=False,
              DATA_FEATURIZED_PATH=DATA_FEATURIZED_PATH,
              FEATURES_PATH=FEATURES_PATH,
              INPUT_FEATURES_PATH=INPUT_FEATURES_PATH,
              OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
              PROFILE_PATH=PROFILE_PATH
              )

    split(DATA_FEATURIZED_PATH=DATA_FEATURIZED_PATH,
          DATA_SPLIT_PATH=DATA_SPLIT_PATH)

    scale(DATA_SPLIT_PATH=DATA_SPLIT_PATH,
          DATA_SCALED_PATH=DATA_SCALED_PATH,
          INPUT_SCALER_PATH=INPUT_SCALER_PATH,
          OUTPUT_SCALER_PATH=OUTPUT_SCALER_PATH,
          OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
          SCALER_PATH=SCALER_PATH)

    sequentialize(DATA_SCALED_PATH=DATA_SCALED_PATH,
                  DATA_SEQUENTIALIZED_PATH=DATA_SEQUENTIALIZED_PATH,
                  OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH)

    combine(DATA_SEQUENTIALIZED_PATH=DATA_SEQUENTIALIZED_PATH,
            DATA_COMBINED_PATH=DATA_COMBINED_PATH)

    move_replay_portion(dataset, DATA_COMBINED_PATH,
                        Path("./replay_data/"), replay_portion)

    train(DATA_TRAIN_PATH=DATA_COMBINED_PATH / "train.npz",
          MODELS_FILE_PATH=Path("model_info") / model_name / "model.h5",
          MODELS_PATH=Path("model_info") / model_name,
          model_exists=False,
          OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
          PLOTS_PATH=PLOTS_PATH,
          TRAININGLOSS_PLOT_PATH=TRAININGLOSS_PLOT_PATH)

    evaluate(MODELS_FILE_PATH=Path("model_info") / model_name / "model.h5",
             DATA_TRAIN_PATH=DATA_COMBINED_PATH / "train.npz",
             DATA_TEST_PATH=DATA_COMBINED_PATH / "test.npz",
             DATA_CALIBRATE_PATH=DATA_COMBINED_PATH / "calibrate.npz",
             INPUT_FEATURES_PATH=INPUT_FEATURES_PATH,
             INTERVALS_PLOT_PATH=INTERVALS_PLOT_PATH,
             METRICS_FILE_PATH=METRICS_FILE_PATH,
             OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
             PLOTS_PATH=PLOTS_PATH,
             PREDICTION_PLOT_PATH=PREDICTION_PLOT_PATH,
             PREDICTIONS_FILE_PATH=PREDICTIONS_FILE_PATH,
             PREDICTIONS_PATH=PREDICTIONS_PATH)

    model_path = Path("./model_info",  model_name)
    metrics_path = model_path / "metrics"
    metrics_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(METRICS_FILE_PATH, metrics_path / f"0-{dataset}-{dataset}.json")

    yaml_string = f"datasets: [{dataset}]"
    yaml_data = yaml.safe_load(yaml_string)
    yaml.safe_dump(yaml_data, open(model_path / "continual.yaml", "w"))


def continue_from_model(model_name: str, dataset: str, replay_portion: float):
    model_path = Path("./model_info",  model_name)
    datasets_yaml = yaml.safe_load(open(model_path / "continual.yaml", "r"))
    replay_datasets = datasets_yaml["datasets"]

    clean_folders()
    update_dataset_param(dataset)

    clean(DATA_PATH_RAW=DATA_PATH_RAW,
          inference_df=None,
          FEATURES_PATH=FEATURES_PATH,
          PROFILE_PATH=PROFILE_PATH,
          REMOVABLE_FEATURES=REMOVABLE_FEATURES,
          DATA_CLEANED_PATH=DATA_CLEANED_PATH,
          OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH
          )

    featurize(DATA_CLEANED_PATH=DATA_CLEANED_PATH,
              inference=False,
              DATA_FEATURIZED_PATH=DATA_FEATURIZED_PATH,
              FEATURES_PATH=FEATURES_PATH,
              INPUT_FEATURES_PATH=INPUT_FEATURES_PATH,
              OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
              PROFILE_PATH=PROFILE_PATH
              )

    split(DATA_FEATURIZED_PATH=DATA_FEATURIZED_PATH,
          DATA_SPLIT_PATH=DATA_SPLIT_PATH)

    scale(DATA_SPLIT_PATH=DATA_SPLIT_PATH,
          DATA_SCALED_PATH=DATA_SCALED_PATH,
          INPUT_SCALER_PATH=INPUT_SCALER_PATH,
          OUTPUT_SCALER_PATH=OUTPUT_SCALER_PATH,
          OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
          SCALER_PATH=SCALER_PATH,
          EXISTING_INPUT_SCALER=INPUT_SCALER_PATH,
          EXISTING_OUTPUT_SCALER=OUTPUT_SCALER_PATH)

    sequentialize(DATA_SCALED_PATH=DATA_SCALED_PATH,
                  DATA_SEQUENTIALIZED_PATH=DATA_SEQUENTIALIZED_PATH,
                  OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH)

    combine(DATA_SEQUENTIALIZED_PATH=DATA_SEQUENTIALIZED_PATH,
            DATA_COMBINED_PATH=DATA_COMBINED_PATH)

    move_replay_portion(dataset, DATA_COMBINED_PATH,
                        Path("./replay_data/"), replay_portion)

    combine_replay_data(replay_datasets, DATA_COMBINED_PATH,
                        Path("./replay_data"))

    train(DATA_TRAIN_PATH=DATA_COMBINED_PATH / "train.npz",
          MODELS_FILE_PATH=Path("model_info") / model_name / "model.h5",
          MODELS_PATH=Path("model_info") / model_name,
          model_exists=True,
          OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
          PLOTS_PATH=PLOTS_PATH,
          TRAININGLOSS_PLOT_PATH=TRAININGLOSS_PLOT_PATH)

    evaluate(MODELS_FILE_PATH=Path("model_info") / model_name / "model.h5",
             DATA_TRAIN_PATH=DATA_COMBINED_PATH / "train.npz",
             DATA_TEST_PATH=DATA_COMBINED_PATH / "test.npz",
             DATA_CALIBRATE_PATH=DATA_COMBINED_PATH / "calibrate.npz",
             INPUT_FEATURES_PATH=INPUT_FEATURES_PATH,
             INTERVALS_PLOT_PATH=INTERVALS_PLOT_PATH,
             METRICS_FILE_PATH=METRICS_FILE_PATH,
             OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
             PLOTS_PATH=PLOTS_PATH,
             PREDICTION_PLOT_PATH=PREDICTION_PLOT_PATH,
             PREDICTIONS_FILE_PATH=PREDICTIONS_FILE_PATH,
             PREDICTIONS_PATH=PREDICTIONS_PATH)

    metrics_path = model_path / "metrics"
    shutil.copy(METRICS_FILE_PATH, metrics_path / f"{len(replay_datasets)}-{dataset}-{dataset}.json")
    
    for ds in replay_datasets:
        replay_path = Path("./replay_data")
        
        evaluate(MODELS_FILE_PATH=Path("model_info") / model_name / "model.h5",
                 DATA_TRAIN_PATH=replay_path / "train" / f"{ds}.npz",
                 DATA_TEST_PATH=replay_path / "test" / f"{ds}.npz",
                 DATA_CALIBRATE_PATH=replay_path / "calibrate" / f"{ds}.npz",
                 INPUT_FEATURES_PATH=INPUT_FEATURES_PATH,
                 INTERVALS_PLOT_PATH=INTERVALS_PLOT_PATH,
                 METRICS_FILE_PATH=METRICS_FILE_PATH,
                 OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
                 PLOTS_PATH=PLOTS_PATH,
                 PREDICTION_PLOT_PATH=PREDICTION_PLOT_PATH,
                 PREDICTIONS_FILE_PATH=PREDICTIONS_FILE_PATH,
                 PREDICTIONS_PATH=PREDICTIONS_PATH)
        
        shutil.copy(METRICS_FILE_PATH,  metrics_path / f"{len(replay_datasets)}-{dataset}-{ds}.json")

    replay_datasets.append(dataset)
    yaml.safe_dump(datasets_yaml, open(model_path / "continual.yaml", "w"))


def move_replay_portion(dataset: str, source: Path, destination: Path, replay_portion: float):
    try:
        train = np.load(source / "train.npz")
        #test = np.load(source / "test.npz")
        #calibrate = np.load(source / "calibrate.npz")

        X_train = train["X"]
        y_train = train["y"]
        train_indices = np.random.choice(
            X_train.shape[0], int(replay_portion * X_train.shape[0]))

        X_train = X_train[train_indices, :]
        y_train = y_train[train_indices, :]

        np.savez(destination / "train" /
                 f"{dataset}.npz", X=X_train, y=y_train)

        shutil.copy(source / "test.npz", destination /
                    "test" / f"{dataset}.npz")
        shutil.copy(source / "calibrate.npz", destination /
                    "calibrate" / f"{dataset}.npz")

    except FileNotFoundError:
        # print("FileNotFoundError")
        pass


def combine_replay_data(datasets: List[str], source_new: Path, source_replay: Path):
    train = np.load(source_new / "train.npz", "r")
    X_train = train["X"]
    y_train = train["y"]

    for dataset in datasets:
        train_path = source_replay / "train" / f"{dataset}.npz"
        train_replay = np.load(train_path, "r")

        X_train = np.vstack((X_train, train_replay["X"]))
        y_train = np.vstack((y_train, train_replay["y"]))

    np.savez(source_new / "train.npz", X=X_train, y=y_train)


def update_dataset_param(dataset: str):
    params = yaml.safe_load(open("params.yaml", "r"))

    params["profile"]["dataset"] = dataset

    yaml.safe_dump(params, open("params.yaml", "w"))


def clean_folders():
    for dir in [DATA_CLEANED_PATH, DATA_FEATURIZED_PATH, DATA_SCALED_PATH, DATA_SEQUENTIALIZED_PATH, DATA_SPLIT_PATH]:
        for f in os.listdir(dir):
            os.remove(os.path.join(dir, f))


def move_param_file(param_filename: str):    
    shutil.copy(Path("./param_files") / param_filename, "params.yaml")
    

if __name__ == "__main__":
    pass
