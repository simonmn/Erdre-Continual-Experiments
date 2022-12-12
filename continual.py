import os
from pathlib import Path
from typing import List
import numpy as np
import shutil
import yaml
import json
import pandas as pd

np.random.seed(2022)
from erdre import profiling, clean, featurize, split, scale, sequentialize, combine, train, evaluate

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


def create_base_model(model_name: str, dataset: str, replay_portion: float, model_path: Path):
    np.random.seed(2022)
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

    # Store portion for future replay
    move_replay_portion(dataset, DATA_COMBINED_PATH,
                        Path("./replay_data/"), replay_portion/100)

    #model_path = Path("./models",  model_name, replay_portion)
    model_file_path = model_path / "model.h5"

    # Train new model
    train(DATA_TRAIN_PATH=DATA_COMBINED_PATH / "train.npz",
          MODELS_FILE_PATH=model_file_path,
          MODELS_PATH=model_path,
          model_exists=False,
          OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
          PLOTS_PATH=PLOTS_PATH,
          TRAININGLOSS_PLOT_PATH=TRAININGLOSS_PLOT_PATH)

    # Evaluate model on basedata
    evaluate(MODELS_FILE_PATH=model_file_path,
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
    metrics_path.mkdir(parents=True, exist_ok=True)
    model_storage_path = model_path / "models"
    model_storage_path.mkdir(parents=True, exist_ok=True)
    
    # Store metric
    shutil.copy(METRICS_FILE_PATH, metrics_path / f"0-{dataset}-{dataset}.json")
    # Store model
    shutil.copy(model_path / "model.h5", model_storage_path / "model_0.h5")

    yaml_string = f"datasets: [{dataset}]"
    yaml_data = yaml.safe_load(yaml_string)
    yaml.safe_dump(yaml_data, open(model_path / "history.yaml", "w"))


def continue_from_model(model_name: str, dataset: str, replay_portion: float, model_path: Path):
    np.random.seed(2022)
    datasets_yaml = yaml.safe_load(open(model_path / "history.yaml", "r"))
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

    # Store portion for future replay
    move_replay_portion(dataset, DATA_COMBINED_PATH,
                        Path("./replay_data/"), replay_portion/100)

    replay_path = Path("./replay_data")
    # Prepare new and replay data for training
    combine_replay_data(replay_datasets, DATA_COMBINED_PATH,
                        replay_path)

    #model_path = Path("./models",  model_name, replay_portion)
    model_file_path = model_path / "model.h5"

    # Load old model and train it
    train(DATA_TRAIN_PATH=DATA_COMBINED_PATH / "train.npz",
          MODELS_FILE_PATH=model_file_path,
          MODELS_PATH=model_path,
          model_exists=True,
          OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
          PLOTS_PATH=PLOTS_PATH,
          TRAININGLOSS_PLOT_PATH=TRAININGLOSS_PLOT_PATH)

    # Evaluate model on newest data
    evaluate(MODELS_FILE_PATH=model_file_path,
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
    # Store metrics
    shutil.copy(METRICS_FILE_PATH, metrics_path / f"{len(replay_datasets)}-{dataset}-{dataset}.json")
    
    # Prepare replay test data
    combine_test_data(replay_datasets, DATA_COMBINED_PATH,
                        replay_path)
    
    # Evaluate on replay test data
    evaluate(MODELS_FILE_PATH=model_path / "model.h5",
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
    
    shutil.copy(METRICS_FILE_PATH, metrics_path / f"{len(replay_datasets)}-{dataset}-combined_previous_data.json")
    
    # Evaluate on each available replay dataset (individually)
    for ds in replay_datasets:
        evaluate(MODELS_FILE_PATH=model_path / "model.h5",
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
    yaml.safe_dump(datasets_yaml, open(model_path / "history.yaml", "w"))


def run_continual_on_all_datasets_all_replays(basename: str, param_file: str, datasets: List[str], replay_values: List[int]):
    np.random.seed(2022)
    assert min(
        replay_values) >= 0, "All replay portions must be larger than 0 (percentage to replay)."
    assert max(
        replay_values) <= 100, "All replay portions must be smaller than 100 (percentage to replay)."

    move_param_file(param_file)
    
    # Run first time, creates original model which we use for other replay values
    run_continual_on_all_datasets(basename, datasets, replay_values[0])
    evaluate_basemodel_on_all(Path("./models", basename, str(replay_values[0])), datasets)

    basemodel = Path("./models", basename, str(replay_values[0]), "models", "model_0.h5")
    for replay_value in replay_values[1:]:
        # Use basemodel instead of creating new, currently a new model is created, but discarded as
        # there is currently tight coupling between moving replay portion for future use and 
        # creating the base model.
        run_continual_on_all_datasets(basename, datasets, replay_value, basemodel)
        evaluate_basemodel_on_all(Path("./models", basename, str(replay_value)), datasets)


def run_continual_on_all_datasets(basename: str, datasets: List[str], replay_value: int, basemodel: Path = None):
    np.random.seed(2022)
    
    model_path = Path("./models",  basename, str(replay_value))
    
    create_base_model(basename,
                      datasets[0], replay_value, model_path)
    
    if basemodel is not None:
        shutil.copy(basemodel, Path(model_path, "models", "model_0.h5"))
        shutil.copy(basemodel, Path(model_path, "model.h5"))
    
    for dataset in datasets[1:]:
        continue_from_model(basename, dataset, replay_value, model_path)
    convert_json_metrics_to_csv(basename, replay_value)
    wanted_metrics(model_path, datasets[0])


def evaluate_basemodel_on_all(model_path: Path, datasets: List[str]):
    np.random.seed(2022)
    print(f"Evaluating {model_path} on {datasets}")
    replay_path = Path("./replay_data")
    for dataset in datasets:
        train_path = replay_path / "train" / f"{dataset}.npz"
        test_path = replay_path / "test" / f"{dataset}.npz"
        calibrate_path = replay_path / "calibrate" / f"{dataset}.npz"
        
        evaluate(MODELS_FILE_PATH=model_path / "model.h5",
             DATA_TRAIN_PATH=train_path,
             DATA_TEST_PATH=test_path,
             DATA_CALIBRATE_PATH=calibrate_path,
             INPUT_FEATURES_PATH=INPUT_FEATURES_PATH,
             INTERVALS_PLOT_PATH=INTERVALS_PLOT_PATH,
             METRICS_FILE_PATH=METRICS_FILE_PATH,
             OUTPUT_FEATURES_PATH=OUTPUT_FEATURES_PATH,
             PLOTS_PATH=PLOTS_PATH,
             PREDICTION_PLOT_PATH=PREDICTION_PLOT_PATH,
             PREDICTIONS_FILE_PATH=PREDICTIONS_FILE_PATH,
             PREDICTIONS_PATH=PREDICTIONS_PATH)
        
        metrics_path = model_path / "metrics" / "baseline_model"
        metrics_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(METRICS_FILE_PATH, metrics_path / f"0-baseline-{dataset}.json")
        
        csv_file = open(model_path / "baseline_metrics.csv", "w")
        
        for file in metrics_path.glob("*.json"):
            metrics = open(file, "r")
            json_metrics = json.load(metrics)
            csv_file.write(f"model_version,trained_data,evaluated_data,{','.join(json_metrics.keys())}\n")
            metrics.close()
            break
        
        for file in metrics_path.glob("*.json"):
            idx, trained_data, evaluated_data = str(file).split("/")[-1][:-5].split("-")
            metrics = open(file, "r")
            json_metrics = json.load(metrics)
            
            csv_file.write(f"{idx},{trained_data},{evaluated_data},{','.join([str(json_metrics[key]) for key in json_metrics.keys()])}\n")
        
        csv_file.close()


def convert_json_metrics_to_csv(basename: str, replay_value: int):
    path = Path("models", basename, str(replay_value))
    csv_file = open(path / "metrics.csv", "w")
    metrics_path = path / "metrics"
    
    for file in metrics_path.glob("*.json"):
        metrics = open(file, "r")
        json_metrics = json.load(metrics)
        csv_file.write(f"model_version,trained_data,evaluated_data,{','.join(json_metrics.keys())}\n")
        metrics.close()
        break
    
    for file in metrics_path.glob("*.json"):
        idx, trained_data, evaluated_data = str(file).split("/")[-1][:-5].split("-")
        metrics = open(file, "r")
        json_metrics = json.load(metrics)
        
        csv_file.write(f"{idx},{trained_data},{evaluated_data},{','.join([str(json_metrics[key]) for key in json_metrics.keys()])}\n")
    
    csv_file.close()
    
    _sort_dataframe(path)


def baseline_metrics_to_csv(basedir: str):
    path = Path(basedir, "metrics")
    csv_file_path = path.parent / "baseline_metrics.csv"
    csv_file = open(csv_file_path, "w")
    metrics_path = path / "baseline_model"
    
    for file in metrics_path.glob("*.json"):
        with open(file, "r") as metrics:
            json_metrics = json.load(metrics)
        csv_file.write(f"model_version,trained_data,evaluated_data,{','.join(json_metrics.keys())}\n")
        break
    
    for file in metrics_path.glob("*.json"):
        idx, trained_data, evaluated_data = str(file).split("/")[-1][:-5].split("-")
        with open(file, "r") as metrics:
            json_metrics = json.load(metrics)
        
        csv_file.write(f"{idx},{trained_data},{evaluated_data},{','.join([str(json_metrics[key]) for key in json_metrics.keys()])}\n")
    
    csv_file.close()
    
    df = pd.read_csv(csv_file_path)
    df = df.sort_values(["model_version", "trained_data", "evaluated_data"], ascending=[True, True, True])
    df = df.reset_index(drop=True)
    cols = [col for col in df.columns if col not in ["mse", "rmse", "mape"]]
    df.to_csv(csv_file_path, index=False, columns=cols)


def baseline_to_tex(name: str, dir: str):
    csv_path = dir / "baseline_metrics.csv"
    df = pd.read_csv(csv_path)
    df.to_latex(name, index=False, columns=[col for col in df.columns if col not in ["model_version", "trained_data", "mse", "rmse" "mape"]], float_format="%.3f")


def combine_metrics_different_replay(gathered_filename: str, filename: str, exp_name: str, replays: List[int], metric: str):
    complete_df = pd.DataFrame()
    for r in replays:
        csv_path = Path("./models", exp_name, str(r), filename)
        df = pd.read_csv(csv_path)
        complete_df["evaluated_data"] = df["evaluated_data"]
        complete_df[f"{metric}_{r}"] = df[metric]
    complete_df.to_latex(Path("./models", exp_name, gathered_filename), index=False, float_format="%.3f")


def _sort_dataframe(path: Path):
    metrics_path = path / "metrics.csv"
    
    df = pd.read_csv(metrics_path)
    df = df.sort_values(["model_version", "trained_data", "evaluated_data"], ascending=[True, True, True])
    df = df.reset_index(drop=True)
    df.to_csv(metrics_path, index=False, columns=[col for col in df.columns if col not in ["mse", "rmse" "mape"]])


def wanted_metrics(path, basedataset):
    metrics_path = path / "metrics.csv"
    
    df = pd.read_csv(metrics_path)
    
    train_eval_same = df[df["trained_data"] == df["evaluated_data"]]
    eval_base_data = df[df["evaluated_data"] == basedataset]
    eval_combined_previous_data = df[df["evaluated_data"] == "combined_previous_data"]
    
    included_columns = [col for col in train_eval_same.columns if col != "model_version"]
    
    #train_eval_same.to_latex(path / "same_train_eval.tex", index=False, columns=included_columns)
    #eval_base_data.to_latex(path / "eval_base_data.tex", index=False, columns=included_columns)
    #eval_combined_previous_data.to_latex(path / "eval_combined_previous_data.tex", index=False, columns=included_columns)
    train_eval_same.to_csv(path / "same_train_eval.csv", index=False, columns=included_columns)
    eval_base_data.to_csv(path / "eval_base_data.csv", index=False, columns=included_columns)
    eval_combined_previous_data.to_csv(path / "eval_combined_previous_data.csv", index=False, columns=included_columns)


def move_replay_portion(dataset: str, source: Path, destination: Path, replay_portion: float):
    np.random.seed(2022)
    try:
        train = np.load(source / "train.npz")
        #test = np.load(source / "test.npz")
        #calibrate = np.load(source / "calibrate.npz")

        X_train = train["X"]
        y_train = train["y"]
        rng = np.random.RandomState(2022)
        train_indices = rng.choice(
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


def combine_test_data(datasets: List[str], source_new: Path, source_replay: Path):
    test = np.load(source_new / "test.npz", "r")
    X_test = test["X"]
    y_test = test["y"]

    for dataset in datasets:
        test_path = source_replay / "test" / f"{dataset}.npz"
        test_replay = np.load(test_path, "r")

        X_test = np.vstack((X_test, test_replay["X"]))
        y_test = np.vstack((y_test, test_replay["y"]))

    np.savez(source_new / "test.npz", X=X_test, y=y_test)


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
    
    replay_path = Path("replay_data")
    
    (replay_path / "train").mkdir(parents=True, exist_ok=True)
    (replay_path / "test").mkdir(parents=True, exist_ok=True)
    (replay_path / "calibrate").mkdir(parents=True, exist_ok=True)


def move_param_file(param_filename: str):    
    shutil.copy(Path("./param_files") / param_filename, "params.yaml")
    

if __name__ == "__main__":
    pass
