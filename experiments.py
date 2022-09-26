import json
import os
import yaml
import pandas as pd
from tabulate import tabulate
from pathlib import Path
from typing import List
from continual import create_base_model, continue_from_model, move_param_file


def run_continual_on_all_datasets_all_replays(basename: str, param_file: str, datasets: List[str], replay_values: List[int]):
    assert min(
        replay_values) >= 0, "All replay portions must be larger than 0 (percentage to replay)."
    assert max(
        replay_values) <= 100, "All replay portions must be smaller than 100 (percentage to replay)."

    move_param_file(param_file)

    for replay_value in replay_values:
        run_continual_on_all_datasets(basename, datasets, replay_value)


def run_continual_on_all_datasets(basename: str, datasets: List[str], replay_value: int):
    create_base_model(f"{basename}_{replay_value}_replay",
                      datasets[0], replay_value/100)
    
    for dataset in datasets[1:]:
        continue_from_model(
            f"{basename}_{replay_value}_replay", dataset, replay_value/100)


def convert_json_metrics_to_csv(basedir: str):
    path = Path("model_info", basedir)
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
    
    _sort_dataframe(basedir)


def _sort_dataframe(basedir):
    path = Path("model_info", basedir)
    metrics_path = path / "metrics.csv"
    
    df = pd.read_csv(metrics_path)
    df = df.sort_values(["model_version", "trained_data", "evaluated_data"], ascending=[True, False, False])
    df = df.reset_index(drop=True)
    
    df.to_csv(metrics_path, index=False)
    

def wanted_metrics(basedir, basedataset):
    path = Path("model_info", basedir)
    metrics_path = path / "metrics.csv"
    
    df = pd.read_csv(metrics_path)
    
    train_eval_same = df[df["trained_data"] == df["evaluated_data"]]
    eval_base_data = df[df["evaluated_data"] == basedataset]
    
    
    train_eval_same.to_csv(path / "same_train_eval.csv", index=False)
    train_eval_same.to_latex(path / "same_train_eval.tex")
    eval_base_data.to_csv(path / "eval_base_data.csv", index=False)
    eval_base_data.to_latex(path / "eval_base_data.tex")
    

def wanted_metrics_all():
    for dir in Path("model_info").glob("*"):
        basedir = Path(dir.name)
        basedataset = yaml.safe_load(open(dir / "continual.yaml"))["datasets"][0]
        wanted_metrics(basedir, basedataset)


def gather_all_wanted_metrics_different_replay(base):
    collected_eval_base_data = pd.DataFrame(index=["trained_data", "evaluated_data"])
    collected_same_train_eval = pd.DataFrame(index=["trained_data", "evaluated_data"])
    
    for dir in Path("model_info").glob(f"{base}_*"):
        eval_base_data = pd.read_csv(dir / "eval_base_data.csv")
        same_train_eval = pd.read_csv(dir / "same_train_eval.csv")
        
        replay_portion = dir.name.replace(base, "").split("_")[0]
        
        collected_eval_base_data
    
    collected_eval_base_data.to_csv(f"./{base}_eval_base.csv")
    collected_same_train_eval.to_csv(f"./{base}_same_train.csv")
    

if __name__ == "__main__":
    cnc_datasets = [
        "cnc_milling_with_toolwear_baseline",
        "cnc_milling_with_toolwear_1",
        "cnc_milling_with_toolwear_2",
        "cnc_milling_with_toolwear_4",
        "cnc_milling_with_toolwear_5",
        "cnc_milling_with_toolwear_6",
        "cnc_milling_with_toolwear_8",
        "cnc_milling_with_toolwear_9",
        "cnc_milling_with_toolwear_10",
        "cnc_milling_with_toolwear_11",
        "cnc_milling_with_toolwear_12",
        "cnc_milling_with_toolwear_13",
        "cnc_milling_with_toolwear_14",
        # "cnc_milling_with_toolwear_16",
        "cnc_milling_with_toolwear_17",
        "cnc_milling_with_toolwear_18",
    ]
    #run_continual_on_all_datasets_all_replays("cnc_milling_toolwear", "cnc_toolwear.yaml", cnc_datasets, [100, 60, 20, 0])


    broaching_twd_datasets = [
        "broaching_twd_1X",
        "broaching_twd_2X",
        "broaching_twd_3X",
        "broaching_twd_4X",
        "broaching_twd_5X",
    ]
    #run_continual_on_all_datasets_all_replays("broaching_twd", "broaching_twd.yaml", broaching_twd_datasets, [100, 60, 20, 0])


    broaching_toolwear_datasets = [
        "broaching_toolwear_1X",
        "broaching_toolwear_2X",
        "broaching_toolwear_3X",
        "broaching_toolwear_4X",
        "broaching_toolwear_5X",
    ]
    #run_continual_on_all_datasets_all_replays("broaching_toolwear", "broaching_toolwear.yaml", broaching_toolwear_datasets, [100, 60, 20, 0])

    #[convert_json_metrics_to_csv(dir) for dir in os.listdir("model_info")]
    
    wanted_metrics_all()

