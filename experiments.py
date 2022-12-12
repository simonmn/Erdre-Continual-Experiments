from pathlib import Path
from typing import List
import numpy as np

from continual import baseline_to_tex, baseline_metrics_to_csv, combine_metrics_different_replay, evaluate_basemodel_on_all, run_continual_on_all_datasets_all_replays

np.random.seed(2022)


def baseline_metrics(name, dir_for_baseline, eval_datasets):
    evaluate_basemodel_on_all(dir_for_baseline, eval_datasets)
    baseline_metrics_to_csv(dir_for_baseline)
    baseline_to_tex(name, dir_for_baseline)


def experiment(exp_name: str, replay_portions: List[int], datasets: List[str], yaml_filename: str, metric: str):
    run_continual_on_all_datasets_all_replays(
        exp_name, yaml_filename,
        datasets, replay_portions)
    
    baseline_metrics(Path("./models", exp_name, "baseline.tex"), Path("./models", exp_name, str(replay_portions[0])), datasets)
    
    combine_metrics_different_replay(
        "eval_prev.tex",
        "eval_combined_previous_data.csv",
        exp_name,
        replay_portions,
        metric
    )
    
    combine_metrics_different_replay(
        "eval_recent.tex",
        "same_train_eval.csv",
        exp_name,
        replay_portions,
        metric
    )


def cnc_experiment():
    experiment(
        "cnc_milling_toolwear",
        [0, 20, 60, 100],
        ["cnc_milling_with_toolwear_baseline",
        #"cnc_milling_with_toolwear_01",
        "cnc_milling_with_toolwear_02",
        "cnc_milling_with_toolwear_03",
        #"cnc_milling_with_toolwear_04",
        #"cnc_milling_with_toolwear_05",
        #"cnc_milling_with_toolwear_06",
        #"cnc_milling_with_toolwear_07",
        "cnc_milling_with_toolwear_09",
        "cnc_milling_with_toolwear_10",
        "cnc_milling_with_toolwear_11",
        "cnc_milling_with_toolwear_12",
        "cnc_milling_with_toolwear_13",
        "cnc_milling_with_toolwear_14",
        "cnc_milling_with_toolwear_15",
        #"cnc_milling_with_toolwear_16",
        "cnc_milling_with_toolwear_17",
        "cnc_milling_with_toolwear_18",
        ],
        "cnc_toolwear.yaml",
        "accuracy"
    )


def bosch_cnc_experiment():
    experiment(
        "bosch_cnc_vibration",
        [0, 20, 60, 100],
        ["bosch_cnc_M01_2019_02",
        "bosch_cnc_M01_2019_08",
        "bosch_cnc_M01_2020_02",
        "bosch_cnc_M01_2021_02",
        "bosch_cnc_M01_2021_08"],
        "bosch_cnc.yaml",
        "accuracy"
    )


def broaching_tw_experiment():
    experiment(
        "broaching_toolwear",
        [0, 20, 60, 100],
        ["broaching_toolwear_1X",
        "broaching_toolwear_2X",
        "broaching_toolwear_3X",
        "broaching_toolwear_4X",
        "broaching_toolwear_5X"],
        "broaching_toolwear.yaml",
        "r2"
        )


if __name__ == "__main__":
    
    #bosch_cnc_experiment()
    
    #cnc_experiment()
    
    #broaching_tw_experiment()
    
    piston_rod_datasets = ["piston_rod_set_1",
                           "piston_rod_set_2",
                           "piston_rod_set_3",
                           "piston_rod_set_4",
                           "piston_rod_set_5"]
    experiment("piston_rod", [0, 20, 60, 100], piston_rod_datasets, "piston_rod.yaml", "r2")
