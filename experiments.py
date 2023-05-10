from pathlib import Path
from typing import List
import numpy as np

from continual import (
    baseline_to_tex,
    baseline_metrics_to_csv,
    combine_metrics_different_replay,
    evaluate_basemodel_on_all,
    run_continual_on_all_datasets_all_replays,
)

np.random.seed(2022)


def baseline_metrics(name, dir_for_baseline, eval_datasets):
    evaluate_basemodel_on_all(dir_for_baseline, eval_datasets)
    baseline_metrics_to_csv(dir_for_baseline)
    baseline_to_tex(name, dir_for_baseline)


def experiment(
    exp_name: str,
    replay_portions: List[int],
    datasets: List[str],
    yaml_filename: str,
    metric: str,
    track_emissions: bool,
):
    run_continual_on_all_datasets_all_replays(
        exp_name, yaml_filename, datasets, replay_portions, track_emissions
    )

    baseline_metrics(
        Path("./models", exp_name, "baseline.tex"),
        Path("./models", exp_name, str(replay_portions[0])),
        datasets,
    )

    combine_metrics_different_replay(
        "eval_prev.tex",
        "eval_combined_previous_data.csv",
        exp_name,
        replay_portions,
        metric,
    )

    combine_metrics_different_replay(
        "eval_recent.tex", "same_train_eval.csv", exp_name, replay_portions, metric
    )


def bosch_cnc_experiment(track_emissions: bool = False):
    experiment(
        "bosch_cnc_vibration",
        [0, 20, 60, 100],
        [
            "bosch_cnc_M01_2019_02",
            "bosch_cnc_M01_2019_08",
            "bosch_cnc_M01_2020_02",
            "bosch_cnc_M01_2021_02",
            "bosch_cnc_M01_2021_08",
        ],
        "bosch_cnc.yaml",
        "f1",
        track_emissions,
    )


def broaching_tw_experiment(track_emissions: bool = False):
    experiment(
        "broaching_toolwear",
        [0, 20, 60, 100],
        [
            "broaching_toolwear_1X",
            "broaching_toolwear_2X",
            "broaching_toolwear_3X",
            "broaching_toolwear_4X",
            "broaching_toolwear_5X",
        ],
        "broaching_toolwear.yaml",
        "r2",
        track_emissions,
    )


def piston_rod_experiment(track_emissions: bool = False):
    piston_rod_datasets = [
        "piston_rod_set_1",
        "piston_rod_set_2",
        "piston_rod_set_3",
        "piston_rod_set_4",
        "piston_rod_set_5",
    ]
    experiment(
        "piston_rod",
        [0, 20, 60, 100],
        piston_rod_datasets,
        "piston_rod.yaml",
        "r2",
        track_emissions,
    )


if __name__ == "__main__":
    import tensorflow as tf

    print("GPUs:", tf.config.list_physical_devices("GPU"))

    bosch_cnc_experiment(True)

    # broaching_tw_experiment(True)

    # piston_rod_experiment(True)
