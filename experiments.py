from continual import baseline_to_tex, baseline_metrics_to_csv, combine_metrics_different_replay, evaluate_basemodel_on_all, run_continual_on_all_datasets_all_replays
import numpy as np
np.random.seed(2022)


def baseline_metrics(name, dir_for_baseline, eval_datasets):
    evaluate_basemodel_on_all(dir_for_baseline, eval_datasets)
    baseline_metrics_to_csv(dir_for_baseline)
    baseline_to_tex(name, dir_for_baseline)


def cnc_experiment():
    experiment_name = "cnc_milling_toolwear"
    replay_portions = [0, 20, 60, 100]
    # Commented datasets did not finish machining or is in the baseline dataset
    # Baseline consists of exp 1 and 6
    cnc_datasets = [
        "cnc_milling_with_toolwear_baseline",
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
    ]
    
    """ cnc_datasets = [
        "cnc_milling_with_toolwear_1_dataset",
        "cnc_milling_with_toolwear_2_dataset",
        "cnc_milling_with_toolwear_3_dataset",
        "cnc_milling_with_toolwear_4_dataset",
        "cnc_milling_with_toolwear_5_dataset",
    ] """
    
    run_continual_on_all_datasets_all_replays(
        experiment_name, "cnc_toolwear.yaml",
        cnc_datasets, replay_portions)
    
    baseline_metrics("cnc_milling_baseline.tex", f"{experiment_name}_{replay_portions[0]}_replay", cnc_datasets)
    
    cnc = [f"{experiment_name}_{r}_replay" for r in replay_portions]
    
    combine_metrics_different_replay(
        "cnc_eval_prev.tex",
        "eval_combined_previous_data.csv",
        cnc,
        replay_portions,
        "accuracy"
    )
    combine_metrics_different_replay(
        "cnc_eval_recent.tex",
        "same_train_eval.csv",
        cnc,
        replay_portions,
        "accuracy"
    )
    
def bosch_cnc_experiment():
    experiment_name = "bosch_cnc_vibration"
    replay_portions = [0, 20, 60, 100]
    # Commented datasets did not finish machining or is in the baseline dataset
    # Baseline consists of exp 1 and 6
    cnc_datasets = [
        "bosch_cnc_M01_2019_02",
        "bosch_cnc_M01_2019_08",
        "bosch_cnc_M01_2020_02",
        "bosch_cnc_M01_2021_02",
        "bosch_cnc_M01_2021_08",
    ]

    run_continual_on_all_datasets_all_replays(
        experiment_name, "bosch_cnc.yaml",
        cnc_datasets, replay_portions)
    
    baseline_metrics("bosch_cnc_baseline.tex", f"{experiment_name}_{replay_portions[0]}_replay", cnc_datasets)
    
    cnc = [f"{experiment_name}_{r}_replay" for r in replay_portions]
    
    combine_metrics_different_replay(
        "bosch_cnc_eval_prev.tex",
        "eval_combined_previous_data.csv",
        cnc,
        replay_portions,
        "accuracy"
    )
    
    combine_metrics_different_replay(
        "bosch_cnc_eval_recent.tex",
        "same_train_eval.csv",
        cnc,
        replay_portions,
        "accuracy"
    )


def broaching_tw_experiment():
    experiment_name = "broaching_toolwear"
    replay_portions = [0, 20, 60, 100]
    broaching_toolwear_datasets = [
        "broaching_toolwear_1X",
        "broaching_toolwear_2X",
        "broaching_toolwear_3X",
        "broaching_toolwear_4X",
        "broaching_toolwear_5X",
    ]
    
    run_continual_on_all_datasets_all_replays(
        experiment_name, "broaching_toolwear.yaml",
        broaching_toolwear_datasets, replay_portions)
    
    baseline_metrics("broaching_toolwear_baseline.tex", f"{experiment_name}_{replay_portions[0]}_replay", broaching_toolwear_datasets)
    
    btw = [f"{experiment_name}_{r}_replay" for r in replay_portions]
    
    combine_metrics_different_replay(
        "broaching_eval_prev.tex",
        "eval_combined_previous_data.csv",
        btw,
        replay_portions,
        "r2"
    )
    
    combine_metrics_different_replay(
        "broaching_eval_recent.tex",
        "same_train_eval.csv",
        btw,
        replay_portions,
        "r2"
    )


if __name__ == "__main__":
    
    #bosch_cnc_experiment()
    
    #cnc_experiment()
    
    broaching_tw_experiment()
