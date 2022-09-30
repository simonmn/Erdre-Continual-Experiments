from continual import baseline_to_tex, baseline_metrics_to_csv, combine_metrics_different_replay, run_continual_on_all_datasets_all_replays
import numpy as np
np.random.seed(2022)


def cnc_experiment():
    experiment_name = "cnc_milling_toolwear"
    replay_portions = [0, 20, 60, 100]
    # Commented datasets did not finish machining or is in the baseline dataset
    # Baseline consists of exp 2 and 6
    cnc_datasets = [
        "cnc_milling_with_toolwear_baseline",
        "cnc_milling_with_toolwear_01",
        #"cnc_milling_with_toolwear_02",
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
    run_continual_on_all_datasets_all_replays(
        experiment_name, "cnc_toolwear.yaml",
        cnc_datasets, replay_portions)
    
    cnc = ["cnc_milling_toolwear_0_replay",
           "cnc_milling_toolwear_20_replay",
           "cnc_milling_toolwear_60_replay",
           "cnc_milling_toolwear_100_replay"]
    baseline_metrics_to_csv(cnc[0])
    baseline_to_tex("cnc_milling_baseline.tex", cnc[0], "accuracy")
    
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
    
    btw = ["broaching_toolwear_0_replay",
           "broaching_toolwear_20_replay",
           "broaching_toolwear_60_replay",
           "broaching_toolwear_100_replay"]
    baseline_metrics_to_csv(btw[0])
    baseline_to_tex("broaching_toolwear_baseline.tex", btw[0], "r2")
    
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
    
    cnc_experiment()
    
    #broaching_tw_experiment()