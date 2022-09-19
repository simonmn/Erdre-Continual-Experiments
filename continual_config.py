from pathlib import Path

#DATA_PATH = Path("./assets") / "data"
ASSETS_PATH = Path("./assets")

PROFILE_PATH = ASSETS_PATH / "profile"
FEATURES_PATH = ASSETS_PATH / "profile"
INPUT_FEATURES_PATH = FEATURES_PATH / "input_features.csv"
OUTPUT_FEATURES_PATH = FEATURES_PATH / "output_features.csv"
REMOVABLE_FEATURES = FEATURES_PATH  / "removable_features.csv"

DATA_PATH = ASSETS_PATH / "data"
DATA_PATH_RAW = DATA_PATH / "raw"
DATA_CLEANED_PATH = DATA_PATH / "cleaned"
DATA_FEATURIZED_PATH = DATA_PATH / "featurized"
DATA_SEQUENTIALIZED_PATH = DATA_PATH / "sequentialized"
DATA_SPLIT_PATH = DATA_PATH / "split"
DATA_SCALED_PATH = DATA_PATH / "scaled"
DATA_COMBINED_PATH = DATA_PATH / "combined"

MODELS_PATH = ASSETS_PATH / "models"
MODELS_FILE_PATH = MODELS_PATH / "model.h5"

METRICS_PATH = ASSETS_PATH / "metrics"
METRICS_FILE_PATH = METRICS_PATH / "metrics.json"

PREDICTIONS_PATH = ASSETS_PATH / "predictions"
PREDICTIONS_FILE_PATH = PREDICTIONS_PATH / "predictions.csv"

PLOTS_PATH = ASSETS_PATH / "plots"
PREDICTION_PLOT_PATH = PLOTS_PATH / "prediction.png"
INTERVALS_PLOT_PATH = PLOTS_PATH / "intervals.png"
TRAININGLOSS_PLOT_PATH = PLOTS_PATH / "trainingloss.png"

SCALER_PATH = ASSETS_PATH / "scalers"
INPUT_SCALER_PATH = SCALER_PATH / "input_scaler.z"
OUTPUT_SCALER_PATH = SCALER_PATH / "output_scaler.z"
