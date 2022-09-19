from continual import create_base_model, continue_from_model, move_param_file


if __name__ == "__main__":
    move_param_file("broaching_twd.yaml")
    
    create_base_model("broaching_twd", "broaching_twd_1X", 0.2)
    continue_from_model("broaching_twd", "broaching_twd_2X", 0.2)
    continue_from_model("broaching_twd", "broaching_twd_3X", 0.2)
    continue_from_model("broaching_twd", "broaching_twd_4X", 0.2)
    continue_from_model("broaching_twd", "broaching_twd_5X", 0.2)