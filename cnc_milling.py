from continual import create_base_model, continue_from_model, move_param_file


if __name__ == "__main__":
    move_param_file("cnc_tool_wear.yaml")
    
    # 20% replay
    create_base_model("cnc_milling_tool_wear_20_replay", "cnc_milling_with_toolwear_first", 0.2)
    continue_from_model("cnc_milling_tool_wear_20_replay", "cnc_milling_with_toolwear_second", 0.2)
    continue_from_model("cnc_milling_tool_wear_20_replay", "cnc_milling_with_toolwear_third", 0.2)
    continue_from_model("cnc_milling_tool_wear_20_replay", "cnc_milling_with_toolwear_fourth", 0.2)
    
    # 60% replay
    create_base_model("cnc_milling_tool_wear_60_replay", "cnc_milling_with_toolwear_first", 0.6)
    continue_from_model("cnc_milling_tool_wear_60_replay", "cnc_milling_with_toolwear_second", 0.6)
    continue_from_model("cnc_milling_tool_wear_60_replay", "cnc_milling_with_toolwear_third", 0.6)
    continue_from_model("cnc_milling_tool_wear_60_replay", "cnc_milling_with_toolwear_fourth", 0.6)
    
    # 00% replay
    create_base_model("cnc_milling_tool_wear_00_replay", "cnc_milling_with_toolwear_first", 0)
    continue_from_model("cnc_milling_tool_wear_00_replay", "cnc_milling_with_toolwear_second", 0)
    continue_from_model("cnc_milling_tool_wear_00_replay", "cnc_milling_with_toolwear_third", 0)
    continue_from_model("cnc_milling_tool_wear_00_replay", "cnc_milling_with_toolwear_fourth", 0)