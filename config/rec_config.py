# setup system_instruction, data_used,
run_gpt3_1106_args = {
    # General arguments
    "prompt_strategy": "io_prompt",
    "system_instruction": "rankonly_nonjson",
    "tag": None,
    "data_path": "data/MIND_valid.csv",
    "run_num": 100,
    # Model arguments
    "model_args": {
        "model_name": "gpt-3.5-turbo-1106",
        "model_family": "openai",
        "extra_params": {},
        "model_params": {
            "temperature": 0,
            "max_tokens": 50,
            "max_retries": 5
        }
    }
}

run_gpt4_1106_args = {
    # General arguments
    "prompt_strategy": "io_prompt",
    "system_instruction": "rankonly_nonjson",
    "tag": None,
    "data_path": "data/MIND_valid.csv",
    "run_num": 100,
    # Model arguments
    "model_args": {
        "model_name": "gpt-4-1106-preview",
        "model_family": "openai",
        "extra_params": {},
        "model_params": {
            "temperature": 0,
            "max_tokens": 50,
            "max_retries": 5
        }
    }
}

run_gpt4_0613_args = {
    # General arguments
    "prompt_strategy": "io_prompt",
    "system_instruction": "rankonly_nonjson",
    "tag": None,
    "data_path": "data/MIND_valid.csv",
    "run_num": 100,
    # Model arguments
    "model_args": {
        "model_name": "gpt-4-0613",
        "model_family": "openai",
        "extra_params": {},
        "model_params": {
            "temperature": 0,
            "max_tokens": 50,
            "max_retries": 5
        }
    }
}

temperature_mapper = {
    50: 0, 100: 0.01, 200: 0.02, 400: 0.03, 800: 0.04, 1600: 0.05, 3200: 0.06
}
