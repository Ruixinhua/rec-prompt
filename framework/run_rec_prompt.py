import pandas as pd

import config
import copy
from utils import seed_everything, get_project_root, load_cmd_line
from prompt_tuning import tuning_prompt


def load_rec_prompt_args(**kwargs):
    optimizer_json = copy.deepcopy(getattr(config, kwargs.get("optimizer")))
    data_root = kwargs.get("data_root", get_project_root() / f"data")
    dataset = kwargs.get("dataset", "valid")
    data_path = data_root / f"MIND_{dataset}.csv"
    samples_df = pd.read_csv(data_path).sample(kwargs.get("valid_sample_num", 100))
    tuning_args = {
        "initial_prompt": kwargs.get("initial_prompt"), "system_instruction": kwargs.get("system_instruction"),
        "observation_instruction": kwargs.get("observation_instruction"), "samples_df": samples_df,
        "guide_instruction": kwargs.get("guide_instruction"), "add_rank_first": kwargs.get("add_rank_first"),
        "epochs": kwargs.get("epoch", 3), "sample_num": kwargs.get("sample_num", 1),
        "database_path": get_project_root() / "database/.langchain.db",
        "tuning_max_tokens": kwargs.get("tuning_max_tokens")
    }
    recommender_json = copy.deepcopy(getattr(config, kwargs.get("recommender")))
    rec_max_tokens = kwargs.get("rec_max_tokens", 50)
    recommender_json["model_params"]["max_tokens"] = rec_max_tokens
    if kwargs.get("return_json_rec"):
        recommender_json["extra_params"] = {"response_format": {"type": "json_object"}}
    tuning_args["tag"] = f"""{kwargs.get("name")}--{kwargs.get('system_instruction')}--{kwargs.get('recommender')}--{kwargs.get('optimizer')}--{tuning_args.get('observation_instruction')}--{tuning_args.get('guide_instruction')}--sample_num-{tuning_args.get('sample_num')}"""
    tuning_args.update(kwargs)
    tuning_args.update({"recommender_args": recommender_json, "optimizer_args": optimizer_json})
    return tuning_args


if __name__ == "__main__":
    seed_everything(42)
    tuning_args_list = {
        "3tune3_io": {
            "recommender": "gpt3_1106", "optimizer": "gpt3_1106", "rec_max_tokens": 50, "tuning_max_tokens": 1600,
            "initial_prompt": "io_prompt", "system_instruction": "pure_nonjson",
            "observation_instruction": "observation_instruction_3tune3_io", "valid_sample_num": 100,
            "guide_instruction": "guide_instruction_3tune3_io", "epoch": 10, "sample_num": 1},
        "4tune3_io": {
            "recommender": "gpt3_1106", "optimizer": "gpt4_1106", "rec_max_tokens": 50, "tuning_max_tokens": 1600,
            "initial_prompt": "io_prompt", "system_instruction": "pure_nonjson", "add_rank_first": False,
            "observation_instruction": "observation_instruction_4tune3_io", "valid_sample_num": 100,
            "guide_instruction": "guide_instruction_4tune3_io", "epoch": 10, "sample_num": 1},
        "4tune3_io_cat": {
            "recommender": "gpt3_1106", "optimizer": "gpt4_1106", "rec_max_tokens": 1600, "tuning_max_tokens": 1600,
            "initial_prompt": "io_prompt", "system_instruction": "pure_nonjson", "dataset": "valid_cat",
            "observation_instruction": "observation_instruction_4tune3_io", "valid_sample_num": 100,
            "guide_instruction": "guide_instruction_4tune3_io", "epoch": 10, "sample_num": 1},
        "3tune3_cot": {
            "recommender": "gpt3_1106", "optimizer": "gpt3_1106", "rec_max_tokens": 50, "tuning_max_tokens": 1600,
            "initial_prompt": "cot_prompt", "system_instruction": "pure_nonjson", "add_rank_first": False,
            "observation_instruction": "observation_instruction_3tune3_cot", "valid_sample_num": 100,
            "guide_instruction": "guide_instruction_3tune3_cot", "epoch": 10, "sample_num": 1},
        "4tune3_cot": {
            "recommender": "gpt3_1106", "optimizer": "gpt4_1106", "rec_max_tokens": 50, "tuning_max_tokens": 1600,
            "initial_prompt": "cot_prompt", "system_instruction": "pure_nonjson",
            "observation_instruction": "observation_instruction_4tune3_cot", "valid_sample_num": 100,
            "guide_instruction": "guide_instruction_4tune3_cot", "epoch": 10, "sample_num": 1},
        "4tune3_cot_cat": {
            "recommender": "gpt3_1106", "optimizer": "gpt4_1106", "rec_max_tokens": 1600, "tuning_max_tokens": 1600,
            "initial_prompt": "cot_prompt", "system_instruction": "pure_nonjson", "dataset": "valid_cat",
            "observation_instruction": "observation_instruction_4tune3_cot", "valid_sample_num": 100,
            "guide_instruction": "guide_instruction_4tune3_cot", "epoch": 10, "sample_num": 1},
        "4tune4_io": {
            "recommender": "gpt4_1106", "optimizer": "gpt4_1106", "rec_max_tokens": 50, "return_json_rec": True,
            "initial_prompt": "io_prompt", "system_instruction": "rankonly_json_format", "add_rank_first": False,
            "observation_instruction": "observation_instruction_4tune4_io", "valid_sample_num": 10,
            "guide_instruction": "guide_instruction_4tune4_io", "epoch": 1, "sample_num": 1},
        "4tune4_cot": {
            "recommender": "gpt4_1106", "optimizer": "gpt4_1106", "rec_max_tokens": 50, "return_json_rec": True,
            "initial_prompt": "cot_pure_prompt", "system_instruction": "rankonly_json_format",
            "observation_instruction": "observation_instruction_4tune4_cot", "valid_sample_num": 10,
            "guide_instruction": "guide_instruction_4tune4_cot", "epoch": 1, "sample_num": 1},
        "4tune4_cot_cat": {
            "recommender": "gpt4_1106", "optimizer": "gpt4_1106", "rec_max_tokens": 50, "return_json_rec": True,
            "initial_prompt": "cot_pure_prompt", "system_instruction": "rankonly_json_format", "dataset": "test_cat",
            "observation_instruction": "observation_instruction_4tune4_cot", "valid_sample_num": 100,
            "guide_instruction": "guide_instruction_4tune4_cot", "epoch": 3, "sample_num": 1, "add_rank_first": False},
    }
    args = load_cmd_line()
    tuning_list = args.get("tuning_list")
    for name in tuning_list:
        tuning_args = load_rec_prompt_args(name=name, **tuning_args_list[name])
        print(f"Start running {tuning_args['tag']}...")
        tuning_prompt(**tuning_args)
