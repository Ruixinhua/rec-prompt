import pandas as pd
import config

from utils import load_cmd_line, seed_everything, get_project_root, setup_logging
from recommender import run_recommender


def load_args(run_args, **kwargs):
    data_root = kwargs.get("data_root", get_project_root() / f"data")
    data_used = kwargs.get("dataset", "valid")
    data_path = data_root / f"MIND_{data_used}.csv"
    run_num = kwargs.get("run_num")
    if "valid" in data_used and run_num is None:
        run_num = 100
    elif "test" in data_used and run_num is None:
        run_num = 400
    system_instruction = kwargs.get("system_instruction")
    max_tokens = kwargs.get("max_tokens", run_args["model_args"]["model_params"]["max_tokens"])
    if isinstance(max_tokens, str) and max_tokens.lower() == "none":
        max_tokens = None
    return_json = kwargs.get("return_json")
    if "nonjson" in system_instruction and return_json is None:
        run_args["model_args"]["extra_params"] = {}
    elif "json" in system_instruction and return_json is None:
        run_args["model_args"]["extra_params"] = {"response_format": {"type": "json_object"}}
    if return_json:
        run_args["model_args"]["extra_params"] = {"response_format": {"type": "json_object"}}
    else:
        run_args["model_args"]["extra_params"] = {}
    run_args["model_args"]["model_params"]["max_tokens"] = max_tokens
    prompt_strategy = kwargs.get("prompt_strategy", run_args["prompt_strategy"])
    tag = kwargs.get("tag", f"{run_args['model_args']['model_name']}--{prompt_strategy}--{system_instruction}--{data_used}")
    if kwargs.get("extra_tag"):
        tag += f"--{kwargs['extra_tag']}"
    run_args.update({
        "data_path": data_path, "system_instruction": system_instruction, "tag": tag, "run_num": run_num,
        "prompt_strategy": prompt_strategy,
        "generated_output_path": get_project_root() / f"output/generated_data/{data_used}/{tag}.csv",
        "score_path": kwargs.get("score_path", get_project_root() / f"output/result/{data_used}/{tag}.csv"),
    })
    run_args["generated_output_path"].parent.mkdir(parents=True, exist_ok=True)
    run_args["score_path"].parent.mkdir(parents=True, exist_ok=True)
    run_args["logger"] = setup_logging(log_path=get_project_root() / f"log/{data_used}/{tag}.log")
    run_args["samples_df"] = pd.read_csv(run_args["data_path"]).sample(run_args["run_num"])
    return run_args


if __name__ == "__main__":
    args = load_cmd_line()
    seed_everything(args.get("seed", 42))
    run_args_list = args.get("run_args_list", [])
    for run_args_name in run_args_list:
        if not hasattr(config, run_args_name):
            raise ValueError(f"Invalid running arguments name: {run_args_name}")
        arguments = getattr(config, run_args_name)
        arguments = load_args(arguments, **args["cmd_args"])
        score = run_recommender(**arguments)
        logger = setup_logging(log_path=get_project_root() / f"log/result.log")
        data_tag = "w/o category"
        if "cat" in args.get("dataset"):
            data_tag = "with category"
        # print the current score in the following format: prompt_strategy "group_auc", "mean_mrr", "ndcg_5", "ndcg_10"
        logger.info(f"{arguments['prompt_strategy']}--{data_tag} AUC: {score['group_auc']}, MRR: {score['mean_mrr']}, nDCG@5: {score['ndcg_5']}, nDCG@10: {score['ndcg_10']}")
        # print(f"Save the score to the path: {arguments['score_path']}")
