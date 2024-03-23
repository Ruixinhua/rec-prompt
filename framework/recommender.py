import copy
import json
import os
import re
import ast
import pandas as pd
import numpy as np
import utils.metric_utils as module_metric

from string import Template
from tqdm import tqdm
import template
from utils import convert2list, save2csv, cal_avg_scores, extract_output, evaluate_list, run_llm_model, get_project_root
from utils import load_model_tokenizer, inference_llm_hf, is_descending, setup_logging, load_model


def check_ranking(rank_list, candidates):
    """
    :param rank_list: a list of ranking
    :param candidates: a list of candidates
    :return:
    """
    if not rank_list:
        return rank_list
    new_ranks = []
    candidate_dict = {c.split(": ", 1)[0]: c.split(": ", 1)[1] for c in candidates.split("\n")}
    for rank in rank_list:
        match = re.findall(r"C\d+", rank)
        if match:
            if 2 <= len(str(match[0])) <= 3:
                new_ranks.append(match[0])
        else:
            for i, c in candidate_dict.items():
                c = " ".join(c.split())
                rank = " ".join(rank.split())
                if c in rank:
                    new_ranks.append(i)
                    break
    return new_ranks


def run_llm(prompt_str, **kwargs):
    """
    :param prompt_str: a string indicating the prompt string with filled placeholders
    :param kwargs: a dict with keys: use_guidance, caching, system_instruction, candidate, use_hf
    :return
    """
    candidate = kwargs.get("candidate")
    if kwargs.get("use_hf", False):
        model, tokenizer = kwargs.get("model"), kwargs.get("tokenizer")
        params = {
            "max_length": kwargs.get("max_length"), "max_new_tokens": kwargs.get("max_tokens"),
            "do_sample": kwargs.get("do_sample")
        }
        output = inference_llm_hf(model, tokenizer, prompt_str, **params)
    else:
        model = kwargs.get("model")
        output = run_llm_model(model, prompt_str, **kwargs)
    try:
        find_json = re.findall(r"```json\n(.*?)\n```", output, re.DOTALL)
        if find_json:
            output = find_json[0]
        output_json = {k.lower(): v for k, v in json.loads(output).items()}
        if "ranking" in output_json:
            ranks = output_json.get("ranking")
            if isinstance(ranks, str):
                ranks = extract_output(ranks, candidate, match_pattern=True)
        else:
            ranks = output_json.get(list(output_json.keys())[0])
    except json.JSONDecodeError:
        matches = re.search(r'"ranking"\s*:\s*(\[[^\]]*\])', output)
        if matches:
            ranks = ast.literal_eval(matches.group(1))
        else:
            ranks = extract_output(output, candidate, match_pattern=True)
    ranks = check_ranking(ranks, candidate)
    return output, ranks


def run_recommender(**kwargs):
    """
    :param : template of prompt with placeholders: history and candidate
    """
    prompt_strategy = kwargs.get("prompt_strategy")
    model_args = kwargs.get("model_args")
    tag = f"{model_args['model_name']}--{kwargs.get('tag', 'test')}"
    logger = kwargs.get("logger", setup_logging(log_path=get_project_root() / f"log/rec_test/{tag}.log"))
    if prompt_strategy is None or not len(prompt_strategy):
        logger.error("Please provide a prompt template")
        raise ValueError("Please provide a prompt template")
    else:
        prompt_strategy = getattr(template, prompt_strategy) if hasattr(template, prompt_strategy) else prompt_strategy
    samples_df = kwargs.get("samples_df", pd.read_csv(get_project_root() / f"data/MIND_test.csv"))
    metric_list = ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]
    data_cols = ["impression_id", "history", "candidate", "label"]
    metric_funcs = [getattr(module_metric, met) for met in metric_list]
    max_output_tokens = kwargs.get("max_output_tokens", 2048)
    instruct = kwargs.get("system_instruction", "rankonly_nonjson")
    instruct = getattr(template, instruct) if hasattr(template, instruct) else instruct
    use_hf = kwargs.get("use_hf", False)
    if use_hf:
        model, tokenizer = load_model_tokenizer(model_args['model_name'])
        model_args.update({
            "model": model, "tokenizer": tokenizer, "max_length": kwargs.get("max_length", 4096),
            "do_sample": kwargs.get("do_sample", False)
        })
    generated_output_path = kwargs.get("generated_output_path")
    score_path = kwargs.get("score_path")
    os.makedirs(os.path.dirname(generated_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(score_path), exist_ok=True)
    results = []
    in_order_num = 0
    # use the current time as run_id
    run_id = pd.Timestamp.now().strftime("%Y-%m-%d/%H:%M:%S")
    score_df = pd.DataFrame()
    for index in tqdm(samples_df.index, total=len(samples_df)):
        tmp_params = copy.deepcopy(model_args)
        line = {col: samples_df.loc[index, col] for col in data_cols}
        prompt_str = Template(prompt_strategy).safe_substitute(history=line["history"], candidate=line["candidate"])
        run_kwargs = {
            "system_instruction": instruct, "candidate": line["candidate"], "use_hf": use_hf,
            "model": load_model(**model_args)
        }
        if kwargs.get("add_rank_first"):
            format_str = """You should output the ranked list FIRST and then explain the results."""
            prompt_str = prompt_str.rstrip("\n") + "\n" + format_str
        output, ranks = run_llm(prompt_str, **run_kwargs)

        def check_output(rank, out):
            is_wrong_format = rank is False or (isinstance(rank, list) and len(rank) == 0)
            if instruct == "explanation_json":
                try:
                    output_json = json.loads(out)
                    keys = ["ranking", "topics of the user's interest", "candidates' topics", "candidates relevant to the user's interest"]
                    if all(k in output_json for k in keys):
                        topic_interest = output_json.get("topics of the user's interest")
                        for indices in topic_interest.values():
                            if not isinstance(indices, list):
                                return True
                            for i in indices:
                                if len(i) > 3 or not re.search(r"H\d+", i):
                                    return True
                        candidate_topics = output_json.get("candidates' topics")
                        for i in candidate_topics.keys():
                            if len(i) > 3 or not re.search(r"C\d+", i):
                                return True
                        relevant_indices = output_json.get("candidates relevant to the user's interest")
                        for i in relevant_indices:
                            if len(i) > 3 or not re.search(r"C\d+", i):
                                return True
                    else:
                        return True
                except json.JSONDecodeError:
                    return True
            return is_wrong_format
        while check_output(ranks, output):
            print("Re-Run Model")
            if tmp_params["model_params"]["max_tokens"] * 2 <= max_output_tokens:
                tmp_params["model_params"]["max_tokens"] = tmp_params["model_params"]["max_tokens"] * 2
            # tmp_params["model_params"]["temperature"] += 0.01
            if tmp_params["model_params"]["max_tokens"] * 2 > max_output_tokens:
                tmp_params["model_params"]["temperature"] += 0.1
            # tmp_params["model_params"]["temperature"] = temperature_mapper[tmp_params["model_params"]["max_tokens"]]
            run_kwargs["model"] = load_model(**tmp_params)
            output, ranks, = run_llm(prompt_str, **run_kwargs)
        if tmp_params["model_params"]["max_tokens"] != model_args["model_params"]["max_tokens"]:
            print(f"Length of Input: {len(prompt_str)}: {tmp_params['model_params']['max_tokens']}")
        if ranks is False or (isinstance(ranks, list) and len(ranks) == 0):
            logger.info(f"RUN LLM({model_args['model_name']}): Reach the maximum tokens {max_output_tokens}.")
            continue
        line["rank"] = ','.join(ranks)
        line["full_prompt"] = prompt_str
        line["output"] = output
        output_list, label_list = convert2list(line["rank"], line["label"], line["candidate"])
        in_order_num += 1 if is_descending(output_list[np.nonzero(output_list)[0]]) else 0
        line.update(evaluate_list(output_list, label_list, metric_funcs))
        results.append(line)
        save2csv(results, generated_output_path)
        score_df = cal_avg_scores(results, score_path, model_args['model_name'], metric_list,
                                  in_order_num=in_order_num, saved_params=model_args, run_id=run_id)
    return score_df
