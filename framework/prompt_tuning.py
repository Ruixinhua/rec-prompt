import os
import re
import json
import copy
import pandas as pd

from string import Template
from datetime import datetime
import template
from template import sample_temp, prompt_temp, best_prompt_temp
from utils import setup_logging, seed_everything, load_cmd_line, load_model, run_llm_model, get_project_root
from recommender import run_recommender


def build_sample(history, candidate, answer, click, **kwargs):
    full_prompt = Template(kwargs["prompt_temp"]).safe_substitute(history=history, candidate=candidate)
    return Template(sample_temp).safe_substitute(full_prompt=full_prompt, answer=answer, click=click)


def cal_avg_performance(result):
    return round(result[["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]].mean(axis=1)[0], 2)


def update_monitor_scores(monitor_records, monitor_path, monitor_columns):
    monitor_scores = pd.DataFrame.from_records(monitor_records, columns=monitor_columns)
    if os.path.exists(monitor_path):
        monitor_scores = pd.concat([monitor_scores, pd.read_csv(monitor_path, index_col=False)])
    monitor_scores.drop_duplicates(subset=["epoch", "prompt_strategy", "group_auc"], inplace=True)
    monitor_scores.to_csv(monitor_path, index=False)
    return monitor_scores


def generate_improved_prompt(prompt4opt, model, **kwargs):
    return run_llm_model(model, prompt4opt, **kwargs)


def build_prompt(guide_instruction, initial_prompt, best_prompt, observe_instruction, samples, **kwargs):
    sample_num = kwargs.get('sample_num')
    sample_seed = kwargs.get("sample_seed", 42)
    prompt4opt = guide_instruction + Template(prompt_temp).safe_substitute(prompt_temp=initial_prompt)
    for line in samples.sample(n=sample_num, random_state=sample_seed).to_dict(orient="records"):
        prompt4opt += build_sample(prompt_temp=initial_prompt, history=line["history"], answer=line["output"],
                                   candidate=line["candidate"], click=line["label"])
    prompt4opt += Template(best_prompt_temp).safe_substitute(best_prompt_temp=best_prompt)
    prompt4opt += observe_instruction
    return prompt4opt


def tuning_prompt(**kwargs):
    recommender = kwargs.get("recommender", "gpt-3.5-turbo-1106")
    optimizer = kwargs.get("optimizer", "gpt-3.5-turbo-1106")
    sample_num = kwargs.get('sample_num', 1)  # the number of samples used for optimizer
    valid_samples = kwargs.get("samples_df", pd.read_csv(get_project_root() / "data/MIND_valid.csv"))
    valid_samples = valid_samples.sample(n=kwargs.get("valid_sample_num", 100), random_state=42)
    tag = kwargs.get("tag", f"{recommender}--{optimizer}--sample_num-{sample_num}")
    logger = setup_logging(log_path=get_project_root() / f"log/prompt_tuning/{tag}.log")
    logger.info(f"Initialization -- Recommender: {recommender} -- Optimizer: {optimizer}.")
    initial_prompt = getattr(template, kwargs.get("initial_prompt", "initial_prompt"))
    generated_output_path = get_project_root() / f"output/generated_data/prompt_tuning/{tag}/epoch_0.csv"
    generated_output_path.parent.mkdir(parents=True, exist_ok=True)
    score_path = get_project_root() / f"output/result/prompt_tuning/{tag}/epoch_0.csv"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    recommender_args = kwargs.get("recommender_args", kwargs)
    if "model_params" not in recommender_args:
        recommender_args["model_name"] = recommender
        recommender_args["model_params"] = {
            "temperature": 0, "max_tokens": 50, "max_retries": 5}
        recommender_args["extra_params"] = {}
    recommender_params = copy.deepcopy(kwargs)
    recommender_params.update({
        "model_args": recommender_args, "samples_df": valid_samples, "score_path": score_path,
        "prompt_strategy": initial_prompt, "generated_output_path": generated_output_path
    })
    initial_result = run_recommender(**recommender_params)
    avg_score = cal_avg_performance(initial_result)
    optimizer_args = kwargs.get("optimizer_args", kwargs)
    if "model_params" not in optimizer_args:
        optimizer_args["model_name"] = optimizer
        optimizer_args["model_params"] = {
            "temperature": kwargs.get("temperature", 0), "max_tokens": 1024,
            "max_retries": kwargs.get("max_retries", 5)}
    prompt_optimizer = load_model(**optimizer_args)
    initial_score = initial_result[["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]].loc[0].tolist()
    record = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, initial_prompt] + initial_score + [avg_score]
    print(f"Epoch-0: Current Score is {initial_score}")
    monitor_columns = ["id", "epoch", "prompt_strategy", "group_auc", "mean_mrr", "ndcg_5", "ndcg_10", "avg_score"]
    monitor_records = [record]
    rec_instruct = kwargs.get("system_instruction", "rankonly_nonjson")
    rec_instruct = getattr(template, rec_instruct) if hasattr(template, rec_instruct) else rec_instruct
    guide_instruction = getattr(template, kwargs.get("guide_instruction", "guide_instruction_naive"))
    observe_instruction = getattr(template, kwargs.get("observation_instruction", "observation_instruction_naive"))
    meta_instr = kwargs.get("meta_instruction", "meta_instruction")
    instruct = getattr(template, meta_instr) if hasattr(template, meta_instr) else meta_instr
    run_kwargs = {"system_instruction": instruct}
    monitor_path = get_project_root() / f"output/result/prompt_tuning/{tag}/monitor_scores.csv"
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    assistant_path = get_project_root() / f"output/result/prompt_tuning/{tag}/assistant.json"
    assistant_path.parent.mkdir(parents=True, exist_ok=True)
    instruction_path = get_project_root() / f"output/result/prompt_tuning/{tag}/instruction.json"
    instruction_path.parent.mkdir(parents=True, exist_ok=True)
    instructions = {
        "initial_prompt": initial_prompt,
        "rec_system_instruction": rec_instruct,
        "guide_instruction": guide_instruction,
        "observe_instruction": observe_instruction,
        "tuning_system_instruction": instruct,
    }
    with open(instruction_path, "w") as f:
        json.dump(instructions, f, indent=4)
    monitor_scores = update_monitor_scores(monitor_records, monitor_path, monitor_columns)
    best_scores = {"epoch": 0, "best_score": avg_score, "best_prompt_temp": initial_prompt}
    assistant_prompts = {}
    for epoch in range(1, kwargs.get("epochs", 5)+1):
        sample_seed = kwargs.get("sample_seed", 42)
        optimizer_params = copy.deepcopy(optimizer_args)
        generated_output_path = get_project_root() / f"output/generated_data/prompt_tuning/{tag}/epoch_{epoch-1}.csv"
        samples = kwargs.get("samples", pd.read_csv(generated_output_path))
        prompt_params = {
            "guide_instruction": guide_instruction, "initial_prompt": initial_prompt, "best_prompt": initial_prompt,
            "observe_instruction": observe_instruction, "samples": samples, "sample_num": sample_num
        }
        prompt4opt = build_prompt(**prompt_params)
        go_on = True
        current_prompt = generate_improved_prompt(prompt4opt, prompt_optimizer, **run_kwargs)
        assistant_prompts[f"epoch-{epoch}"] = {
            "optimizer prompt": prompt4opt, "assistant answer": current_prompt,
        }
        while go_on:
            with open(assistant_path, "w") as f:
                json.dump(assistant_prompts, f, indent=4)
            match = re.search(r"# Prompt Template Begin\n(.*?)# Prompt Template End", current_prompt, re.DOTALL)
            if match is not None:
                current_prompt = match.group(1)
                is_exist = False
                for epoch_no, prompts in assistant_prompts.items():
                    if "prompt template" in prompts and current_prompt == prompts["prompt template"]:
                        is_exist = True
                        break
                if not is_exist and "${history}" in current_prompt and "${candidate}" in current_prompt:
                    go_on = False
            else:
                optimizer_params['model_params']["max_tokens"] += 1024
            if go_on:
                sample_seed += 1
                prompt4opt = build_prompt(sample_seed=sample_seed, **prompt_params)
                prompt_optimizer = load_model(**optimizer_params)
                current_prompt = generate_improved_prompt(prompt4opt, prompt_optimizer, **run_kwargs)
                assistant_prompts[f"epoch-{epoch}"]["assistant answer"] = current_prompt
        logger.info(f"Epoch-{epoch}: Start to run recommender with the improved prompt.")
        assistant_prompts[f"epoch-{epoch}"]["prompt template"] = current_prompt
        with open(assistant_path, "w") as f:
            json.dump(assistant_prompts, f, indent=4)
        generated_output_path = get_project_root() / f"output/generated_data/prompt_tuning/{tag}/epoch_{epoch}.csv"
        score_path = get_project_root() / f"output/result/prompt_tuning/{tag}/epoch_{epoch}.csv"
        recommender_params.update({
            "prompt_strategy": current_prompt, "generated_output_path": generated_output_path, "score_path": score_path
        })
        if kwargs.get("tuning_max_tokens"):
            recommender_params["model_args"]["model_params"]["max_tokens"] = kwargs.get("tuning_max_tokens")
        current_result = run_recommender(**recommender_params)
        avg_score = cal_avg_performance(current_result)
        current_score = current_result[["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]].loc[0].tolist()
        record = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, current_prompt] + current_score + [avg_score]
        monitor_records.append(record)
        monitor_scores = update_monitor_scores(monitor_records, monitor_path, monitor_columns)
        print(f"Epoch-{epoch}: Current Score is {current_score}")
        print(current_prompt)
        if avg_score > best_scores["best_score"]:
            best_scores["epoch"] = epoch
            best_scores["best_score"] = avg_score
            best_scores["best_prompt_temp"] = current_prompt
            with open(get_project_root() / f"output/result/prompt_tuning/{tag}/best_scores.json", "w") as f:
                json.dump(best_scores, f, indent=4)
    return monitor_scores, best_scores
