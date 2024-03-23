import json
import re

import pandas as pd
from utils import load_cmd_line, to_format_excel
from pathlib import Path


def evaluate_user_correctness(path, mode, **kwargs):
    correct_df = pd.read_excel(path)
    correct_df["correct_json"] = correct_df[mode].apply(json.loads)
    correct = 0
    complete = 0
    total = 0
    history_total = 0
    for index, row in correct_df.iterrows():
        topics_user = [t for t in row["topics_user"].split("\n") if len(t) > 0]
        correct_json = row["correct_json"]["correctness"]
        total += len(topics_user)
        history = row["history"].split("\n")
        history_num = len(row["history"].split("\n"))
        history_total += history_num
        history_set = []
        for topic in topics_user:
            hs = correct_json[topic] if topic in correct_json else []  # "H1", "H2", "H3"
            for h in hs:
                hf = re.findall(r"H\d+", h)
                if h and len(hf) == 0:
                    for i, hi in enumerate(history):
                        if h in hi:
                            hf.append(f"H{i+1}")
                    if len(hf) == 0:
                        print(h)
                history_set.extend(hf)
            correct += 1 if len(hs) > 0 else 0
        history_set = set(history_set)
        num = 0
        for i in range(1, history_num + 1):
            if f"H{i}" in history_set:
                num += 1
        complete += num
    precision = f"{(100*correct / total):.2f}%"
    values = [
        {"model": kwargs.get('model', 'gpt-3.5'), "evaluator": kwargs.get('evaluator', 'gpt-3.5'), "metric": "user-correctness", "value(%)": precision, "value(#)": f"{correct}/{total}"},
        {"model": kwargs.get('model', 'gpt-3.5'), "evaluator": kwargs.get('evaluator', 'gpt-3.5'), "metric": "user-completeness", "value(%)": f"{(100*complete / history_total):.2f}%", "value(#)": f"{complete}/{history_total}"}
    ]
    print(values)
    return values


def evaluate_can_correctness(path, mode, **kwargs):
    correct_df = pd.read_excel(path)
    correct_df["correct_json"] = correct_df[mode].apply(json.loads)
    correct_df["topics_can"] = correct_df.topics_can.apply(json.loads)
    correct = 0
    total = 0
    for index, row in correct_df.iterrows():
        topics_can = row["topics_can"]
        correct_json = row["correct_json"]["correctness"]
        for no, topics in topics_can.items():
            total += len(topics)
            correct += sum([correct_json[no][topic] for topic in topics
                            if no in correct_json and topic in correct_json[no]])
    precision = f"{(100*correct / total):.2f}%"
    values = [
        {"model": kwargs.get('model', 'gpt-3.5'), "evaluator": kwargs.get('evaluator', 'gpt-3.5'), "metric": "candidate-correctness", "value(%)": precision, "value(#)": f"{correct}/{total}"},
    ]
    print(values)
    return values


def evaluate_reasonability(path, mode, **kwargs):
    correct_df = pd.read_excel(path)
    correct_df["reason_json"] = correct_df[mode].apply(json.loads)
    pos_correct, pos_total, neg_correct, neg_total = 0, 0, 0, 0
    for index, row in correct_df.iterrows():
        reason_json = row["reason_json"]
        label = row["label"]
        for i, value in reason_json.items():
            value = int(list(json.loads(reason_json[i]).values())[0])
            if i != label:
                neg_total += 1
                neg_correct += 1 if value == 0 else 0
            else:
                pos_total += 1
                pos_correct += 1 if value == 1 else 0
    values = [
        {"model": kwargs.get('model', 'gpt-3.5'), "evaluator": kwargs.get('evaluator', 'gpt-3.5'), "metric": "positive-reasonability", "value(%)": f"{(100*pos_correct / pos_total):.2f}%", "value(#)": f"{pos_correct}/{pos_total}"},
        {"model": kwargs.get('model', 'gpt-3.5'), "evaluator": kwargs.get('evaluator', 'gpt-3.5'), "metric": "negative-reasonability", "value(%)": f"{(100*neg_correct / neg_total):.2f}%", "value(#)": f"{neg_correct}/{neg_total}"}
    ]
    print(values)
    return values


if __name__ == "__main__":
    args = load_cmd_line()
    evaluate_mode = args.get("evaluate_mode", "user_correctness")
    root_dir = args.get("root_dir", "explanation_100")
    evaluation_file = args.get("evaluation_file", "gpt3-1106_topics_gpt3-1106_user_correctness.xlsx")
    stat_file = args.get("stat_file", "overall_stat.xlsx")
    stat_path = Path(f"stat/{stat_file}")
    stat_path.parent.mkdir(exist_ok=True, parents=True)
    model, evaluator = evaluation_file.split("_")[0], evaluation_file.split("_")[2]
    if stat_path.exists():
        stat_df = pd.read_excel(stat_path)
    else:
        stat_df = pd.DataFrame(columns=["model", "evaluator", "metric", "value(%)", "value(#)"])
    correctness_path = f"{root_dir}/{evaluate_mode}/{evaluation_file}"
    if evaluate_mode == "user_correctness":
        eval_func = evaluate_user_correctness
    elif evaluate_mode == "can_correctness":
        eval_func = evaluate_can_correctness
    elif evaluate_mode == "reasonability":
        eval_func = evaluate_reasonability
    else:
        raise ValueError("Please specify the correct evaluation mode.")
    for v_d in eval_func(correctness_path, evaluate_mode, model=model, evaluator=evaluator):
        stat_df = pd.concat([stat_df, pd.DataFrame([v_d])])
    column_widths = {
        "A:E": 20,
    }
    # remove duplicate rows of stat_df
    stat_df = stat_df.drop_duplicates(subset=["model", "evaluator", "metric", "value(%)", "value(#)"], keep="last")
    to_format_excel(stat_df, stat_path, column_widths=column_widths, cell_height=20)
