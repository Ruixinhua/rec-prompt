import copy
import json

import pandas as pd
from collections import defaultdict
from utils import load_cmd_line, to_format_excel, get_project_root


def convert_format(name):
    return name.replace("-", " ").replace("_", " ").lower()


def compute_score(**kwargs):
    recommender = kwargs.get("recommender", "gpt-3.5-turbo-1106")
    strategy = kwargs.get("strategy", "cot_prompt")
    annotation_path = kwargs.get("annotation_path", "explanation_100/annotation/news_topics_machine.xlsx")
    evaluator = kwargs.get("evaluator", "gpt-3.5-turbo-1106")
    stat_file = kwargs.get("stat_file", "final_scores.csv")
    stat_path = get_project_root() / f"output/stat/{stat_file}"
    stat_path.parent.mkdir(exist_ok=True, parents=True)
    if stat_path.exists():
        stat_df = pd.read_csv(stat_path)
    else:
        stat_df = pd.DataFrame()
    annotation_df = pd.read_excel(annotation_path)
    annotation_df = annotation_df[annotation_df.annotator == evaluator]
    annotation_df["annotation"] = annotation_df.annotation.apply(json.loads)
    news_topics_dict = {
        row.news_headline: [t.split("--")[0] for t, c in row.annotation.items() if c] for i, row in annotation_df.iterrows()
    }
    topic_mapper_path = get_project_root() / "data/news_topics/topics_mapper_format.json"
    topic_mapper = json.load(open(topic_mapper_path))
    news_topics_mapped = defaultdict(set)

    for news, topics in news_topics_dict.items():
        for topic in set(topics):
            topic = convert_format(topic)
            if topic not in topic_mapper:
                # print(f"Topic Mapper: {topic}")
                continue
            for t in topic_mapper[topic]:
                news_topics_mapped[news].add(t)
    topics_path = get_project_root() / f"output/explain_case/{recommender}--{strategy}--explanation_json--case_cat.csv"
    topics_df = pd.read_csv(topics_path)
    stat_scores = defaultdict(int)
    topics_df["output_json"] = topics_df.output.apply(json.loads)

    for i, row in topics_df.iterrows():
        history = [h.split(" Category=>", 1)[0].split(": ", 1)[1] for h in row.history.split("\n")]
        stat_scores["history_total"] += len(history)
        output_json = row.output_json
        topic_interest = output_json.get("topics of the user's interest")
        relevant_indices = output_json.get("candidates relevant to the user's interest")

        def create_final_topics(ts):
            ts_dict = defaultdict(list)
            final_topics = []
            for t in ts:
                if ";" in t:
                    ts_dict[t].extend(t.split(";"))
                if " " in t:
                    ts_dict[t].append(t.split(" "))
                ts_dict[t].append(t)
            for t, t_list in ts_dict.items():
                t = convert_format(t)
                if t in topic_mapper:
                    final_topics.extend(topic_mapper[t])
                else:
                    for ts in t_list:
                        if isinstance(ts, list):
                            for tt in ts:
                                tt = convert_format(tt)
                                if tt in topic_mapper:
                                    final_topics.extend(topic_mapper[tt])
                        else:
                            ts = convert_format(ts)
                            if ts in topic_mapper:
                                final_topics.extend(topic_mapper[ts])
            return set([convert_format(t) for t in final_topics])
        topics_user = create_final_topics(topic_interest.keys())
        stat_scores["topics_total"] += len(topics_user)
        history_topics_all = []
        for news in history:
            if news not in news_topics_mapped:
                # print(f"History News: {news}")
                continue
            topics = news_topics_mapped[news]
            for topic in topics:
                if topic in topics_user:
                    stat_scores["Completeness"] += 1
                    break
            history_topics_all.extend(topics)
        history_topics_all = set([convert_format(t) for t in history_topics_all])
        stat_scores["Correctness"] += len(history_topics_all & topics_user)
        candidate = {c.split(" Category=>", 1)[0].split(": ", 1)[0]: c.split(" Category=>", 1)[0].split(": ", 1)[1]
                     for c in row.candidate.split("\n")}
        candidate_topics = output_json.get("candidates' topics")
        for can, topics in candidate_topics.items():
            if candidate[can] not in news_topics_mapped:
                # if candidate[can] not in news_topics_mapped and len(topics):
                    # print(f"Candidate News: {candidate[can]}")
                continue
            can_topics = create_final_topics(topics)
            stat_scores["topics_total"] += len(can_topics)
            for topic in can_topics:
                topics = [convert_format(t) for t in news_topics_mapped[candidate[can]]]
                if topic in topics:
                    stat_scores["Correctness"] += 1
                    break
    metric_group = [("Correctness", "topics_total"),
                    ("Completeness", "history_total")]
    for group in metric_group:
        metric, total = group
        precision = f"{(100 * stat_scores[metric] / stat_scores[total]):.2f}%"
        print(f"{kwargs.get('tag')} Metric {metric}: {precision}")
        # print(f"{metric}: {precision}")
        rec = kwargs.get("tag").split("--")[0]
        opt = kwargs.get("tag").split("--")[1]
        values = [
            {"Model": rec, "Evaluator": opt, "metric": metric, "value(%)": precision,
             # "value(#)": f"{stat_scores[metric]}/{stat_scores[total]}"
            },
        ]
        stat_df = pd.concat([stat_df, pd.DataFrame(values, index=None)], ignore_index=True)
    column_widths = {
        "A:E": 20,
    }
    # remove duplicate rows of stat_df
    stat_df = stat_df.drop_duplicates(subset=["Model", "Evaluator", "metric", "value(%)"], keep="last")
    stat_df.to_csv(stat_path, index=False, header=True, encoding="utf-8")
    # to_format_excel(stat_df, stat_path, column_widths=column_widths, cell_height=20, case_num=len(stat_df))


if __name__ == "__main__":
    args = load_cmd_line()
    # strategy = args.get("strategy", "rec_cot")
    strategy = args.get("strategy", "cot_prompt")

    args_list = [
        {
            "tag": "CoT-3.5--Eval-3.5", "strategy": "cot_prompt",
            "recommender": "gpt-3.5-turbo-1106", "evaluator": "gpt-3.5-turbo-1106",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },
        {
            "tag": "CoT-3.5--Eval-4", "strategy": "cot_prompt",
            "recommender": "gpt-3.5-turbo-1106", "evaluator": "gpt-4-1106-preview",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },
        {
            "tag": "CoT-4--Eval-3.5", "strategy": "cot_prompt",
            "recommender": "gpt-4-1106-preview", "evaluator": "gpt-3.5-turbo-1106",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },

        {
            "tag": "CoT-3.5--Human", "strategy": "cot_prompt",
            "recommender": "gpt-3.5-turbo-1106", "evaluator": "rui98",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },
        {
            "tag": "CoT-4--Human", "strategy": "cot_prompt",
            "recommender": "gpt-4-1106-preview", "evaluator": "rui98",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },

        {
            "tag": "CoT-4--Eval-4", "strategy": "cot_prompt",
            "recommender": "gpt-4-1106-preview", "evaluator": "gpt-4-1106-preview",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },

        {
            "tag": "Rec-3.5--Eval-3.5", "strategy": "rec_cot",
            "recommender": "gpt-3.5-turbo-1106", "evaluator": "gpt-3.5-turbo-1106",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },
        {
            "tag": "Rec-3.5--Eval-4", "strategy": "rec_cot",
            "recommender": "gpt-3.5-turbo-1106", "evaluator": "gpt-4-1106-preview",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },
        {
            "tag": "Rec-4--Eval-3.5", "strategy": "rec_cot",
            "recommender": "gpt-4-1106-preview", "evaluator": "gpt-3.5-turbo-1106",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },
        {
            "tag": "Rec-4--Eval-4", "strategy": "rec_cot",
            "recommender": "gpt-4-1106-preview", "evaluator": "gpt-4-1106-preview",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },
        {
            "tag": "Rec-3.5--Human", "strategy": "rec_cot",
            "recommender": "gpt-3.5-turbo-1106", "evaluator": "rui98",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },
        {
            "tag": "Rec-4--Human", "strategy": "rec_cot",
            "recommender": "gpt-4-1106-preview", "evaluator": "rui98",
            "annotation_path": get_project_root() / "data/explain_eval/annotation/news_topics_all.xlsx",
        },
    ]
    for params in args_list:
        compute_score(**params)
