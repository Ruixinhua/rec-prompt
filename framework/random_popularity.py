import random
import os
import pandas as pd

from tqdm import tqdm
from collections import defaultdict
from utils import metric_utils as module_metric
from utils import convert2list, evaluate_list, save2csv, cal_avg_scores, seed_everything, load_cmd_line


def load_all_data(mind_type, root_dir=None):
    if root_dir is None:
        root_dir = rf"C:\Users\Rui\Documents\Explainable_AI\ExplainableNRS\dataset\MIND\{mind_type}"
    test_df = pd.read_csv(f"{root_dir}/test/behaviors.tsv", sep="\t", header=None,
                          names=["impression_id", "user_id", "time", "history", "label"])
    test_df["subset"] = "test"
    train_df = pd.read_csv(f"{root_dir}/train/behaviors.tsv", sep="\t", header=None,
                           names=["impression_id", "user_id", "time", "history", "label"])
    valid_df = pd.read_csv(f"{root_dir}/valid/behaviors.tsv", sep="\t", header=None,
                           names=["impression_id", "user_id", "time", "history", "label"])
    news_df = pd.read_csv(f"{root_dir}/news.csv")
    news_topic = dict(zip(news_df.news_id.tolist(), news_df.category.tolist()))
    all_df = pd.concat([train_df, valid_df, test_df])
    return all_df, news_topic


def cal_count(all_df):
    count = defaultdict(int)
    # stat the popularity of history clicked news articles
    for h in all_df.history:
        if type(h) == float:
            continue
        for n in h.split(" "):
            count[n] += 1
    for h in all_df.label:
        if type(h) == float:
            continue
        for n in h.split(" "):
            count[n.split("-")[0]] += 1
    return count


def apply_popularity(x):
    return ",".join([f"C{item + 1}" for item in sorted(range(len(x)), key=lambda i: x[i], reverse=True)])


def generate_most_pop_predictions(samples, mn="MostPop", **kwargs):
    all_df, news_topic = load_all_data("small")
    all_df = all_df.sample(n=kwargs.get("sample_num", 200000), random_state=kwargs.get("seed", 42))
    count = cal_count(all_df)
    samples["popularity"] = samples.candidate_news_id.apply(lambda x: [count[n] for n in x.split("\n")])
    samples[mn] = samples.popularity.apply(apply_popularity)
    return samples


def cal_topic_pop(row, news_topic, count):
    cans = row.candidate_news_id.split("\n")
    his = row.history_news_id.split("\n")
    cans_topic = [news_topic[n] for n in cans]
    his_topic = [news_topic[n] for n in his]
    pop = []
    for i, n in enumerate(cans):
        if cans_topic[i] in his_topic:
            pop.append(max(count[n] * 10, max(count.values()) + 1))
        else:
            pop.append(count[n])
    return pop


def generate_topic_pop_predictions(samples, mn="TopicPop", **kwargs):
    all_df, news_topic = load_all_data("small")
    all_df = all_df.sample(n=kwargs.get("sample_num", 200000), random_state=kwargs.get("seed", 42))
    count = cal_count(all_df)
    samples["popularity"] = samples.apply(lambda x: cal_topic_pop(x, news_topic, count), axis=1)
    samples[mn] = samples.popularity.apply(apply_popularity)
    return samples


def generate_random_predictions(candidate_col):
    """Generate random recommendations based on the candidate column with correct format."""
    return candidate_col.apply(
        lambda x: ",".join(
            [
                f"C{item + 1}"
                for item in random.sample(range(len(x.split("\n"))), len(x.split("\n")))
            ]
        )
    )


if __name__ == "__main__":
    args = load_cmd_line()
    seed_everything(args.get("seed", 42))
    data_root_dir = args.get("data_root_dir", "test_group/variant5")
    variant_name = args.get("variant_name", "sample400by_ratio")
    sampled_df = pd.read_csv(f"{data_root_dir}/{variant_name}.csv")
    num = args.get("num", 400)
    sampled_df = sampled_df.sample(args.get("num", 400))
    os.makedirs("generated_data", exist_ok=True)
    os.makedirs("result", exist_ok=True)
    model_name = args.get("model_name", "random")
    params = {"sample_num": args.get("sample_num", 200000), "seed": args.get("seed", 42)}
    if model_name == "random":
        sampled_df[model_name] = generate_random_predictions(sampled_df["candidate"])
    elif model_name == "MostPop":
        sampled_df = generate_most_pop_predictions(sampled_df, model_name, **params)
    elif model_name == "TopicPop":
        sampled_df = generate_topic_pop_predictions(sampled_df, model_name, **params)
    data_cols = ["impression_id", "history", "candidate", "label", model_name]
    metric_list = ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]
    suffix = f"{model_name}_{variant_name}_{args.get('seed', 42)}"
    generated_data_root = f"generated_data/{data_root_dir}_{num}"
    os.makedirs(generated_data_root, exist_ok=True)
    score_root = f"result/{data_root_dir}_{num}"
    generated_output_path = f"{generated_data_root}/{suffix}.csv"
    os.makedirs(score_root, exist_ok=True)
    score_path = f"{score_root}/{suffix}.csv"
    metric_funcs = [getattr(module_metric, met) for met in ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]]
    results = []
    for index in tqdm(sampled_df.index, total=len(sampled_df)):
        line = {col: sampled_df.loc[index, col] for col in data_cols}
        output_list, label_list = convert2list(line[model_name], line["label"], line["candidate"])
        line.update(evaluate_list(output_list, label_list, metric_funcs))
        line.update(evaluate_list(output_list, label_list, metric_funcs))
        results.append(line)
        save2csv(results, generated_output_path)
        cal_avg_scores(results, score_path, model_name, metric_list)
    df = pd.read_csv(score_path)
    df["data_group"] = variant_name
    df["seed"] = args.get("seed", 42)
    df.to_csv(score_path, index=False)
