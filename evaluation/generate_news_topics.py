import json
import torch

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from itertools import chain
from collections import defaultdict
from utils import get_project_root


class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX


def find_similar_groups(similarity_matrix, threshold=0.5):
    n = len(similarity_matrix)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= threshold:
                uf.union(i, j)

    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root in groups:
            groups[root].append(i)
        else:
            groups[root] = [i]
    return list(groups.values())


def load_news_info():
    saved_info_path = get_project_root() / "data/news_topics/news_topics_info.json"
    if saved_info_path.exists():
        return json.load(open(saved_info_path))
    news_df = pd.read_csv(get_project_root() / "data/MIND_case.csv")
    news_info = defaultdict(dict)
    for i, row in news_df.iterrows():
        titles = row.history_title.split("\n") + row.candidate_title.split("\n")
        category = row.history_category.split("\n") + row.candidate_category.split("\n")
        subvert = row.history_subvert.split("\n") + row.candidate_subvert.split("\n")
        abstract = row.history_abstract.split("\n") + row.candidate_abstract.split("\n")
        for t, c, s, a in zip(titles, category, subvert, abstract):
            news_info[t] = {"category": c, "subvert": s, "abstract": a}
    with open(saved_info_path, "w") as f:
        json.dump(news_info, f, indent=4)
    return news_info


def load_news_topics():
    root_dir = get_project_root() / "output/generated_data/case_cat"
    saved_topics_path = get_project_root() / "data/news_topics_count.json"
    if saved_topics_path.exists():
        return json.load(open(saved_topics_path))
    gpt3_output_df = pd.read_csv(root_dir / "gpt-3.5-turbo-1106--explain_machine_json--full.csv")
    gpt4_output_df = pd.read_csv(root_dir / "gpt-4-1106-preview--explain_machine_json--full.csv")
    news_topics_score = defaultdict(dict)
    for i, row in chain(gpt3_output_df.iterrows(), gpt4_output_df.iterrows()):
        history = [h.split("Category=>")[0].split(": ", 1) for h in row.history.split("\n")]
        history_mapper = {h[0]: h[1] for h in history}
        candidate = [c.split("Category=>")[0].split(": ", 1) for c in row.candidate.split("\n")]
        candidate_mapper = {c[0]: c[1] for c in candidate}
        output = json.loads(row.output)
        topic_user = output["topics of the user's interest"]
        topic_can = output["candidates' topics"]
        for topic, h_index in topic_user.items():
            for index in h_index:
                if index not in history_mapper:
                    continue
                if topic not in news_topics_score[history_mapper[index]]:
                    news_topics_score[history_mapper[index]][topic] = 0
                news_topics_score[history_mapper[index]][topic] += 1
        for c_index, topics in topic_can.items():
            for topic in topics:
                if c_index not in candidate_mapper:
                    continue
                if topic not in news_topics_score[candidate_mapper[c_index]]:
                    news_topics_score[candidate_mapper[c_index]][topic] = 0
                news_topics_score[candidate_mapper[c_index]][topic] += 1
    with open(saved_topics_path, "w") as f:
        json.dump(news_topics_score, f, indent=4)
    return news_topics_score


def generate_topic_embeddings(topics):
    sentences_embeddings = {}
    model_name = "paraphrase-MiniLM-L6-v2"  # all-MiniLM-L6-v2/sentence-t5-base
    model = SentenceTransformer(f'sentence-transformers/{model_name}')
    for topic in tqdm(topics, total=len(topics)):
        if topic not in sentences_embeddings:
            sentences_embeddings[topic] = model.encode(topic, convert_to_tensor=True)
    return sentences_embeddings


def cal_similarity():
    # Compute embedding for both lists
    news_topics = load_news_topics()
    topics_all = []
    for news, topic_scores in news_topics.items():
        topics_all.extend(topic_scores.keys())
    topics_all = set(topics_all)
    sentences_embeddings = generate_topic_embeddings(topics_all)
    news_groups = {}
    for news, topic_scores in tqdm(news_topics.items(), total=len(news_topics)):
        topics = list(topic_scores.keys())
        topic_embeddings = torch.stack([sentences_embeddings[t] for t in topics])
        topic_similarities = util.pytorch_cos_sim(topic_embeddings, topic_embeddings)
        groups = find_similar_groups(topic_similarities, 0.7)
        topic_group = [[topics[g] for g in group] for group in groups]
        news_groups[news] = topic_group
    return news_groups


def sort_scores():
    saved_score_path = get_project_root() / "data/news_topics/news_topics_scores.json"
    saved_sorted_path = get_project_root() / "data/news_topics/news_topics_sorted.json"
    if saved_score_path.exists() and saved_sorted_path.exists():
        return json.load(open(saved_score_path)), json.load(open(saved_sorted_path))
    news_groups = cal_similarity()
    news_topics_count = load_news_topics()
    news_topics_scores = defaultdict(list)
    news_topics_sorted = defaultdict(list)
    for news, groups in news_groups.items():
        group_scores_dict = defaultdict(int)
        for group in groups:
            group_scores_dict.update({
                topic: news_topics_count[news][topic]
                if news in news_topics_count and topic in news_topics_count[news] else 0
                for topic in group})
        group_scores = [sum([group_scores_dict[topic] for topic in group]) for group in groups]
        # sort groups by group_scores: from high to low
        groups = [group for _, group in sorted(zip(group_scores, groups), reverse=True)]
        for group in groups:
            # sort topics by group_scores: from high to low
            g_topics = [group_scores_dict[topic] for topic in group]
            group = [topic for _, topic in sorted(zip(g_topics, group), reverse=True)]
            news_topics_scores[news].extend([f"{topic}--{group_scores_dict[topic]}" for topic in group])
            news_topics_sorted[news].extend(group)
    with open(saved_score_path, "w") as f:
        json.dump(news_topics_scores, f, indent=4)
    with open(saved_sorted_path, "w") as f:
        json.dump(news_topics_sorted, f, indent=4)


if __name__ == "__main__":
    sort_scores()
