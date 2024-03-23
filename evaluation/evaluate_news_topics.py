import json
import os
import pandas as pd

from string import Template
from tqdm import tqdm
from utils import load_cmd_line, run_llm_model, to_format_excel, load_model, get_project_root

topic_inst = """You are a helpful assistant designed to output JSON. You are given some news information and a list of candidate topics that may related to the news content. You aim to check the correctness of these topics under "# Candidate Topics" according to the content under "# News Content"."""
check_topic = """Examine the news content under the "# News Content" section and candidate topics under the "# Candidate Topics" section. You should check topics under "# Candidate Topics" individually to see whether they describe the corresponding candidate news under "# News Content". The news content includes the headline, category, sub-category, and abstract. If the topic is related to the news content, then the correctness should be 1. Otherwise, the correctness is 0.
# News Content
```
${news}
```
# Candidate Topics
```
${topics}
```
Your response should include all topics under the section "# Candidate Topics" and should be in the following format:
```json
{
    "xxx": 1,
    "xxx": 1,
    "xxx": 0,
    ...
    "xxx": 0
}
```"""


if __name__ == "__main__":
    args = load_cmd_line()
    evaluate_model = args.get("evaluate_model", "gpt-3.5-turbo-1106")
    news_topics_path = get_project_root() / "data/news_topics/news_topics_sorted.json"
    news_topics = json.load(open(args.get("news_topics_path", news_topics_path)))
    news_info_path = get_project_root() / "data/news_topics/news_topics_info.json"
    news_info = json.load(open(args.get("news_info_path", news_info_path)))
    topic_path = "data/explain_eval/annotation/news_topics_machine.xlsx"
    topic_path = args.get("topic_path", get_project_root() / topic_path)
    topic_df = pd.DataFrame(columns=["news_headline", "annotator", "annotation", "news_info"])
    key = "annotation"
    column_widths = {
        "A:A": 50, "B:B": 10, "C:C": 30, "D:D": 60
    }
    response_format = {"type": "json_object"}
    if evaluate_model != "gpt-3.5-turbo-1106" and evaluate_model != "gpt-4-1106-preview":
        response_format = None
    if os.path.exists(topic_path):
        topic_df = pd.read_excel(topic_path)
    model_args = {
        "model_family": "openai",
        "model_name": evaluate_model,
        "model_params": {
            "temperature": 0, "max_tokens": 200
        },
        "extra_params": {"response_format": response_format}
    }
    model = load_model(**model_args)
    for index, (news_headline, topics) in tqdm(enumerate(news_topics.items()), total=len(news_topics)):
        check_condition = (topic_df["news_headline"] == news_headline) & (topic_df["annotator"] == evaluate_model)
        line = topic_df[check_condition]
        # if key in line and not check_empty(line[key].values) and len(line[key].values) > 0:
        #     continue
        info = news_info[news_headline]
        news_content = f"""**Headline**: {news_headline}\n**Category**: {info['category']}\n**Sub-category**: {info['subvert']}\n**Abstract**: {info['abstract']}"""
        evaluation_prompt = Template(check_topic).safe_substitute({"news": news_content, "topics": "\n".join(topics)})
        try:
            predict = run_llm_model(model, evaluation_prompt, system_instruction=topic_inst)
            topic_record = {"annotator": evaluate_model, "news_headline": news_headline, "annotation": predict,
                            "news_info": news_content}
            topic_df = pd.concat([topic_df, pd.DataFrame(topic_record, index=[f"{evaluate_model}_{index}"])],
                                 ignore_index=True)
            to_format_excel(topic_df, topic_path, column_widths=column_widths, cell_height=100, case_num=len(topic_df))
        except Exception as e:
            print(e)
            continue
