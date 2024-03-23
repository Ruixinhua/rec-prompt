import json
import argilla as rg
import pandas as pd

from pathlib import Path
from utils import load_cmd_line, to_format_excel, get_project_root, load_api_key


if __name__ == "__main__":
    HF_TOKEN = load_api_key(api_key="HF_TOKEN")
    args = load_cmd_line()
    news_ds_name = args.get("news_ds_name", "NewsTopicsAnnotationWithScores")

    workspace_name = args.get("workspace_name", "llm4rec_v3")
    default_anno_dir = get_project_root() / "data/explain_eval/annotation"
    topic_path = Path(args.get("topic_path", f"{default_anno_dir}/news_topics_human_scores.xlsx"))
    topic_path.parent.mkdir(parents=True, exist_ok=True)
    rg.init(
        api_url="https://Rui98-LLL4RecTopicScore.hf.space",
        api_key="owner.apikey",
        extra_headers={"Authorization": f"Bearer {HF_TOKEN}"},
        workspace=workspace_name,
    )
    users = [u for u in rg.User.list() if u.role == "annotator"]
    feedback = rg.FeedbackDataset.from_argilla(news_ds_name, workspace=workspace_name)
    user_mapper = {u.id: u.username for u in users}
    topic_df = pd.DataFrame(columns=["news_headline", "annotator", "annotation", "news_info"])

    for record_ix, record in enumerate(feedback):
        if len(record.responses):
            news_record = {"news_headline": record.fields["news_headline"], "news_info": record.fields["news_info"]}
            topics_list = {topic.split(": ", 1)[0]: topic.split(": ", 1)[1] for topic in
                           record.fields["topics"].split("\n")}
            for response in record.responses:
                if response.status != "submitted":
                    print(user_mapper[response.user_id], response)
                    continue
                topic_record = {"annotator": user_mapper[response.user_id]}
                topics = {}
                results = response.values["topics-index"].value
                for i in range(len(topics_list)):
                    if str(i+1) in results:
                        topics[topics_list[str(i+1)]] = 1
                    else:
                        topics[topics_list[str(i+1)]] = 0
                topic_record["annotation"] = json.dumps(topics, indent=4)
                topic_record.update(news_record)
                topic_df = pd.concat([topic_df, pd.DataFrame(topic_record, index=[
                    f"{topic_record['annotator']}_{record_ix}"])], ignore_index=True)
    column_width = {
        "A:A": 50, "B:B": 10, "C:C": 30, "D:D": 60
    }
    to_format_excel(topic_df, topic_path, column_widths=column_width, cell_height=80, case_num=len(topic_df))
