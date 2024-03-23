import json
import argilla as rg

from utils import load_cmd_line, load_api_key, get_project_root


def delete_dataset(dn, wn):
    try:
        rg.FeedbackDataset.from_argilla(name=dn, workspace=wn).delete()
        return True
    except RuntimeError as e:
        print("Failed to delete the dataset from the server.", e)
        return False


def push_dataset(ds, dn, wn):
    try:
        ds.push_to_argilla(name=dn, workspace=wn)
        return True
    except RuntimeError as e:
        print(f"Failed to push the dataset {dn} to the workspace {wn}.", e)
        return False


if __name__ == "__main__":
    HF_TOKEN = load_api_key(api_key="HF_TOKEN")
    args = load_cmd_line()
    news_ds_name = args.get("news_ds_name", "NewsTopicsAnnotation")
    news_topics = json.load(open(args.get("news_topics_path", get_project_root() / "data/news_topics/news_topics_sorted.json")))
    news_info = json.load(open(args.get("news_info_path", get_project_root() / "data/news_topics/news_topics_info.json")))
    override = args.get("override", True)
    workspace_name = args.get("workspace_name", "llm4rec_v3")
    try:
        rg.init(
            api_url="https://Rui98-LLL4RecTopicScore.hf.space",
            api_key="owner.apikey",
            extra_headers={"Authorization": f"Bearer {HF_TOKEN}"},
            workspace=workspace_name,
        )
    except ValueError as e:
        print(e)
        rg.init(
            api_url="https://Rui98-LLL4RecTopicScore.hf.space",
            api_key="owner.apikey",
            extra_headers={"Authorization": f"Bearer {HF_TOKEN}"},
        )
        rg.Workspace.create(workspace_name)
        rg.set_workspace(workspace_name)
    news_guide = "Examining the news headline and select all topics that are related to the news. Choose all topic indices that are relevant to the news based on the headline, category, sub-category, and abstract."
    news_fields = [
        rg.TextField(name="news_headline", title="News Headline"),
        rg.TextField(name="news_info", title="News Information"),
        rg.TextField(name="topics", title="Candidate Topics"),
    ]
    question_title = """Examine the news headline under the **News Headline** section and select all topics under the **Candidate Topics** section that are related to the news. Choose all topic indices that are relevant to the news. If none of the topics are relevant, select "None of the above"."""
    topic_indices = list(range(1, 100))
    topic_indices.append("None of the above")
    news_questions = [
        rg.MultiLabelQuestion(name=f"topics-index",
                              title=question_title,
                              labels=topic_indices,
                              required=True,
                              visible_labels=len(topic_indices),)
    ]
    news_dataset = rg.FeedbackDataset(
        guidelines=news_guide,
        fields=news_fields,
        questions=news_questions
    )
    news_records = []
    for news_headline, topics in news_topics.items():
        info = news_info[news_headline]
        info_str = f"Category: {info['category']}\nSub-category: {info['subvert']}\nAbstract: {info['abstract']}"
        fields_data = {
            "news_headline": news_headline,
            "topics": "\n".join([f"{i+1}: {topics[i]}" for i in range(len(topics))]),
            "news_info": info_str
        }
        news_record = rg.FeedbackRecord(fields=fields_data, external_id=news_headline)
        news_records.append(news_record)
    news_dataset.add_records(news_records)
    success = push_dataset(news_dataset, news_ds_name, workspace_name)
    if not success and override:
        delete_dataset(news_ds_name, workspace_name)
        success = push_dataset(news_dataset, news_ds_name, workspace_name)
    if success:
        print(f"Push the news dataset {news_ds_name} to the server.")
    users = [u for u in rg.User.list() if u.role == "annotator"]
    workspace = rg.Workspace.from_name(workspace_name)
    for user in users:
        try:
            workspace.add_user(user.id)
        except Exception as e:
            print(e)
            pass
