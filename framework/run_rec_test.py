import config

from utils import seed_everything, get_project_root
from recommender import run_recommender
from run_recommender import load_args


if __name__ == "__main__":
    seed_everything(42)
    args_list = [
        # GPT-3.5 IO and CoT Prompting
        {"run_args": "run_gpt3_1106_args", "dataset": "test", "system_instruction": "pure_nonjson",
         "prompt_strategy": "io_prompt", "return_json": False, "run_num": 400, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "test", "system_instruction": "pure_nonjson",
         "prompt_strategy": "cot_prompt", "return_json": False, "run_num": 400, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "test_cat", "system_instruction": "pure_nonjson",
         "prompt_strategy": "io_prompt", "return_json": False, "run_num": 400, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "test_cat", "system_instruction": "pure_nonjson",
         "prompt_strategy": "cot_prompt", "return_json": False, "run_num": 400, "max_tokens": 1600},

        # GPT-3.5 RecPrompt IO and CoT Prompting w/o category and test on w/o category dataset
        {"run_args": "run_gpt3_1106_args", "dataset": "test", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_io_4t3", "return_json": False, "run_num": 400, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "test", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_cot_4t3", "return_json": False, "run_num": 400, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "test", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_io_3t3", "return_json": False, "run_num": 400, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "test", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_cot_3t3", "return_json": False, "run_num": 400, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "test_cat", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_io_3t3", "return_json": False, "run_num": 400, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "test_cat", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_cot_3t3", "return_json": False, "run_num": 400, "max_tokens": 1600},
        # # GPT-3.5 RecPrompt IO and CoT Prompting with Category and test on category dataset
        {"run_args": "run_gpt3_1106_args", "dataset": "test_cat", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_io_cat_4t3", "return_json": False, "run_num": 400, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "test_cat", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_cot_cat_4t3", "return_json": False, "run_num": 400, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "test_cat", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_cot_cat_4t3", "return_json": False, "run_num": 400, "max_tokens": 2400},
        {"run_args": "run_gpt3_1106_args", "dataset": "test_cat", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_cot_cat_4t3", "return_json": False, "run_num": 400, "max_tokens": 3600},
        # GPT-3.5 RecPrompt IO and CoT Prompting w/o Category and test on category dataset
        {"run_args": "run_gpt3_1106_args", "dataset": "test_cat", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_io_4t3", "return_json": False, "run_num": 400, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "test_cat", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_cot_4t3", "return_json": False, "run_num": 400, "max_tokens": 1600},

        # GPT-4 IO and CoT Prompting: return JSON format.
        {"run_args": "run_gpt4_1106_args", "dataset": "test", "system_instruction": "rankonly_json_format",
         "prompt_strategy": "io_pure_prompt", "return_json": True, "run_num": 400, "max_tokens": 50},
        {"run_args": "run_gpt4_1106_args", "dataset": "test_cat", "system_instruction": "rankonly_json_format",
         "prompt_strategy": "io_pure_prompt", "return_json": True, "run_num": 400, "max_tokens": 50},
        {"run_args": "run_gpt4_1106_args", "dataset": "test", "system_instruction": "rankonly_json_format",
         "prompt_strategy": "cot_pure_prompt", "return_json": True, "run_num": 400, "max_tokens": 50},
        {"run_args": "run_gpt4_1106_args", "dataset": "test_cat", "system_instruction": "rankonly_json_format",
         "prompt_strategy": "cot_pure_prompt", "return_json": True, "run_num": 150, "max_tokens": 50},

        # GPT-4 RecPrompt-IO Prompting: return non-JSON format.
        {"run_args": "run_gpt4_1106_args", "dataset": "test", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_io_4t4", "return_json": False, "run_num": 400, "max_tokens": 50},
        {"run_args": "run_gpt4_1106_args", "dataset": "test_cat", "system_instruction": "pure_nonjson",
         "prompt_strategy": "rec_io_4t4", "return_json": False, "run_num": 400, "max_tokens": 50},
        # GPT-4 RecPrompt-CoT Prompting: return JSON format.
        {"run_args": "run_gpt4_1106_args", "dataset": "test", "system_instruction": "rankonly_json_format",
         "prompt_strategy": "rec_cot_cat_4t4", "return_json": True, "run_num": 400, "max_tokens": 50},
        {"run_args": "run_gpt4_1106_args", "dataset": "test_cat", "system_instruction": "rankonly_json_format",
         "prompt_strategy": "rec_cot_cat_4t4", "return_json": True, "run_num": 400, "max_tokens": 100},
        {"run_args": "run_gpt4_1106_args", "dataset": "test_cat", "system_instruction": "rankonly_json_format",
         "prompt_strategy": "rec_cot_cat_4t4", "return_json": True, "run_num": 400, "max_tokens": 200},

        # GPT-4 RecPrompt-CoT Prompting GPT-4 Tune GPT-3.5 test w/o category: return JSON format
        {"run_args": "run_gpt4_1106_args", "dataset": "test", "system_instruction": "rankonly_json_format",
         "prompt_strategy": "rec_cot_4t3", "return_json": True, "run_num": 400, "max_tokens": 50},
        # GPT-4 RecPrompt-CoT Prompting GPT-4 Tune GPT-3.5 test on the category dataset: return JSON format
        {"run_args": "run_gpt4_1106_args", "dataset": "test_cat", "system_instruction": "rankonly_json_format",
         "prompt_strategy": "rec_cot_4t3", "return_json": True, "run_num": 400, "max_tokens": 50},

        {"run_args": "run_gpt3_1106_args", "dataset": "case_cat", "system_instruction": "explanation_json",
         "prompt_strategy": "cot_prompt", "return_json": True, "run_num": 100, "max_tokens": 1600},
        {"run_args": "run_gpt4_1106_args", "dataset": "case_cat", "system_instruction": "explanation_json",
         "prompt_strategy": "cot_prompt", "return_json": True, "run_num": 100, "max_tokens": 1600},
        {"run_args": "run_gpt3_1106_args", "dataset": "case_cat", "system_instruction": "explanation_json",
         "prompt_strategy": "rec_cot_cat_4t3", "return_json": True, "run_num": 100, "max_tokens": 1600},
        {"run_args": "run_gpt4_1106_args", "dataset": "case_cat", "system_instruction": "explanation_json",
         "prompt_strategy": "rec_cot_cat_4t4", "return_json": True, "run_num": 100, "max_tokens": 1600},
    ]
    for args in args_list:
        if not hasattr(config, args.get("run_args")):
            raise ValueError(f"Invalid running arguments name: {args.get('run_args')}")
        run_args_name = args.get("run_args")
        args["run_args"] = getattr(config, run_args_name)
        args["score_path"] = get_project_root() / f"output/result/final_test_all.csv"
        data_tag = "w/o category"
        if "cat" in args.get("dataset"):
            data_tag = "with category"
        args = load_args(**args)
        print(f"Start running {run_args_name} {args['tag']}...")
        args["model_args"]["data_tag"] = data_tag
        args["model_args"]["prompt_strategy"] = args.get("prompt_strategy")
        args["model_args"]["system_instruction"] = args.get("system_instruction")
        score = run_recommender(**args)
        current_score = score[["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]].to_latex(float_format='%.2f')
        print(f"Current Score is {current_score}")
