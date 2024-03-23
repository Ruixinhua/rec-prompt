import json
import re
import copy
import os
import logging
import sys
import torch
import numpy as np
import random
import math
import pandas as pd

from logging import getLogger
from pathlib import Path
from enum import Enum

from logging.handlers import RotatingFileHandler


def setup_logging(level=logging.INFO, **kwargs):
    # Check if log directory exists, create it if it doesn't
    log_path = kwargs.get("log_path", get_project_root() / "log/tmp.log")
    log_path = log_path if isinstance(log_path, Path) else Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Configure logging
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=[RotatingFileHandler(log_path, maxBytes=10485760, backupCount=5)])
    # Get the logger for 'httpx'
    httpx = logging.getLogger("httpx")

    # Set the logging level to WARNING to ignore INFO and DEBUG logs
    httpx.handlers = [RotatingFileHandler(get_project_root() / "log/httpx.log", maxBytes=10485760, backupCount=5)]
    httpx.setLevel(logging.WARNING)
    logger = logging.getLogger(kwargs.get("run_name", "rec-prompt"))
    return logger


def convert_config_dict(config_dict):
    r"""This function convert the str parameters to their original type."""
    check_set = (str, int, float, list, tuple, dict, bool, Enum)
    for key in config_dict:
        param = config_dict[key]
        if not isinstance(param, str):
            continue
        try:
            value = eval(
                param
            )  # convert str to int, float, list, tuple, dict, bool. use ',' to split integer values
            if value is not None and not isinstance(value, check_set):
                value = param
        except (NameError, SyntaxError, TypeError, ValueError):
            if isinstance(param, str):
                if param.lower() == "true":
                    value = True
                elif param.lower() == "false":
                    value = False
                else:
                    if "," in param:  # split by ',' if it is a string
                        value = []
                        for v in param.split(","):
                            if len(v) == 0:
                                continue
                            try:
                                v = eval(v)
                            except (NameError, SyntaxError, TypeError, ValueError):
                                v = v
                            value.append(v)
                    else:
                        value = param
            else:
                value = param
        config_dict[key] = value
    return config_dict


def load_cmd_line(**kwargs):
    """
    Load command line arguments
    :return: dict
    """
    cmd_config_dict = {}
    unrecognized_args = []
    if "ipykernel_launcher" not in sys.argv[0]:
        for arg in sys.argv[1:]:
            if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                unrecognized_args.append(arg)
                continue
            cmd_arg_name, cmd_arg_value = arg[2:].split("=")
            if (
                cmd_arg_name in cmd_config_dict
                and cmd_arg_value != cmd_config_dict[cmd_arg_name]
            ):
                raise SyntaxError(
                    "There are duplicate commend arg '%s' with different value." % arg
                )
            else:
                cmd_config_dict[cmd_arg_name] = cmd_arg_value
    if len(unrecognized_args) > 0:
        logger = kwargs.get("logger", getLogger())
        logger.warning(
            f"Unrecognized command line arguments(correct is '--key=value'): {' '.join(unrecognized_args)}"
        )
    cmd_config_dict = convert_config_dict(cmd_config_dict)
    cmd_config_dict["cmd_args"] = copy.deepcopy(cmd_config_dict)
    return cmd_config_dict


def check_empty(value):
    if value in [None, "", "NA", "missing", "nan"] or (isinstance(value, float) and math.isnan(value)):
        return True
    return False


def to_format_excel(saved_df, saved_path, column_widths=None, **kwargs):
    """Save the dataframe to an Excel file with the given column widths in a beautiful format.

    :param saved_df: dataframe to be saved
    :param saved_path: path to save the dataframe
    :param column_widths: column widths in the format of {"A:A": 15, "B:C": 80, "D:D": 10, "E:XFD": 80}
    """
    sheet_name = kwargs.get("sheet_name", "overall")
    logger = kwargs.get("logger", getLogger())
    if column_widths is None:
        column_widths = {
            "A:A": 15, "B:C": 80, "D:D": 10, "E:XFD": 80  # XFD is the last column in Excel
        }
    writer = pd.ExcelWriter(saved_path, engine='xlsxwriter')
    saved_df.to_excel(writer, sheet_name=sheet_name, index=False)
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    # Format for cell: wrap text, align top-left
    cell_format = workbook.add_format({
        'text_wrap': True,
        'valign': 'top',
        'align': 'left',
    })
    for col_num, value in column_widths.items():
        # Apply the format to the cells with content
        worksheet.set_column(col_num, value, cell_format)
    worksheet.freeze_panes(1, 1)

    # Define a format for the cell content with text wrapping, align to the top and left
    cell_format = workbook.add_format()
    cell_format.set_text_wrap()
    cell_format.set_align('top')
    cell_format.set_align('left')
    # Define and apply a header format
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'valign': 'top',
        'align': 'left',
        'fg_color': '#D7E4BC',
        'border': 1
    })
    worksheet.set_row(0, 18)
    for i in range(1, kwargs.get("case_num", 100)):
        worksheet.set_row(i, kwargs.get("cell_height", 500))  # 设置行高为20
    for col_num, value in enumerate(saved_df.columns.values):
        worksheet.write(0, col_num, value, header_format)
    writer.close()
    logger.info(f"Save the dataframe to {saved_path}.")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save2csv(results_list, saved_path):
    df = pd.DataFrame.from_records(results_list)
    df.to_csv(saved_path, index=False)
    return df


def unique_in_order(iterable):
    unique_list = []
    for item in iterable:
        if item not in unique_list:
            unique_list.append(item)
    return unique_list


def search_patten(text, patten):
    extracted_text = re.findall(patten, text, re.DOTALL)
    for text in extracted_text:
        ranks = unique_in_order(re.findall(r"C\d+", text))
        if len(ranks):
            return ranks
    return False


def extract_raw_output(out, cans):
    ranks = []
    filter_index = 0
    for match in re.finditer(re.escape("Ranked news:"), out):
        filter_index = match.end()
    out = out[filter_index:]
    out = " ".join(out.split())
    for i, c in enumerate(cans.split('\n')):
        if not re.search(r"C\d+: ", c):
            continue
        c = re.sub(r"C\d+: ", '', c.strip())
        c = " ".join(c.split())
        start = len(out)
        for match in re.finditer(re.escape(c), out.strip()):
            start = min(start, match.start())
        ranks.append(start)
    index_sorted = sorted(range(len(ranks)), key=lambda k: ranks[k])
    return [f"C{i+1}" for i in index_sorted][:10]


def extract_output(output, candidates=None, match_pattern=False, **kwargs):
    """
    extract the ranked news from the output
    :param output: String of the output
    :param candidates: String of the candidates
    :param match_pattern: Boolean, if True, return the ranks if the pattern is found in the output else return False
    :return: ranks or False
    """
    pattens = [
        r"<START>(.+?)<END>",
        r"Ranked news: ([^\n]+)",
        r"ranking: ([^\n]+)",
        r"Candidate news ranked solely by relevance to the user's interests: ([^\n]+)"
    ]
    pattens.extend(kwargs.get("pattens", []))
    ranks = False
    for patten in pattens:
        # Check if the pattern was found in the previous patten
        ranks = search_patten(output, patten)
        if ranks:
            break
    if match_pattern:
        if not ranks or len(ranks) == 0:
            check_keywords = ["ranked solely by relevance to the user's interests", "solely by relevance to the user's interests", "ranked news", "ranking"]
            for keyword in check_keywords:
                if keyword in output.lower():
                    ranks = re.findall(r"C\d+", output)
                    if len(ranks):
                        return ranks
            return False
        else:
            return ranks
    else:
        if not ranks or len(ranks) == 0:
            ranks = extract_raw_output(output, candidates)
            if len(ranks) == 0:
                return False
            else:
                return ranks
        else:
            return ranks


def search_label_index(label: str, candidates: str):
    """Search the index of the label in the candidates."""
    match = re.search(r"C\d+", label)
    if match:
        label_index = [int(re.findall(r'\d+', i)[0]) - 1 for i in label.split(",")]
        label_list = [1 if i in label_index else 0 for i in range(len(candidates.split("\n")))]
    else:
        label_list = [1 if candidate in label else 0 for candidate in candidates.split("\n")]
    return label_list


def convert2list(ranks: str, label: str, candidates: str):
    length = len(candidates.split("\n"))
    label_list = np.array(search_label_index(label, candidates))
    output_list = np.array([0.0] * length)
    for i, c in enumerate(ranks.split(',')):
        index = int(re.findall(r'\d+', c)[0]) - 1
        if index >= length:
            continue
        output_list[index] = round(1 / (i + 1), 3)
    return output_list, label_list


def evaluate_list(output: list, label: list, metrics=None):
    """
    Evaluate the output and label using the metrics.
    :param output: a list of output
    :param label: a list of label
    :param metrics: metrics list should be in "group_auc", "mean_mrr", "ndcg_5", "ndcg_10".
    :return:
    """
    return {func.__name__: round(func(label, output), 5) for func in metrics}


def cal_avg_scores(results_list, saved_path, model="gpt-3.5-turbo", metrics=None, **kwargs):
    """Calculate the average scores of the results."""
    if metrics is None:
        metrics = ["nDCG@5", "nDCG@10", "MRR"]
    df = pd.DataFrame.from_records(results_list)
    df[metrics] = df[metrics] * 100
    avg_scores = df[metrics].mean().round(2).to_dict()
    avg_scores["model"] = model
    avg_scores["sample_num"] = len(results_list)
    df = pd.DataFrame.from_records([avg_scores])
    flag = 1
    while flag:
        try:
            old = pd.read_csv(saved_path) if os.path.exists(saved_path) else pd.DataFrame()
            flag = 0
        except:
            continue
    saved_params = kwargs.get("saved_params", {})
    # store these params to the dataframe
    for param, value in saved_params.items():
        saved_value = value
        if isinstance(value, dict):
            saved_value = json.dumps(value)
        if isinstance(value, list):
            saved_value = ",".join(value)
        df[param] = saved_value
    in_order_num = kwargs.get("in_order_num", 0)
    df["in_order_ratio"] = round(in_order_num / len(results_list), 3)
    run_id = kwargs.get("run_id", 0)
    df["run_id"] = run_id
    new = pd.concat([old, df])
    new.drop_duplicates(subset=["run_id", "model"], inplace=True, keep="last")
    new.drop_duplicates(subset=metrics, inplace=True, keep="last")
    columns = ["run_id", "model"] + metrics + ["sample_num", "in_order_ratio"] + list(saved_params.keys())
    new[columns].to_csv(saved_path, index=False)
    return df


def is_descending(array):
    for i in range(len(array) - 1):
        if array[i] < array[i + 1]:
            return False
    return True


def get_project_name(**kwargs):
    return kwargs.get("project_name", "rec-prompt")


def get_project_root(**kwargs):
    """
    Get the project(rec-prompt) root directory
    :param kwargs: project_name
    :return: Path object
    """
    project_name = kwargs.get("project_name", get_project_name(**kwargs))
    file_parts = Path(os.getcwd()).parts
    abs_path = Path(f"{os.sep}".join(file_parts[: file_parts.index(project_name) + 1]))
    return Path(os.path.relpath(abs_path, os.getcwd()))
