import torch
import os
import json
import time

from langchain.cache import SQLiteCache
from langchain.globals import set_llm_cache
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatCohere
from langchain_core.messages import HumanMessage, SystemMessage
from transformers import AutoTokenizer, AutoModelForCausalLM
from .general_utils import get_project_root, setup_logging


def load_model_tokenizer(model_name, **kwargs):
    cache_dir = kwargs.get("cache_dir", None)
    token = load_api_key(kwargs.get("hg_token_path", "hf_token.json"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=token,
        force_download=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        token=token,
        force_download=False,
        device_map="auto",
        torch_dtype=torch.float16 if kwargs.get("fp16", False) else "auto",
    )
    return model, tokenizer


def inference_llm_hf(model, tokenizer, full_prompt, **kwargs):
    torch.cuda.empty_cache()
    inputs = tokenizer(full_prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    try_num = kwargs.get("try_num", 5)
    num = 0
    while num < try_num:
        try:
            generate_ids = model.generate(
                input_ids,
                max_length=kwargs.get("max_length", 2048),
                max_new_tokens=kwargs.get("max_new_tokens", 1024),
                do_sample=kwargs.get("do_sample", False),
            )
            output = tokenizer.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0][len(full_prompt):]
            return output
        except Exception as e:
            print(e)
            num += 1
            torch.cuda.empty_cache()


def load_api_key(api_json_path=None, api_key="OPENAI_API_KEY"):
    """
    Load api keys from a json file and return keys
    :param api_json_path: json path to the api keys file
    :param api_key: the key to load
    :return: secret api key
    """
    if api_json_path is None:
        api_json_path = get_project_root() / "config" / "api_keys.json"
    return json.load(open(api_json_path))[api_key]


def set_environ(api_json_path=None, api_key="OPENAI_API_KEY"):
    """
    Load api keys from a json file and set the environment variable
    :param api_json_path: json path to the api keys file
    :param api_key: the key to load
    :return: None
    """
    os.environ[api_key] = load_api_key(api_json_path, api_key)


def set_cache(database_path=None):
    """
    Set the cache for the OpenAI models
    :param database_path: default is saved in {root}/.langchain.db using SQLite
    :return: None
    """
    if database_path is None:
        database_path = get_project_root() / "database/.langchain.db"
    set_llm_cache(SQLiteCache(database_path=database_path))


def load_model(**kwargs):
    """
    Load the OpenAI model
    :param kwargs: model_family, model_params, model_name, database_path
    :return:
    """
    model_family = kwargs.get("model_family", "openai")
    model_params = kwargs.get("model_params", {"temperature": 0, "max_tokens": 50})
    extra_params = kwargs.get("extra_params", {})
    if model_family.lower() == "openai":
        model_name = kwargs.get("model_name", "gpt-3.5-turbo-1106")
        set_environ()
        model = ChatOpenAI(model_name=model_name, model_kwargs=extra_params, **model_params)
    elif model_family.lower() == "cohere":
        model_name = kwargs.get("model_name", "command-light")
        set_environ(api_key="COHERE_API_KEY")
        model = ChatCohere(model_name=model_name, **model_params)
    else:
        raise ValueError(f"Model family {model_family} is not supported")
    set_cache(kwargs.get("database_path"))
    return model


def run_llm_model(llm_model, prompt, **kwargs):
    """
    Run the LLM model
    :param llm_model: the LLM, a langchain ChatModel instance
    :param prompt: the input prompt
    :param kwargs: chat_temp, template of input prompt
    :return: output from the LLM
    """
    output = None
    if isinstance(llm_model, ChatOpenAI) or isinstance(llm_model, ChatCohere):
        messages = [
            SystemMessage(content=kwargs.get("system_instruction")),
            HumanMessage(content=prompt),
        ]
        while output is None:
            try:
                output = llm_model.invoke(messages).content
            except Exception as e:
                print(e)
                time.sleep(2)
                continue
        return output
    else:
        raise ValueError(f"Model {llm_model} is not supported")
