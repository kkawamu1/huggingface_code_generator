import base64
import importlib.util
import math
import re
import uuid
from types import ModuleType
from typing import Dict

import datasets
import jupytext
import requests
import streamlit as st
from datasets import DatasetInfo, get_dataset_infos
from datasets.info import DatasetInfosDict

from configuration import INCLUDED_USERS, TASKS_TO_PIPELINE_TAG


def import_from_file(module_name: str, filepath: str) -> ModuleType:
    """
    Imports a module from file.
    Args:
        module_name (str): Assigned to the module's __name__ parameter (does not 
            influence how the module is named outside of this function)
        filepath (str): Path to the .py file
    Returns:
        The module
    """
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_header(text: str):
    """
    Insert section header into a jinja file, formatted as notebook cell.

    Leave 2 blank lines before the header.
    """
    return f"""# # {text}
"""


def code_header(text: str):
    """
    Insert section header into a jinja file, formatted as Python comment.

    Leave 2 blank lines before the header.
    """
    seperator_len = (75 - len(text)) / 2
    seperator_len_left = math.floor(seperator_len)
    seperator_len_right = math.ceil(seperator_len)
    return f"# {'-' * seperator_len_left} {text} {'-' * seperator_len_right}"


def to_notebook(code: str) -> str:
    """Converts Python code to Jupyter notebook format."""
    notebook = jupytext.reads(code, fmt="py")
    # print(jupytext.writes(notebook, fmt="ipynb"))
    return jupytext.writes(notebook, fmt="ipynb")


def download_button(
    object_to_download: str, download_filename: str, button_text: str  # , pickle_it=False
):
    """
    Generates a link to download the given object_to_download.

    From: https://discuss.streamlit.io/t/a-download-button-with-custom-css/4220
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    some_txt_output.txt download_link_text (str): Text to display for download
    link.
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    pickle_it (bool): If True, pickle file.
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    Examples:
    --------
    download_link(your_df, 'YOUR_DF.csv', 'Click to download data!')
    download_link(your_str, 'YOUR_STRING.txt', 'Click to download text!')
    """

    # try:
    #     # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    # except AttributeError:
    #     b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace("-", "")
    button_id = re.sub("\d+", "", button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: .25rem .75rem;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;
            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = (
        custom_css
        + f'<a download="{download_filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br><br>'
    )

    st.markdown(dl_link, unsafe_allow_html=True)


@st.cache
def get_model_to_model_id() -> Dict[str, Dict[str, str]]:
    requests.get("https://huggingface.co")
    response = requests.get("https://huggingface.co/api/models")
    tags = response.json()
    model_to_model_id = {}
    model_to_pipeline_tag = {}

    for model in tags:
        model_name = model['modelId']
        is_community_model = "/" in model_name
        if is_community_model:
            user = model_name.split("/")[0]
            if user not in INCLUDED_USERS:
                continue

        # TODO Right now if pipiline is not defined, skip
        if "pipeline_tag" in model:
            model_to_model_id[model['id']] = model['modelId']
            model_to_pipeline_tag[model['id']] = model["pipeline_tag"]
    return {"model_to_model_id": model_to_model_id, "model_to_pipeline_tag": model_to_pipeline_tag}


@st.cache
def get_datasets() -> Dict[str, str]:
    english_datasets = {}
    response = requests.get(
        "https://huggingface.co/api/datasets?full=true&languages=en")
    tags = response.json()
    for dataset in tags:
        dataset_name = dataset["id"]

        is_community_dataset = "/" in dataset_name
        if is_community_dataset:
            # user = dataset_name.split("/")[0]
            # if user in INCLUDED_USERS:
            #     english_datasets.append(dataset_name)
            continue

        if "cardData" not in dataset:
            continue
        metadata = dataset["cardData"]

        if "languages" not in metadata:
            continue

        if "task_categories" not in metadata:
            continue

        task_is_valid = False
        for task_category in metadata["task_categories"]:

            if any(task_category in task for task in list(TASKS_TO_PIPELINE_TAG.values())):
                task_is_valid = True
        if not task_is_valid:
            continue

        languages = metadata["languages"]

        if "en" in languages or "en-US" in languages:
            english_datasets[dataset_name] = metadata["task_categories"]
    return english_datasets


@st.cache
def get_dataset_infos_dict(dataset: str, subset: str) -> DatasetInfo:
    return DatasetInfosDict(get_dataset_infos(dataset))[subset]

# https://github.com/huggingface/datasets-viewer/blob/master/run.py#L49


def render_features(features):
    # TODO redner translation object with the languages tags
    if isinstance(features, dict):
        return {k: render_features(v) for k, v in features.items()}
    if isinstance(features, datasets.features.ClassLabel):
        return features.names

    if isinstance(features, datasets.features.Value):
        return features.dtype

    if isinstance(features, datasets.features.Sequence):
        return {"[]": render_features(features.feature)}
    return features
