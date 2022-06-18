import collections
import os
from typing import Dict

import streamlit as st
from datasets import get_dataset_config_names
from jinja2 import Environment, FileSystemLoader

import utils
from configuration import OPTIMIZERS_ACCELERATE, OPTIMIZERS_TRAINER, TASKS, TASKS_TO_PIPELINE_TAG
from utils import (get_dataset_infos_dict, get_datasets, get_model_to_model_id,
                   render_features)


def show_API_component(inputs: Dict[str, str]) -> Dict[str, str]:
    template_dict = collections.defaultdict()
    template_dirs = [
        f for f in os.scandir("templates") if f.is_dir() and f.name != "example"
    ]
    template_dirs = sorted(template_dirs, key=lambda e: e.name)
    for template_dir in template_dirs:
        template_dict[template_dir.name] = template_dir.path
    st.write("## API")
    inputs['api'] = st.selectbox(

        "Which Hugging Face API do you want to use?", list(template_dict.keys())
    )
    inputs['template_dir'] = template_dict.get(inputs['api'])
    return inputs


def show_model_component(inputs: Dict[str, str]) -> Dict[str, str]:

    model_info = get_model_to_model_id()
    models = model_info['model_to_model_id']
    models_pipeline = model_info["model_to_pipeline_tag"]
    st.write("## Model")
    models_for_task = []
    for model in models:
        if (models_pipeline[model] == inputs["nlp_task"]):
            models_for_task.append(model)
    model = st.selectbox("Which model?", list(models_for_task))
    inputs["model_checkpoint"] = models.get(model)
    inputs["pretrained"] = st.checkbox("Use pre-trained model")
    return inputs


def show_task_component(inputs: Dict[str, str]) -> Dict[str, str]:
    st.write("## Task")
    task = st.selectbox("Which task?", TASKS)
    inputs["task"] = task
    inputs["nlp_task"] = st.selectbox(
        "Which NLP task?", TASKS_TO_PIPELINE_TAG[task])
    return inputs


def show_input_data_component(inputs: Dict[str, str]) -> Dict[str, str]:
    st.write("## Input data")
    english_datasets = get_datasets()
    english_datasets_for_task = []

    for dataset in english_datasets:
        for task_category in english_datasets[dataset]:
            if task_category == inputs["nlp_task"]:
                english_datasets_for_task.append(dataset)
                continue

    inputs["dataset"] = st.selectbox(
        "Which one?", tuple(english_datasets_for_task)
    )

    configs = get_dataset_config_names(inputs["dataset"])
    inputs["subset"] = st.selectbox("Which subset?", list(configs))

    data_info_dict = get_dataset_infos_dict(
        inputs["dataset"], inputs["subset"])

    assert data_info_dict.splits is not None
    if 'train' in list(data_info_dict.splits.keys()):
        train_index = list(data_info_dict.splits.keys()).index('train')
    else:
        train_index = 0

    inputs["train"] = st.selectbox("Which split for training?", list(
        data_info_dict.splits.keys()), index=train_index)

    if 'validation' in list(data_info_dict.splits.keys()):
        validation_index = list(
            data_info_dict.splits.keys()).index('validation')
    else:
        validation_index = len(list(data_info_dict.splits.keys()))-1

    inputs["validation"] = st.selectbox("Which split for validation?", list(
        data_info_dict.splits.keys()), index=validation_index)

    assert data_info_dict.features is not None
    feature_index = 0
    if inputs["nlp_task"] == 'translation':
        if 'translation' in list(data_info_dict.features.keys()):
            feature_index = list(
                data_info_dict.features.keys()).index('translation')

    inputs["feature"] = st.selectbox(
        "Which data feature?", list(data_info_dict.features.keys()), feature_index)

    if inputs["feature"] == 'translation':
        inputs["source_language"] = st.selectbox(
            "Which language for source?", list(data_info_dict.features['translation'].languages))
        inputs["target_language"] = st.selectbox(
            "Which language for target?", list(data_info_dict.features['translation'].languages))

    return inputs


def show_preprocessing_component(inputs: Dict[str, str]) -> Dict[str, str]:
    st.write("## Preprocessing")
    inputs["block_size"] = st.number_input(
        "The length of each block (i.e. context size)", 1, None, 128)

    if inputs["task"] == "MaskedLM":
        inputs["mlm_probability"] = st.number_input(
            "The probability with which to (randomly) mask tokens in the input", 0.0, 1.00, 0.15)
        inputs["whole_word_masking"] = st.checkbox(
            "Use whole word masking")
    return inputs


def show_training_comoponent(inputs: Dict[str, str]) -> Dict[str, str]:
    st.write("## Training")

    # inputs['with_tracker'] = st.selectbox(
    #     "Loggers to monitor the training ", ["none", "all", "tensorboard", "wandb", "comet_ml"])
    inputs["seed"] = st.number_input(
        "Seed", 1, None, 4)
    
    if inputs['api'] == 'Accelerate':
        optimizer_dict_to_use = OPTIMIZERS_ACCELERATE
    else:
        optimizer_dict_to_use = OPTIMIZERS_TRAINER
        
    inputs["optimizer"] = st.selectbox(
            "Optimizer", list(optimizer_dict_to_use.keys()))
    default_lr = optimizer_dict_to_use[inputs["optimizer"]]
    inputs["lr"] = st.number_input(
        "Learning rate", 0.000, None, default_lr, format="%f"
    )
    inputs["use_weight_decay"] = st.checkbox("Use weight decay")
    if inputs["use_weight_decay"]:
        inputs["weight_decay"] = st.number_input(
            "Weight decay", 0.000, None, 0.01, format="%f"
        )

    inputs["gradient_accumulation_steps"] = st.number_input(
        "Gradient Accumulation Steps", 1, None, 8)

    inputs['lr_scheduler_type'] = st.selectbox(
        "The scheduler type to use", ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    inputs['num_warmup_steps'] = st.number_input(
        "Num warmup steps", 0, None, 0)
    inputs["batch_size"] = st.number_input("Batch size", 1, None, 32)
    inputs["num_epochs"] = st.number_input("Epochs", 1, None, 3)
    return inputs


def show_datset_view_component(inputs: Dict[str, str]) -> Dict[str, str]:
    data_info_dict = get_dataset_infos_dict(
        inputs["dataset"], inputs["subset"])
    st.write(f'## Dataset view: {inputs["dataset"]}/{inputs["subset"]}')
    st.markdown(
        "*Homepage*: "
        + data_info_dict.homepage
        + "\n\n*Dataset*: https://github.com/huggingface/datasets/blob/master/datasets/%s/%s.py"
        % (inputs["dataset"], inputs["dataset"])
    )
    s = []
    s .append('dataset' + "=" + inputs["dataset"])
    s.append('config' + "=" + inputs["subset"])
    st.markdown(
        "*Permalink*: https://huggingface.co/datasets/viewer/?"
        + "&".join(s)
    )
    # https://github.com/huggingface/datasets-viewer/blob/master/run.py#L282
    st.write(f'{data_info_dict.description}')
    st.write(render_features(data_info_dict.features))
    # TODO make a conditional if the size of the data is too big, switch to streaming mode
    # TODO cashe this part of the code
    # selected_dataset = load_dataset(
    #     inputs["dataset"], inputs["subset"], split=inputs["train"], streaming=True)
    # print(selected_dataset)
    # print(next(iter(selected_dataset)))
    return inputs

def show_code_component(inputs: Dict[str, str]) -> Dict[str, str]:
    # Generate code and notebook based on template.py.jinja file in the template dir.
    env = Environment(
        loader=FileSystemLoader(inputs['template_dir']), trim_blocks=True, lstrip_blocks=True,
    )

    template = env.get_template(f'task_templates/{inputs["nlp_task"]}.py.jinja')
    code = template.render(header=utils.code_header, notebook=False, **inputs)
    notebook_code = template.render(
        header=utils.notebook_header, notebook=True, **inputs)

    notebook = utils.to_notebook(notebook_code)

    st.write(f'## Code view: {inputs["api"]}')
    st.write("")  # add vertical space
    col1, col2 = st.columns(2)
    with col1:
        utils.download_button(code, "generated-code.py", "🐍 Download (.py)")
    with col2:
        utils.download_button(
            notebook, "generated-notebook.ipynb", "📓 Download (.ipynb)")
    colab_error = st.empty()
    # Display code.
    st.code(code)
    return inputs
