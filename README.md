---
title: Huggingface Code Generator
emoji: 🤗
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.10.0
app_file: ./app/main.py
pinned: false
license: cc
---

# huggingface_code_generator
**This app is a training code generator for 🤗 transformers**

Ever got frustrated when you needed to go through documentations over and over again to write a script, every time you wanted to finetune your favorite model on huggingface? No more! This application allows you to quickly run your experiment by generating a training script with your training configuration. Select one of the two amazing Huggingface training APIs, a NLP task, a model you want to train/finetune, dataset, and hyperparameters on GUI. That's it. Enjoy! 


<p align="center">
  <img src="assets/demo.png" width="800"/>
</p>

**You can access the [hosted version](https://huggingface.co/spaces/kkawamu1/huggingface_code_generator).**

## Suported Configuration
### API
Currently, two Huggingface training APIs are supported. [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) and [Accelerate](https://huggingface.co/docs/accelerate/index).

### NLP tasks
Currently, three types of tasks are supported. text-generation with [CausalML]((https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM)), fill-mask with [MaskedLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForMaskedLM), and translation with [Seq2SeqLM](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM).

### Models
All the non-communicty HF models that are filtered for a NLP task the user has chosen are available to the user (e.g. GPT2 for text-generation). In addtion, all the models from 'google', 'EleutherAI', "Helsinki-NLP", "bigscience", "facebook", "openai", and "microsoft" are also available. If you wish to add other users, you can edit INCLUDED_USERS in [configuration.py](https://github.com/kkawamu1/huggingface_code_generator/blob/main/app/configuration.py#L1). 

You can optionally choose to train from a pretrained checkpoint. If not, the model will be trained from a scratch.

### Input Data

Users are prompted to select a dataset, a subset to use, a split to use for training, a split to use for validation, and a data features to use. Please note that a data feature to use is expected be of type strings. In case of translation, the users are asked to choose a source language and a target language.

Dataset view is available on the main page. You can see the data features available and their types. To further inspect the dataset for actual examples, you should follow Permalink which will redirect you to [the official datasets viewr](https://huggingface.co/datasets/viewer/).

The datasets with English tag are available. You can modify this out by editing [utils.py](https://github.com/kkawamu1/huggingface_code_generator/blob/main/app/utils.py#L157). 


### Preprocessing

Users are asked to choose a length of each block (i.e. context size). This is the length of the input string to the model during training and validation. If you increase this number by too much, you might get a out of memory error when you train a model.

### Training parameters
The following parameters are available for users to choose:
1. seed for a reproducibility
2. Optimizer - Note that different sets of optimizers are available depending on the API selected.
3. Learning rate
4. Weight decay
5. Gradient Accumulation Steps
6. The scheduler type to use
7. Num warmup steps
8. Batch size
9. Epochs

## Training Code

You can copy the generated code. Alternatively, you can download as noteboook file or py file. It is one click away! Install the requirements writen at the top of the script/notebook, and just run it on your environment. Also, the code will generate a file for a CO2 emissions with [codecarbon](https://codecarbon.io/) after the training is done. We can be responsible for the environement, and do better; and that starts today!


## Setup
Clone the repository, install the required packages and run:
```bash
streamlit run ./app/main.py
```

## Special thanks

Finally, I would like to thank an amazing repository [traingenerator](https://github.com/jrieke/traingenerator) and its author jrieke from which I got the inspiration and I took some of the code piece and a code structure. This app is different from traingenerator in that this app focuses on Huggingface APIs and makes an extensive use of Huggingface library. Thus, this app supports NLP pielines unlike traingenerator. This app also displays the dataset card and data features of the public datasets on the same web page, allowing users to easily select both training parameters and dataset configuration in one tab.
