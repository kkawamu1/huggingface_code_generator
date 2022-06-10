INCLUDED_USERS = ['google', 'EleutherAI',
                  "Helsinki-NLP", "bigscience", "facebook", "openai", "microsoft"]

#TODO create a tempalte for text2text-generation
# TASKS_TO_PIPELINE_TAG = {
#     "CausalLM": ['text-generation'], "MaskedLM": ["fill-mask"], "Seq2SeqLM": ['text2text-generation', 'translation']}
TASKS_TO_PIPELINE_TAG = {
    "CausalLM": ['text-generation'], "MaskedLM": ["fill-mask"], "Seq2SeqLM": ['translation']}


TASKS = list(TASKS_TO_PIPELINE_TAG.keys())

OPTIMIZERS_ACCELERATE = {
    "AdamW": 0.0001, "Adadelta": 1.0, "Adagrad": 0.01, "Adam": 0.001, "SparseAdam": 0.001, "Adamax": 0.002, "ASGD": 0.01, "LBFGS": 1.0, "NAdam": 0.002, "RAdam": 0.001, "RMSprop": 0.01, "Rprop": 0.01, "SGD": 0.01
}

OPTIMIZERS_TRAINER = {'adamw_hf': 0.0001, 'adamw_torch': 0.0001, 'adamw_apex_fused': 0.0001, 'adafactor': 0.0001}
