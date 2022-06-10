import streamlit as st

from components import (show_API_component, show_code_component,
                        show_datset_view_component, show_input_data_component,
                        show_model_component, show_preprocessing_component,
                        show_task_component, show_training_comoponent)

st.set_page_config(
    page_title="Training Code Generator for Hugging Face Models ",  layout="wide"
)

st.markdown("<br>", unsafe_allow_html=True)

"""
# Training Code Generator for Hugging Face Models 🤗 
"""
st.markdown("<br>", unsafe_allow_html=True)
"""
---
"""

inputs = {}

with st.sidebar:
    st.info(
        "**Select the configuration**"
    )
    inputs = show_API_component(inputs)
    inputs = show_task_component(inputs)
    inputs = show_model_component(inputs)
    inputs = show_input_data_component(inputs)
    inputs = show_preprocessing_component(inputs)
    inputs = show_training_comoponent(inputs)

inputs = show_datset_view_component(inputs)
inputs = show_code_component(inputs)