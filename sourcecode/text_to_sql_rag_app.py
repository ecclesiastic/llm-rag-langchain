# Importing important package 

import os
import streamlit as st 
import text_to_sql_rag_lib as glib
import yaml

from langchain.prompts import load_prompt, PromptTemplate
from pathlib import Path

st.set_page_config(page_title="文字生成Query Demo") #HTML title
st.title("文字生成Query Demo") #page title

# if 'vector_index' not in st.session_state:
#     with st.spinner("Indexing document..."):
#         st.session_state.vector_index = glib.create_get_index()
prompt = st.text_area("enter your query")

current_dir = Path(__file__)
root_dir = [p for p in current_dir.parents if p.parts[-1] == 'llm-rag-langchain'][0]
prompt_file_path = f"{root_dir}/prompt/table_schema_prompt.yaml"
with open(prompt_file_path, 'r', encoding='utf-8') as file:
    file_content = yaml.safe_load(file)
# Assuming load_prompt can accept a dictionary, pass the loaded YAML content
# If load_prompt only accepts file paths, directly use the path
# prompt_template = load_prompt(file_content)  # If it accepts content
# prompt_template = load_prompt(prompt_file_path)  # If it accepts file path
# Assuming the YAML content contains the necessary fields for PromptTemplate
prompt_template = PromptTemplate(
    input_variables=file_content.get('input_variables'),
    template=file_content.get('template'),
    output_parser=file_content.get('output_parser', None)  # Assuming output_parser is optional
)
final_prompt = prompt_template.format(input=prompt)
go_button = st.button("Go", type="primary")
response_content = None
if go_button: 
    
    with st.spinner("Evaluating..."): 
        # response_content = glib.call_rag_function(index=st.session_state.vector_index, input_text=input_text) #call the model through the supporting library
        response_content = glib.call_llama3(final_prompt)
        st.write(response_content) 

continue_button = st.button("Continue", type="primary")
if continue_button:
    with st.spinner("Continue Evaluating..."):
        template = (
            "Combine the chat history, previous answer, and continue generate answer"
            "Chat History: {} "
            "Previous answer: {} "
        )
        prompt = template.format(final_prompt, response_content)
        response_content = glib.call_llama3(prompt)
        st.write(response_content)