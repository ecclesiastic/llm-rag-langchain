# Importing important package 

import os
import boto3
import torch

from langchain_community.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.bedrock import Bedrock
from langchain.prompts import load_prompt
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

# Create llm object for Amazon Bedrock

def create_llm(model_name):
    
    model_kwargs = {
        "maxTokenCount": 1024,
        "stopSequences": [], 
        "temperature": 0, 
        "topP": 0.9 
    }
    
    llm = Bedrock(
        credentials_profile_name="default",
        region_name='us-east-1',
        endpoint_url=os.environ.get("DEMO_ENDPOINT_URL"), 
        model_id=model_name,
        model_kwargs=model_kwargs) 
    
    return llm

def create_get_index():
    bedrock = boto3.client('bedrock-runtime')
    embeddings = BedrockEmbeddings(
        client=bedrock,
        credentials_profile_name="default", 
        region_name='us-east-1', 
        endpoint_url=os.environ.get("DEMO_ENDPOINT_URL"), 
    ) 
    
    pdf_path = "mysql_table_definition.pdf"

    loader = PyPDFLoader(file_path=pdf_path) 
    
    text_splitter = RecursiveCharacterTextSplitter( 
        separators=["\n\n", "\n", ".", " "], 
        chunk_size=1000, 
        chunk_overlap=100 
    )
    
    index_creator = VectorstoreIndexCreator( 
        vectorstore_cls=FAISS, 
        embedding=embeddings, 
        text_splitter=text_splitter, 
    )
    
    index_from_loader = index_creator.from_loaders([loader]) 
    
    return index_from_loader 
    
def call_rag_function(index, input_text): 

    model_name="amazon.titan-tg1-large"
    
    llm = create_llm(model_name)
    
    response_text = index.query(question=input_text, llm=llm) 
    
    return response_text


def call_bedrock(input_text):
    inference_modifier = {'max_tokens_to_sample': 5000,
                          "temperature": 0,
                          "top_k": 250,
                          "top_p": 1,
                          "stop_sequences": ["\n\nHuman"]
                          }

    textgen_llm = Bedrock(model_id="anthropic.claude-v2",
                          credentials_profile_name="default",
                          region_name='us-east-1',
                          model_kwargs=inference_modifier
                          )
    num_tokens = textgen_llm.get_num_tokens(input_text)
    print(f"Our prompt has {num_tokens} tokens")

    return textgen_llm(input_text)

def call_llama3(input_text):
    # Use local model
    local_model_path = "open_llama_3b_v2"
    local_tokenizer_path = "open_llama_3b_v2"
    tokenizer = AutoTokenizer.from_pretrained(local_tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path, device_map='sequential'
    )

    # Download Llama3 chinese fine tune model
    # model_id = "shenzhi-wang/Llama3-8B-Chinese-Chat"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, torch_dtype="auto", device_map="auto"
    # )

    # Prepare the input using the tokenizer
    input_ids = tokenizer(f"User: {input_text}\n", return_tensors="pt").input_ids.to("cpu")

    # Use if enough memory
    # input_ids = tokenizer.apply_chat_template(
    #     messages, add_generation_prompt=True, return_tensors="pt"
    # ).to(model.device)

    # Generate the output using the model
    outputs = model.generate(
        input_ids,
        max_new_tokens=128,
        temperature=0,
        low_memory=True,
    )

    # Decode the output and print the response
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))

