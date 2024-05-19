# Importing important package 

import os
import boto3
from langchain_community.embeddings import BedrockEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms.bedrock import Bedrock
from langchain.prompts import load_prompt

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
