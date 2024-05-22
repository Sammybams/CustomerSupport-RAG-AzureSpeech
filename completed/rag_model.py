import os

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

# Add OpenAI library
import openai

# Get Configuration Settings
from dotenv import load_dotenv
load_dotenv()

# Configure OpenAI API using Azure OpenAI
openai.api_key = os.getenv("API_KEY")
openai.api_base = os.getenv("ENDPOINT")
openai.api_type = "azure"  # Necessary for using the OpenAI library with Azure OpenAI
openai.api_version = "2023-05-15"  # Latest / target version of the API

import streamlit as st

# Define the model
def rag_respopnse():
    return