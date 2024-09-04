# Streamlit
import streamlit as st
st.set_page_config(page_title="Customer Support Solution", page_icon="")

import os

# Langchain components
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Add OpenAI library
import openai

# Get Configuration Settings
from dotenv import load_dotenv
load_dotenv()

# Configure OpenAI API using Azure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("ENDPOINT")
openai.api_type = "azure"  # Necessary for using the OpenAI library with Azure OpenAI
openai.api_version = "2024-02-01"  # Latest / target version of the API

# Implementation
from langchain.embeddings import OpenAIEmbeddings

# OpenAI Settings
model_deployment = "text-embedding-ada-002"
# SDK calls this "engine", but naming it "deployment_name" for clarity

model_name = "text-embedding-ada-002"

openai_embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
    openai_api_version = os.getenv("OPENAI_API_VERSION"), openai_api_key = os.getenv("API_KEY"),
    openai_api_base = os.getenv("ENDPOINT"), openai_api_type = "azure"
)

from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory, ConversationBufferWindowMemory
from langchain import PromptTemplate

# Prompt Template
template_j = """
Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the user's question. 
All bible texts are to referenced with King James Version. You can give info on bible texts.
If you don't know the answer, just say that you don't know, don't try to make up an answer. Refer to the context provided as "the Junior Sunday school lessons".
------
<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template_j,
)

# Retrieval QA
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

import langchain
langchain.verbose = False

# LLM - Azure OpenAI
llm = ChatOpenAI(temperature = 0.3, openai_api_key = os.getenv("API_KEY"), openai_api_base = os.getenv("ENDPOINT"), model_name="gpt-35-turbo", engine="Voicetask")

# Loading Junior Lessons
from langchain_chroma import Chroma

# Junior
vector_store_junior = Chroma(
    collection_name="Junior_Lessons",
    embedding_function=openai_embeddings,
    persist_directory="/Users/mac/Documents/Gospel-Companion/chroma_afc_sunday_school_lessons_db",  # Where to save data locally, remove if not neccesary
)

# QA Model

retriever_j = vector_store_junior.as_retriever(search_kwargs={'k': 3})

# Initialize session state for qa_stuff
if 'qa_stuff' not in st.session_state:
    st.session_state.qa_stuff = RetrievalQA.from_chain_type(
                                    llm = llm, 
                                    chain_type = "stuff", 
                                    retriever = retriever_j, 
                                    verbose = False,
                                    chain_type_kwargs = {
                                        "verbose": True,
                                        "prompt": prompt,
                                        "memory": ConversationBufferWindowMemory(
                                                k = 5,
                                                memory_key = "history",
                                                input_key = "question")
                                                }
                                    )

# Chatbot interface using streamlit and maintain chat history

def chatbot_interface():
    # st.title("Chat with the OpenAI Chatbot")
    # st.write("This is a simple chatbot interface using OpenAI's GPT-3 model.")
    # st.write("Please type your message in the chat box below and press Enter to send it to the chatbot.")
    # st.write("The chatbot will respond with a message based on the input you provide.")

    # # Initialize the chatbot model
    # chatbot = ChatOpenAI()

    # Initialize session state for chat history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Chatbot interface

    with st.container():
        # messages = st.container(height=1000)
        # Display chat history 
        for chat in st.session_state.history:
            st.chat_message("user").write(chat["user"])
            st.chat_message("assistant").write(chat["assistant"])
            # messages.chat_message("assistant").write(response)

    if prompt := st.chat_input("Message Gospel Companion"):
        # messages.chat_message("user").write(prompt)
        st.chat_message("user").write(prompt)
        response = st.session_state.qa_stuff.run(prompt)
        st.chat_message("assistant").write(response)
        st.session_state.history.append({"user": prompt, "assistant": response})
        st.rerun()
        
# Run the chatbot interface
if __name__ == "__main__":
    chatbot_interface()