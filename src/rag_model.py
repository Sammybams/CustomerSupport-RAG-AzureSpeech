import os

# Handling vector database
from langchain_chroma import Chroma
from langchain import PromptTemplate

# Add OpenAI library
import openai


# Get Configuration Settings
from dotenv import load_dotenv
load_dotenv()

# Configure OpenAI API using Azure OpenAI
openai.api_key = os.getenv("API_KEY")
openai.api_base = os.getenv("ENDPOINT")
openai.api_type = "azure"  # Necessary for using the OpenAI library with Azure OpenAI
openai.api_version = os.getenv("OPENAI_API_VERSION")  # Latest / target version of the API

from langchain.embeddings import OpenAIEmbeddings

# OpenAI Settings
model_deployment = "text-embedding-ada-002"
# SDK calls this "engine", but naming it "deployment_name" for clarity

model_name = "text-embedding-ada-002"

# Embeddings
openai_embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
    openai_api_version = os.getenv("OPENAI_API_VERSION"), openai_api_key = os.getenv("API_KEY"),
    openai_api_base = os.getenv("ENDPOINT"), openai_api_type = "azure"
)

# vector_store_contoso = Chroma(
#     collection_name="Contoso-Electronics-Docs",
#     embedding_function=openai_embeddings,
#     persist_directory="./Contoso-Electronics-Chroma-Vector-DB", # Where to save data locally, remove if not neccesary
# )

def conversation_history_prompt(history, question):
    # Define the template string for summarizing conversation history
    template_summary = """
    "Given a chat history (delimited by <hs></hs>) and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is.
    ------
    <hs>
    {history}
    </hs>
    ------
    Question: {question}
    Summary:
    """

    # Create a PromptTemplate object
    prompt = PromptTemplate(
        input_variables=["history", "question"],
        template=template_summary,
    )

    return prompt.format(history=history, question=question)

def get_conversation_summary(history, question):
    # Get the conversation summary prompt
    formatted_prompt = conversation_history_prompt(history, question)

    # Query the Azure OpenAI LLM with the formatted prompt
    response = openai.ChatCompletion.create(
        engine="Voicetask",  # Replace with your Azure OpenAI deployment name
        # prompt=formatted_prompt,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ],
        # max_tokens=50,
        temperature=0.5
    )
    
    # Extract and return the summary from the response
    return response.choices[0].message['content']
