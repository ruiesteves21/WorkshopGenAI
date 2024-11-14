# Imports
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Set the API Key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI model with LangChain
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Function to read PDF file content
def read_pdf_file(file_path: str) -> str:
    # Load PDF content using UnstructuredPDFLoader
    pdf_loader = UnstructuredPDFLoader(file_path)
    docs = pdf_loader.load()
    # Assuming the first document contains the full content
    return docs[0].page_content

# Function to communicate with LLM (GPT model)
def talk_with_the_llm(llm, prompt: str) -> str:
    # Pass the prompt to the model and get the response
    response = llm.invoke([HumanMessage(content=prompt)]).content
    return response

# Path to your PDF file
FILE = 'GenerativeArtificialIntelligence-IEEEComputerSociety-221003.pdf'

# Read content from the PDF file
file_content = read_pdf_file(FILE)

# Write a detailed prompt that summarizes the recipe in the PDF
prompt = f"""
You are a Generative AI expert.

Your task is to read an article and summarize the text in order to
answer the following questions:

1. What are the GenAI trends?
2. How Generative AI and ethics are related?

Here is the text:

{file_content}
"""

# Get the summary from the LLM
summary = talk_with_the_llm(llm, prompt)

# Print the LLM response (which will contain the summary)
print(summary)
