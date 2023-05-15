import os
import zipfile
import shutil
from apikey import apikey
from pathlib import Path
import logging

import streamlit as st
from langchain.llms import OpenAI, LlamaCPP
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper, CodeRepositoryAnalyzer, PDFAnalyzer

os.environ['OPENAI_API_KEY'] = apikey

class ResearchEngine:
    """A class to handle the YouTube GPT Creator operations"""

    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    from langchain.utilities import WikipediaAPIWrapper

def __init__(self, llm_choice: str):
        self.llm_choice = llm_choice
        self.memory = self.ConversationBufferMemory(input_key='research_topic', memory_key='chat_history')

        if self.llm_choice == "OpenAI":
            self.llm = OpenAI(temperature=0.9)
        else:
            self.llm = LlamaCPP(temperature=0.9)

        self.prompt = self.PromptTemplate(
            input_variables=['research_topic', 'wikipedia_research', 'code_analysis', 'pdf_analysis'],
            template='Analyze and provide insights on the topic "{research_topic}" based on the following resources: Wikipedia research: {wikipedia_research}, Code analysis: {code_analysis}, PDF analysis: {pdf_analysis}'
        )

        self.chain = self.LLMChain(llm=self.llm, prompt=self.prompt, verbose=True, output_key='insights', memory=self.memory)
        self.wiki = self.WikipediaAPIWrapper()
        self.code_analyzer = CodeRepositoryAnalyzer()  # hypothetical analyzer
        self.pdf_analyzer = PDFAnalyzer()  # hypothetical analyzer

    def run(self, research_topic: str, code_repo_path: str, pdf_file_path: str):
        """Method to run the research engine"""
        wiki_research = self.wiki.run(research_topic)
        code_analysis = self.code_analyzer.run(code_repo_path)
        pdf_analysis = self.pdf_analyzer.run(pdf_file_path)

        insights = self.chain.run(research_topic=research_topic, wikipedia_research=wiki_research, code_analysis=code_analysis, pdf_analysis=pdf_analysis)
        return insights


# App framework
st.title('🦜🔗 YouTube GPT Creator')
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Choose LLM
llm_choice = st.selectbox("Choose LLM:", ("OpenAI", "LlamaCPP"))

# Show stuff to the screen if there's a PDF file
if pdf_file:
    creator = GPTCreator(llm_choice)

    # Load the PDF file using UnstructuredFileLoader
    loader = UnstructuredFileLoader(pdf_file, mode="elements")
    docs = loader.load()

    # Extract the content and convert it to a string
    content = ' '.join([doc.page_content for doc in docs])

    title, script, wiki_research = creator.run(content)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(creator.title_memory.buffer)

    with st.expander('Script History'):
        st.info(creator.script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)


# Logging configuration
logging.basicConfig(filename='app.log', level=logging.INFO)

# App framework
st.title('🦜🔗 YouTube GPT Creator')
zip_file = st.file_uploader("Upload a ZIP file", type=["zip"])

# Choose LLM
llm_choice = st.selectbox("Choose LLM:", ("OpenAI", "LlamaCPP"))

# Process the ZIP file
if zip_file:
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # Extract to a temporary directory
            temp_dir = Path('temp')
            temp_dir.mkdir(parents=True, exist_ok=True)
            zip_ref.extractall(temp_dir)

            # Process each file in the directory
            for file_path in temp_dir.glob('**/*'):
                if file_path.is_file():
                    # Load the PDF file using UnstructuredFileLoader
                    with file_path.open('rb') as file:
                        loader = UnstructuredFileLoader(file, mode="elements")
                        docs = loader.load()

                    # Extract the content and convert it to a string
                    content = ' '.join([doc.page_content for doc in docs])

                    creator = GPTCreator(llm_choice)
                    title, script, wiki_research = creator.run(content)

                    st.write(f'For file {file_path.name}:')
                    st.write(title)
                    st.write(script)

                    with st.expander(f'Title History ({file_path.name})'):
                        st.info(creator.title_memory.buffer)

                    with st.expander(f'Script History ({file_path.name})'):
                        st.info(creator.script_memory.buffer)

                    with st.expander(f'Wikipedia Research ({file_path.name})'):
                        st.info(wiki_research)

        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

    except zipfile.BadZipFile:
        st.error('Error: Bad zip file')
        logging.error('Bad zip file uploaded')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
        logging.error(f'An error occurred: {str(e)}')
