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
    def __init__(self, llm_choice: str):
        self.llm_choice = llm_choice
        self.memory = ConversationBufferMemory(input_key='research_topic', memory_key='chat_history')

        self.llm = OpenAI(temperature=0.9) if self.llm_choice == "OpenAI" else LlamaCPP(temperature=0.9)

        self.prompt = PromptTemplate(
            input_variables=['research_topic', 'wikipedia_research', 'code_analysis', 'pdf_analysis'],
            template='Analyze and provide insights on the topic "{research_topic}" based on the following resources: Wikipedia research: {wikipedia_research}, Code analysis: {code_analysis}, PDF analysis: {pdf_analysis}'
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, verbose=True, output_key='insights', memory=self.memory)
        self.wiki = WikipediaAPIWrapper()
        self.code_analyzer = CodeRepositoryAnalyzer()
        self.pdf_analyzer = PDFAnalyzer()

    def run(self, research_topic: str, code_repo_path: str, pdf_file_path: str):
        wiki_research = self.wiki.run(research_topic)
        code_analysis = self.code_analyzer.run(code_repo_path)
        pdf_analysis = self.pdf_analyzer.run(pdf_file_path)

        insights = self.chain.run(research_topic=research_topic, wikipedia_research=wiki_research, code_analysis=code_analysis, pdf_analysis=pdf_analysis)
        return insights

@st.cache(allow_output_mutation=True)
def get_insights(research_topic, code_repo_path, pdf_file_path):
    creator = ResearchEngine(llm_choice)
    return creator.run(research_topic, code_repo_path, pdf_file_path)

def main():
    st.title('ResearchEngine')
    llm_choice = st.selectbox("Choose LLM:", ("OpenAI", "LlamaCPP"))
    zip_file = st.file_uploader("Upload a ZIP file", type=["zip"])

    if zip_file:
        process_zip_file(zip_file, llm_choice)

    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if pdf_file:
        process_pdf_file(pdf_file, llm_choice)

def process_zip_file(zip_file, llm_choice):
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            temp_dir = Path('temp')
            temp_dir.mkdir(parents=True, exist_ok=True)
            zip_ref.extractall(temp_dir)

            for file_path in temp_dir.glob('**/*'):
                if file_path.is_file():
                    with file_path.open('rb') as file:
                        process_file(file, llm_choice, file_path.name)

            shutil.rmtree(temp_dir)

    except zipfile.BadZipFile:
        st.error('Error: Bad zip file')
        logging.error('Bad zip file uploaded')
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')
        logging.error(f'An error occurred: {str(e)}')

def process_pdf_file(pdf_file, llm_choice):
    loader = UnstructuredFileLoader(pdf_file, mode="elements")
    docs = loader.load()

    content = ' '.join([doc.page_content for doc in docs])
    process_file(content, llm_choice, 'PDF file')

def process_file(file_content, llm_choice, file_name):
    creator = ResearchEngine(llm_choice)
    title, script, wiki_research = creator.run(file_content)

    st.write(f'For file {file_name}:')
    st.write(title)
    st.write(script)

    with st.expander(f'Title History ({file_name})'):
        st.info(creator.memory.buffer['title'])

    with st.expander(f'Script History ({file_name})'):
        st.info(creator.memory.buffer['script'])

    with st.expander(f'Wikipedia Research ({file_name})'):
        st.info(wiki_research)

if __name__ == "__main__":
    # Logging configuration
    logging.basicConfig(filename='app.log', level=logging.INFO)

    main()
