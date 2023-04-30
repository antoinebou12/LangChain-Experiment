import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI, LlamaCPP
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.document_loaders import UnstructuredFileLoader

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 YouTube GPT Creator')
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Choose LLM
llm_choice = st.selectbox("Choose LLM:", ("OpenAI", "LlamaCPP"))

# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Write me a YouTube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='Write me a YouTube video script based on this title TITLE: {title} while leveraging this Wikipedia research: {wikipedia_research}'
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Llms
if llm_choice == "OpenAI":
    llm = OpenAI(temperature=0.9)
else:
    llm = LlamaCPP(temperature=0.9)

title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a PDF file
if pdf_file:
    # Load the PDF file using UnstructuredFileLoader
    loader = UnstructuredFileLoader(pdf_file, mode="elements")
    docs = loader.load()

    # Extract the content and convert it to a string
    content = ' '.join([doc.page_content for doc in docs])

    title = title_chain.run(content)
    wiki_research = wiki.run(content)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
