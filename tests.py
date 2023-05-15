import pytest
from langchain.llms import OpenAI, LlamaCPP
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper, CodeRepositoryAnalyzer, PDFAnalyzer

from app import ResearchEngine

# Mock the API wrappers to isolate the unit tests
@pytest.fixture(autouse=True)
def mock_wrappers(mocker):
    mocker.patch.object(WikipediaAPIWrapper, 'run', return_value='wikipedia_result')
    mocker.patch.object(CodeRepositoryAnalyzer, 'run', return_value='code_result')
    mocker.patch.object(PDFAnalyzer, 'run', return_value='pdf_result')

# Mock the LLMs
@pytest.fixture(params=[OpenAI, LlamaCPP])
def mock_llm(mocker, request):
    return mocker.patch.object(request.param, '__init__', return_value=None)

# Mock the ConversationBufferMemory
@pytest.fixture
def mock_memory(mocker):
    return mocker.patch.object(ConversationBufferMemory, '__init__', return_value=None)

# Mock the LLMChain
@pytest.fixture
def mock_chain(mocker):
    return mocker.patch.object(LLMChain, 'run', return_value='insights')


def test_research_engine_run(mock_llm, mock_memory, mock_chain):
    engine = ResearchEngine('OpenAI')

    topic = 'test_topic'
    repo_path = 'test_repo_path'
    pdf_path = 'test_pdf_path'

    insights = engine.run(topic, repo_path, pdf_path)

    assert insights == 'insights'
    mock_llm.assert_called_once_with(temperature=0.9)
    mock_memory.assert_called_once_with(input_key='research_topic', memory_key='chat_history')
    mock_chain.assert_called_once()
