"""환경 변수 및 설정"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENFDA_API_KEY = os.getenv("OPENFDA_API")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

# OpenFDA Configuration
OPENFDA_BASE_URL = "https://api.fda.gov/drug"
OPENFDA_LABEL_ENDPOINT = "/label.json"

# Search Configuration
SEARCH_LIMIT = 5

# LLM Configuration
CLASSIFIER_MODEL = "gpt-4.1-nano"
LLM_MODEL = "gpt-4.1-mini"
LLM_TEMPERATURE = 0.0

# LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = "openfda-drug-info-rag"
