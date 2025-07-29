import os
from langchain.chat_models import init_chat_model

os.environ["GOOGLE_API_KEY"] = "..."

BestCodingAgent = "anthropic:claude-4-sonnet-latest"

CodingAgent = init_chat_model("google_genai:gemini-2.0-flash")