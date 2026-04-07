from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from dotenv import load_dotenv
import os

load_dotenv()

llm = AzureAIOpenAIApiChatModel(
    endpoint="https://models.inference.ai.azure.com",
    credential=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o"   
)

print(llm.invoke("Xin chào?").content)
