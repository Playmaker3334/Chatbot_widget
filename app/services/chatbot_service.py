import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from app.config import get_settings


class ChatbotService:
    def __init__(self):
        self.settings = get_settings()
        genai.configure(api_key=self.settings.google_api_key)
        
        self.llm = ChatGoogleGenerativeAI(
            model=self.settings.model_name,
            google_api_key=self.settings.google_api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        self.system_prompt = """You are an expert assistant specializing in exoplanets 
        and astronomical models. Provide clear, accurate, and educational responses 
        about exoplanets, their detection methods, characteristics, and related 
        scientific concepts."""
    
    async def get_response(self, user_message: str) -> str:
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = await self.llm.ainvoke(messages)
            return response.content
        
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")


chatbot_service = ChatbotService()