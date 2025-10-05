from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.services.chatbot_service import chatbot_service
from app.config import get_settings
import os

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="API for Exoplanet Chatbot with LangChain and Gemini",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(static_path, "index.html"))


@app.post("/api/chat", response_model=ChatResponse, responses={500: {"model": ErrorResponse}})
async def chat_endpoint(request: ChatRequest):
    try:
        response_text = await chatbot_service.get_response(request.message)
        return ChatResponse(response=response_text, status="success")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e), "status": "error"}
        )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.app_name}