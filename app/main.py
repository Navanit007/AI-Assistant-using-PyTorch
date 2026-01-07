from fastapi import FastAPI
from pydantic import BaseModel
from .chat import ChatBot

app = FastAPI(title="AI Assistant")

bot = ChatBot()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {"status": "AI Assistant is running"}

@app.post("/chat")
def chat(req: ChatRequest):
    reply = bot.generate_response(req.message)
    return {"response": reply}
