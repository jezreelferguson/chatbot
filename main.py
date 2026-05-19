from fastapi import FastAPI, HTTPException
from schemas.question import QuestionRequest
from ai.llm import ask_ai
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

@app.get("/")
def welcome():
    return {"message": "Welcome to the chatbot API!"}

@app.post("/ask")
async def ask_question(req: QuestionRequest):

    try:
        answer = await ask_ai(req.question, req.session_id)

        return {
            "success": True,
            "question": req.question,
            "session_id": req.session_id,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )