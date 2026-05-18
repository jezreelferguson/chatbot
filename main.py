from fastapi import FastAPI, HTTPException
from schemas.question import QuestionRequest
from ai.llm import ask_ai
app = FastAPI()

@app.get("/")
def welcome():
    return {"message": "Welcome to the chatbot API!"}

@app.post("/ask")
async def ask_question(req: QuestionRequest):

    try:
        answer = await ask_ai(req.question)

        return {
            "success": True,
            "question": req.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )