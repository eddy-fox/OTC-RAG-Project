from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from model import init_all
from search import get_candidates, search_docs
from rag_engine import generate_rag_response

load_dotenv()

# 요청-응답

class CandidatesRequest(BaseModel):
    question: str
    top_n : int = 10

class CandidatesResponse(BaseModel):
    token: str | None
    candidates: list[str]

class AnswerRequest(BaseModel):
    question : str
    chosen_drug_name: str | None
    k: int = 3

class AnswerResponse(BaseModel):
    answer: str
    question: str
    drug: str | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = init_all()
    app.state.vectorstore = state["vectorstore"]
    app.state.drug_names = state["drug_names"]
    app.state.openai_client = state["openai_client"]
    app.state.openai_model = state["openai_model"]
    yield

# fastapi
app = FastAPI(
    title="일반의약품(OTC) RAG API",
    description = "일반의약품 복용/주의사항 근거기반 QA시스템",
    version = "1.0.0",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# 서버상태확인
@app.get("/health")
def health():
    return {
        "status" : "ok",
        "drug_count" : len(app.state.drug_names),
        "chunk_count" : app.state.vectorstore._collection.count()
    }

# 1. 약물명 후보 변환
@app.post("/candidates", response_model=CandidatesResponse)
def candidates(req: CandidatesRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요")
    
    token, cands = get_candidates(
        user_query=req.question,
        drug_names=app.state.drug_names,
        top_n=req.top_n
    )

    return CandidatesResponse(token=token, candidates=cands)

# 2. RAG답변 생성
@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="질문을 입력해주세요")
    
    # 검색
    retriever = search_docs(
        user_query=req.question,
        vectorstore=app.state.vectorstore,
        chosen_drug_name=req.chosen_drug_name,
        k=req.k
    )

    # 생성
    try:
        response, _, _ = generate_rag_response(
            question=req.question,
            retriever=retriever,
            openai_client=app.state.openai_client,
            openai_model=app.state.openai_model
        )
    except Exception as e :
        raise HTTPException(status_code=500, detail=f"답변 생성 오류: {str(e)}")
    
    return AnswerResponse(
        answer=response,
        question=req.question,
        drug=req.chosen_drug_name
    )