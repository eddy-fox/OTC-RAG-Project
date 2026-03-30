import os
import shutil

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from openai import OpenAI
from huggingface_hub import snapshot_download

# 설정값
EMBED_MODEL_NAME = "BAAI/bge-m3"
OPENAI_MODEL_NAME = "gpt-4o-mini"
CHROMA_REPO_ID = "eddyfox8812/otc-chroma-db"
CHROMA_LOCAL_DIR = os.path.abspath("./chroma_otc")
COLLECTION_NAME = "otc_chunks"

# 임베딩 모델
def load_embeddings():
    device = "cuda" if _is_cuda_available() else "cpu"
    return HuggingFaceBgeEmbeddings(
        model_name = EMBED_MODEL_NAME,
        model_kwargs={"device" : device},
        encode_kwargs={"normalize_embeddings" : True}
    )

def _is_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
    
# 벡터DB
def load_vectorstore(embeddings):

    if not os.path.exists(CHROMA_LOCAL_DIR):
        print(f"CHROMA DB 다운로드중 : {CHROMA_REPO_ID}")
        local_repo = snapshot_download(repo_id=CHROMA_REPO_ID, repo_type="dataset")
        src = os.path.join(local_repo, "chroma_otc")
        os.makedirs(os.path.dirname(CHROMA_LOCAL_DIR), exist_ok=True)
        shutil.copytree(src, CHROMA_LOCAL_DIR)
        print(f"복사완료 : {CHROMA_LOCAL_DIR}")

    vectorstore = Chroma(
        persist_directory=CHROMA_LOCAL_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    print(f"Chroma 로드 완료 - 총 청크 수 : {vectorstore._collection.count()}")
    return vectorstore

# 약물명 리스트
def load_drug_names(vectorstore) -> list[str]:
    col = vectorstore._collection
    n = col.count()
    drug_set = set()
    batch = 5000

    for offset in range(0, n, batch): 
        out = col.get(include=["metadatas"], limit=batch, offset=offset)
        for m in out["metadatas"] :
            if m and m.get("drug_name"):
                drug_set.add(m["drug_name"])
    
    drug_names = sorted(drug_set)
    print(f"약물명 리스트 생성 완료 - 총 {len(drug_names)}개")
    return drug_names

# openai 클라이언트
def load_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key :
        raise EnvironmentError(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            ".env 파일을 확인하세요."
        )
    return OpenAI(api_key=api_key)

# 전체 초기화
def init_all() -> dict:
    print("===서버 초기화 시작===")
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)
    drug_names = load_drug_names(vectorstore)
    openai_client = load_openai_client()
    print("===서버 초기화 완료===")
    
    return {
        "vectorstore" : vectorstore,
        "drug_names" : drug_names,
        "openai_client" : openai_client,
        "openai_model" : OPENAI_MODEL_NAME
    }