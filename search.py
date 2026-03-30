import re
from langchain_community.vectorstores import Chroma

# 토큰이 포함된 약물명 후보 오름차순-짧은순으로 정렬
def candidate_drugs(token: str, drug_names:list[str], top_n: int=10) -> list[str] :
    token = token.strip()
    cands = [d for d in drug_names if token in d]
    cands = sorted(cands, key=len)
    return cands[:top_n]

# 토큰 추출 후 가장 많은 약물명 후보와 매칭되는 토큰과 약물명 리스트 반환
def choose_best_token(
        user_query: str, 
        drug_names: list[str],
        top_n: int = 10,
) -> tuple[str | None, list[str], list[str]]:
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", user_query)
    tokens = [t for t in tokens if len(t) >= 2]

    best_token, best_cands = None, []
    for t in tokens :
        cands = candidate_drugs(t, drug_names, top_n=top_n)
        if len(cands) > len(best_cands) : 
            best_token, best_cands = t, cands

    return best_token, best_cands, tokens

# best토큰과 리스트만 반환
def get_candidates(
        user_query: str,
        drug_names: list[str],
        top_n: int = 10,
) -> tuple[str | None, list[str]]:
    token, cands, _ = choose_best_token(user_query, drug_names, top_n=top_n)
    return token, cands

# 검색기
def search_docs(
        user_query:str,
        vectorstore: Chroma,
        chosen_drug_name: str | None = None,
        k: int=3
):
    if chosen_drug_name:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k" : k, "filter" : {"drug_name" : chosen_drug_name}}
        )
    else :
        retriever = vectorstore.as_retriever(search_kwargs={"k" : k})

    return retriever
