from openai import OpenAI
from prompt import SYSTEM_MESSAGE_TEMPLATE

def format_docs(docs, max_chars_per_doc: int=1200, include_meta: bool=True) -> str:
    if not docs :
        return "관련 문서를 찾지 못했습니다"
    
    formatted_results = []
    for i, doc in enumerate(docs, 1) : 
        text = (doc.page_content or "").strip()
        if len(text) > max_chars_per_doc :
            text = text[:max_chars_per_doc] + "...[truncated]"
        
        if include_meta : 
            m = doc.metadata or {}
            header = (
                f"문서{i} | "
                f"drug_name={m.get('drug_name')} | "
                f"section={m.get('section')} | "
                f"chung_id={m.get('chunk_id')}"
            )

            formatted_results.append(f"{header}\n{text}")
        else:
            formatted_results.append(f"문서{i} : {text}")
        
    return "\n---\n".join(formatted_results)


# 검색-포맷팅-프롬프트 조립 종합 함수
def generate_rag_response(
        question: str, 
        retriever,
        openai_client: OpenAI,
        openai_model: str = "gpt-4o-mini"
)-> tuple[str, str, list]:
    
    # 1. 검색
    retrieved_docs = retriever.invoke(question)

    # 2. 문서 포맷팅
    formatted_results = format_docs(retrieved_docs)

    # 3. 프롬프트 조립
    formatted_system = SYSTEM_MESSAGE_TEMPLATE.format(search_result=formatted_results)
    messages=[
        {"role" : "system", "content" : formatted_system},
        {"role" : "user", "content" : question} 
    ]

    # 4. openai api 호출
    completion = openai_client.chat.completions.create(
        model=openai_model,
        messages=messages,
        temperature=0,
        max_tokens=1024
    )

    response = completion.choices[0].message.content

    return response, formatted_results, retrieved_docs