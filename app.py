import streamlit as st
import os
import pandas as pd # pandas는 직접 사용되지 않지만, 일반적인 데이터 처리 라이브러리로 유지합니다.
import sqlite3
import datetime

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.callbacks import get_openai_callback

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    CSVLoader
)

# --- SQLite 로깅 설정 추가 ---
DB_NAME = 'token_usage.sqlite'
# USD/KRW 환율 - 예시 값입니다. 실제 환율로 업데이트하거나, API를 통해 동적으로 가져오도록 수정할 수 있습니다.
USD_TO_KRW_EXCHANGE_RATE = 1370.00 # 예시: 1 USD = 1370 KRW

def init_db():
    """데이터베이스와 테이블을 초기화하고, 필요한 경우 스키마를 업데이트합니다."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # 테이블 생성 (이미 존재하면 생성 안 함)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            business_name TEXT, 
            model_name TEXT NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER NOT NULL,
            cost_usd REAL,
            cost_krw REAL,
            api_call_tag TEXT DEFAULT NULL
        )
    ''')

    # business_name 컬럼이 없는 경우 추가 (기존 DB 호환성)
    cursor.execute("PRAGMA table_info(usage_logs)")
    columns = [info[1] for info in cursor.fetchall()]
    if 'business_name' not in columns:
        try:
            cursor.execute("ALTER TABLE usage_logs ADD COLUMN business_name TEXT")
            print("INFO: 'business_name' column added to 'usage_logs' table.")
        except sqlite3.OperationalError as e:
            # 이미 컬럼이 추가되었거나 다른 문제 발생 시 (예: 동시성 문제)
            print(f"WARNING: Could not add 'business_name' column, it might already exist or another issue occurred: {e}")
    
    conn.commit()
    conn.close()

def log_token_usage(business_name, model_name, prompt_tokens, completion_tokens, total_tokens, cost_usd, api_call_tag=None):
    """토큰 사용량 및 비용을 SQLite 데이터베이스에 기록합니다."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    pt = prompt_tokens if prompt_tokens is not None and prompt_tokens >= 0 else None
    ct = completion_tokens if completion_tokens is not None and completion_tokens >= 0 else None
    tt = total_tokens if total_tokens is not None else 0

    cost_krw = None
    if cost_usd is not None:
        cost_krw = cost_usd * USD_TO_KRW_EXCHANGE_RATE
        cost_krw = round(cost_krw, 4) 

    try:
        cursor.execute('''
            INSERT INTO usage_logs (business_name, model_name, prompt_tokens, completion_tokens, total_tokens, cost_usd, cost_krw, api_call_tag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (business_name, model_name, pt, ct, tt, cost_usd, cost_krw, api_call_tag))
        conn.commit()
        # print(f"Logged to DB: Business: {business_name}, Tag: {api_call_tag}, Model: {model_name}, Tokens: {tt}, Cost KRW: {cost_krw}")
    except sqlite3.Error as e:
        print(f"SQLite 로깅 오류: {e}") 
    finally:
        conn.close()

# --- End of SQLite 로깅 설정 ---


# --- 0. OpenAI API 키 설정 ---
# OPENAI_API_KEY_INPUT은 UI를 통해 사용자가 입력합니다.

# --- 사업별 회계 지침 파일 경로 ---
GUIDELINE_FILES = {
    "경기 사경": "fixed_accounting_guideline.txt",
    "해양수산": "ocean.txt",
    "25-26 KOICA CTS": "CTS.txt"
}
# 기본값으로 사용할 지침 (예: 첫 번째 항목)
DEFAULT_BUSINESS_NAME = list(GUIDELINE_FILES.keys())[0]


# --- Helper Functions ---
@st.cache_resource
def get_embeddings_model(_api_key):
    if not _api_key:
        return None
    try:
        return OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=_api_key)
    except Exception as e:
        st.error(f"임베딩 모델 로딩 실패 (get_embeddings_model): {e}")
        return None

@st.cache_resource
def get_llm(_api_key):
    if not _api_key:
        return None
    try:
        return ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=_api_key)
    except Exception as e:
        st.error(f"LLM 로딩 실패 (get_llm): {e}")
        return None

def load_document_from_local_path(file_path: str, uploaded_file_name_for_source: str = None):
    """로컬 경로의 파일을 로드 (고정 지침 파일 로드용 + 업로드 파일 처리용으로 통합 가능)"""
    if not os.path.exists(file_path):
        st.error(f"파일을 찾을 수 없습니다: {file_path}")
        return None
    
    file_extension = os.path.splitext(file_path)[1].lower()
    loader = None
    source_name = uploaded_file_name_for_source if uploaded_file_name_for_source else os.path.basename(file_path)
    
    st.info(f"파일 로딩 시도: {source_name} (형식: {file_extension})")

    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension in [".xlsx", ".xls"]:
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    elif file_extension in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
    elif file_extension == ".csv":
        loader = CSVLoader(file_path=file_path, encoding="utf-8")
    elif file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        st.error(f"지원하지 않는 파일 형식입니다: {file_extension} ({source_name})")
        return None

    documents = None
    try:
        documents = loader.load()
        if not documents:
            st.write(f"loader.load() 결과가 비어있거나 None입니다. ({source_name})")
        st.success(f"'{source_name}' 파일 로딩 완료 (1차 시도): {len(documents) if documents else 0}개의 Document 객체 생성")
        for doc in documents:
            doc.metadata['source'] = source_name
        return documents
    except UnicodeDecodeError as ude:
        st.warning(f"UTF-8 인코딩으로 '{source_name}' 로딩 실패: {ude}. CP949로 재시도...")
        if file_extension == ".csv" or file_extension == ".txt":
            try:
                if file_extension == ".csv": loader = CSVLoader(file_path=file_path, encoding="cp949")
                else: loader = TextLoader(file_path, encoding="cp949")
                documents = loader.load()
                st.success(f"'{source_name}' 파일 로딩 완료 (CP949 재시도): {len(documents) if documents else 0}개 Document 생성")
                for doc in documents: doc.metadata['source'] = source_name
                return documents
            except Exception as e2:
                st.error(f"CP949 인코딩으로도 '{source_name}' 로딩 실패: {e2}")
                return None
        else: return None
    except Exception as e:
        st.error(f"'{source_name}' 파일 로딩 중 (일반) 오류 발생: {e}")
        return None

def save_uploaded_file_to_temp(uploaded_file):
    """업로드된 파일을 임시 디렉토리에 저장하고 경로를 반환합니다."""
    if uploaded_file is None: return None
    temp_dir = "temp_uploaded_files"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_file_path


# --- Guideline Specific Resource Builders ---
@st.cache_resource(show_spinner="선택된 지침 문서 처리 중...")
def get_cached_guideline_vector_store(_embeddings_model, guideline_path, guideline_name):
    """선택된 지침 파일로부터 벡터 저장소를 구축하고 캐시합니다."""
    if not _embeddings_model:
        st.error("임베딩 모델이 로드되지 않아 지침 벡터 저장소를 구축할 수 없습니다.")
        return None
    if not os.path.exists(guideline_path):
        st.error(f"'{guideline_name}' 지침 파일을 찾을 수 없습니다: {guideline_path}")
        return None
    
    st.info(f"지침 파일 로딩 시도: {guideline_name} ({guideline_path})")
    guideline_documents = load_document_from_local_path(guideline_path, guideline_name)
    if not guideline_documents:
        st.error(f"'{guideline_name}' 지침 문서 로딩에 실패했습니다.")
        return None

    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
        )
        chunked_guideline_docs = text_splitter.split_documents(guideline_documents)
        if not chunked_guideline_docs:
            st.warning(f"'{guideline_name}' 지침 문서가 청크로 분할되지 않았습니다.")
            return None
        
        st.info(f"'{guideline_name}' 지침 문서: {len(guideline_documents)}개 원본 Doc -> {len(chunked_guideline_docs)}개 청크 분할 완료.")
        vector_store = FAISS.from_documents(chunked_guideline_docs, _embeddings_model)
        st.success(f"'{guideline_name}' 지침 벡터 저장소 생성이 완료되었습니다.")
        return vector_store
    except Exception as e:
        st.error(f"'{guideline_name}' 지침 벡터 저장소 구축 실패: {e}")
        return None

@st.cache_resource 
def get_rag_chain_for_guideline(_guideline_vector_store, _llm, guideline_name_for_prompt):
    """주어진 지침 벡터 저장소와 LLM을 사용하여 RAG 체인을 생성하고 캐시합니다."""
    if not _guideline_vector_store or not _llm: return None
    retriever = _guideline_vector_store.as_retriever(search_kwargs={'k': 5})
    # --- MODIFIED PROMPT FOR EMPHASIS ---
    prompt_template_str = f"""
    귀하는 "{guideline_name_for_prompt}" (이하 "본 가이드라인")에 대해 세계 최고 수준의 이해도와 적용 능력을 갖춘 AI 주임 컨설턴트입니다. 귀하의 주된 임무는 사용자가 제공하는 모든 형태의 입력(문서 내용 발췌, 특정 상황 기술, 직접적인 질문, 사업 계획 초안 등, 이하 "사용자 제시 내용")을 본 가이드라인의 조항, 기본 원칙, 목적, 그리고 숨겨진 함의까지 고려하여 입체적이고 심층적으로 분석하고, 그 결과를 사용자에게 전달하는 것입니다. 귀하는 단순 정보 전달자가 아닌, 사용자의 성공적인 가이드라인 준수를 돕고 잠재적 위험으로부터 보호하며, 최적의 의사결정을 지원하는 신뢰할 수 있는 핵심 조력자입니다. 모든 답변은 반드시 한국어로, 극도의 상세함과 명확성, 그리고 사용자 중심의 친절함을 담아 제공되어야 합니다.

    [핵심 수행 지침 및 분석 프레임워크]

    I. 사용자 제시 내용 분석 및 의도 파악 단계:
    1. 입력 유형 식별: 사용자 제시 내용이 다음 중 어떤 유형에 해당하는지 정확히 판단하십시오.
        가. 특정 문서/텍스트/계획안에 대한 본 가이드라인 부합 여부 검토 요청
        나. 특정 가상 또는 실제 상황에 대한 본 가이드라인 적용 및 해석 질의
        다. 본 가이드라인 특정 조항 또는 전체 내용에 대한 직접적인 설명/해석 요청
        라. 본 가이드라인 관련 잠재적 문제점 또는 기회에 대한 자문 요청
        마. 기타 (구체적으로 명시)
    2. 핵심 쟁점 도출: 사용자 제시 내용에서 본 가이드라인과 관련하여 검토가 필요한 핵심 쟁점, 문제점, 또는 질문사항을 명확하게 추출하고 요약하십시오. 사용자가 명시적으로 질문하지 않았더라도, 내용상 본 가이드라인과 관련하여 중요하게 다뤄져야 할 부분이 있다면 이를 포함해야 합니다.

    II. 본 가이드라인 기반 심층 검토 및 분석 단계:
    3. 조항 매칭 및 해석의 정확성:
        - 사용자 제시 내용과 관련된 본 가이드라인의 모든 조항(주요 조항, 하위 조항, 별표, 부칙 등 포함)을 정확히 찾아내고 목록화하십시오.
        - 각 조항의 문언적 의미를 최우선으로 하되, 필요한 경우 해당 조항의 입법 취지, 본 가이드라인 전체의 체계적 지위, 유관 조항과의 관계, 그리고 과거 유사 사례(존재한다면)까지 종합적으로 고려하여 가장 합리적이고 일관된 해석을 도출하십시오.
        - 해석의 여지가 있는 모호한 조항에 대해서는 가능한 모든 해석과 각 해석에 따른 결과를 제시하고, 그중 가장 안전하거나 권장되는 해석을 명시하십시오.
    4. 부합/위배 여부 판단 및 상세 근거 제시:
        - 각 쟁점별로 사용자 제시 내용이 본 가이드라인의 관련 조항에 명확히 부합하는지, 위배되는지, 또는 위배의 소지가 있는지를 명확히 판정하십시오.
        - 모든 판단에는 반드시 본 가이드라인 텍스트 내 **정확한 조항 번호(예: 제X조 제Y항 제Z호)와 해당 조항의 내용을 직접 인용**하여 구체적인 근거로 제시해야 합니다. (예: "[관련 지침 근거: {guideline_name_for_prompt} 제X조 제Y항 - '...조항 내용 전문 또는 핵심 부분 인용...']")
        - 단순히 조항 번호만 언급하는 것이 아니라, 해당 조항이 왜 이 사안에 적용되는지, 그리고 그 조항에 따라 왜 그러한 판단을 내렸는지에 대한 상세한 논리적 설명을 포함해야 합니다.
    5. 잠재적 위험 요소 및 주의사항 도출 (선제적 리스크 관리):
        - 사용자 제시 내용이 현재는 명시적으로 본 가이드라인에 위배되지 않더라도, 향후 상황 변화나 조건 추가에 따라 위배될 가능성이 있거나, 본 가이드라인의 정신에 부합하지 않을 수 있는 모든 잠재적 위험 요소 및 주의사항을 철저히 식별하여 경고해야 합니다.
        - "만약 ~한다면", "특히 ~의 경우", "~을 간과할 경우" 등의 표현을 사용하여 구체적인 상황을 가정하고, 발생 가능한 문제점과 그로 인한 부정적 결과를 명확히 설명하십시오.
    6. 보수적 검토 시나리오 의무 제시:
        - 모든 검토 요청에 대해, 가장 엄격한 기준 또는 보수적인 관점에서 본 가이드라인을 적용했을 경우 발생할 수 있는 최악의 시나리오와 그 결과를 반드시 포함하여 설명해야 합니다. (예: "**[가장 보수적인 관점에서의 검토 의견]** 현재 제시된 내용이 A조건을 충족하는 것으로 보이나, 만약 감독기관이 B라는 추가 자료를 요구하며 A조건의 충족 여부를 더욱 엄격하게 심사할 경우, 본 가이드라인 제Y조 Z항의 취지에 따라 '조건 미비'로 판단될 가능성을 배제할 수 없습니다. 이 경우, <font color='red'>**사업 승인 반려 또는 기 지원금 환수 등의 불이익이 발생할 수 있습니다.**</font>")
        - 이는 사용자가 예상치 못한 불이익을 방지하고, 모든 가능성을 염두에 둔 안전한 의사결정을 내릴 수 있도록 지원하기 위함입니다.

    III. 특수 조건 적용 단계 (해당 시):
    7. "25-26 KOICA CTS" 가이드라인 (즉, "CTS.txt" 파일) 특수 처리:
        - 만약 현재 적용되는 `guideline_name_for_prompt`가 **"25-26 KOICA CTS"** (즉, GUIDELINE_FILES 딕셔너리에서 "CTS.txt" 파일을 가리키는 키)로 지정된 경우, 귀하는 다음의 추가 지침을 최우선적으로 수행해야 합니다:
            - 귀하는 "CTS.txt" 문서의 **1799번째 줄부터 시작되는 것으로 가정되는 '[ FAQ ]' 섹션의 모든 내용을 완벽하게 숙지**하고 있다고 가정합니다. (실제 줄 번호는 파일 내용에 따라 다를 수 있으므로, 해당 표식이 있는 부분부터를 의미합니다.)
            - 사용자 제시 내용 또는 질문이 해당 FAQ 섹션의 사례와 조금이라도 관련성이 있다고 판단될 경우, **반드시 해당 FAQ의 질문과 답변 내용을 명시적으로 언급하고, 현재 사안과의 유사점, 차이점, 그리고 시사점을 상세히 비교 분석하여 설명**해야 합니다. (예: "**[CTS.txt FAQ 사례 연관 검토]** 본 사안은 CTS.txt FAQ 중 'Q. [FAQ 질문 요약]' 사례와 관련성이 높습니다. 해당 FAQ에서는 '[FAQ 답변 요약]'으로 안내하고 있습니다. 귀하의 상황은 [유사점/차이점 설명] 측면에서 해당 사례를 참고하여 [구체적 조언 또는 해석]을 고려해볼 수 있습니다.")
            - FAQ 내용을 단순히 전달하는 것을 넘어, 현재 사용자 상황에 맞게 재해석하고 적용하여 실질적인 도움을 제공해야 합니다.
    8. TIPS 프로그램 관련 사전 고지 의무:
        - 사용자 제시 내용에서 "CTS-TIPS 연계형 사업" 지원 또는 참여와 관련된 맥락이 조금이라도 감지될 경우 (사용자가 직접 질문하지 않더라도), 귀하는 다음의 내용을 반드시 명확하고 강조하여 선제적으로 안내해야 합니다:
            - "**[TIPS 프로그램 관련 중요 고지]** CTS-TIPS 연계형 사업 지원 자격과 관련하여 매우 중요한 점을 안내해 드립니다. <font color='red'>**TIPS 프로그램의 '성공' 판정은 공식적으로 해당 TIPS 과제의 '종료'를 의미합니다.**</font> 따라서, 만약 귀하(또는 귀사)가 현재 TIPS 프로그램에 참여 중이시라면, CTS-TIPS 연계형 사업의 공모 지원 마지막 날(마감일)까지 현재 수행 중인 TIPS 과제가 공식적으로 '종료(성공)' 처리되지 않은 상태라면, 안타깝게도 CTS-TIPS 연계형 사업의 지원 자격 요건을 충족하지 못하는 것으로 간주되어 <font color='red'>**지원이 불가합니다.**</font> 이 점을 반드시 유념하시어 불이익을 받는 일이 없도록 사전에 TIPS 과제 종료 일정을 철저히 확인하시기 바랍니다."
            - 이 안내는 사용자의 잠재적인 자격 미달 리스크를 최소화하기 위한 필수 조치입니다.

    IV. 답변 구성 및 전달 단계:
    9. 답변의 구조화 및 명료성:
        - 모든 답변은 논리적 흐름에 따라 명확하게 구조화되어야 합니다. 권장 구조는 다음과 같습니다.
            1.  **질의/요청사항 재확인 및 분석 개요:** 사용자의 입력 내용을 간략히 요약하고, 어떤 관점에서 본 가이드라인을 검토할 것인지 명시합니다.
            2.  **종합 검토 의견 (결론 요약):** 가장 핵심적인 결론(예: **부합**, <font color='red'>**일부 위배**</font>, **추가 확인 필요** 등)을 먼저 제시하여 사용자가 빠르게 결과를 파악할 수 있도록 합니다.
            3.  **세부 검토 내용 및 근거 조항:**
                * 각 쟁점 또는 항목별로 본 가이드라인과의 부합/위배 여부 상세 분석
                * 각 판단에 대한 본 가이드라인의 **정확한 조항 번호 및 내용 인용**
                * 조항 해석에 대한 상세 설명
            4.  **주요 주의사항 및 잠재적 위험 요소:** 식별된 위험 요소와 주의해야 할 점들을 **굵게** 명시합니다. 특히 위험도가 높다고 판단되는 부분은 <font color='red'>**굵은 빨간색 글씨**</font>로 강조합니다.
            5.  **보수적 검토 시나리오 및 대응 방안 제언:** 가장 엄격한 기준 적용 시 발생 가능한 상황과 이에 대한 사용자의 고려 사항 또는 대응 방안을 제안합니다. 이 부분의 경고는 <font color='red'>**굵은 빨간색 글씨**</font>로 표시해주십시오.
            6.  **특수 조건 관련 내용 (해당 시):** CTS.txt FAQ 연관 분석 또는 TIPS 프로그램 관련 고지 사항을 포함합니다.
            7.  **결론 및 권고 사항:** 전체 검토 내용을 요약하고, 사용자에게 필요한 다음 행동(예: **추가 정보 제공 요청**, **전문가 자문 권유**, **계획 수정 제안** 등)을 명확히 안내합니다.
        - 필요에 따라 표, 글머리 기호 등을 활용하여 정보를 효과적으로 전달하십시오.
        - 답변 내용 중 특히 중요하다고 판단되는 핵심 사항, 위반 가능성이 높은 부분, 사용자에게 불이익이 갈 수 있는 경고, 또는 반드시 확인해야 할 권고 사항에 대해서는 **굵은 글씨(Bold)**로 강조하고, 그중에서도 위험도가 높거나 즉각적인 주의가 필요한 심각한 사안에 대해서는 **<font color='red'>굵은 글씨와 함께 빨간색 글씨</font>** (예: `<font color='red'>**이 부분은 매우 중요합니다.**</font>`)를 사용하여 명확히 구분될 수 있도록 해주십시오. 이는 사용자가 답변의 핵심을 빠르게 파악하고 중요한 정보를 놓치지 않도록 하기 위함입니다.
    10. 사용자 중심의 소통:
        - 답변은 항상 사용자의 입장에서 이해하기 쉽도록 평이하고 명확한 용어를 사용하되, 전문성이 필요한 부분에서는 용어 정의를 함께 제공하십시오.
        - 극도로 정중하고 공감하는 어조를 유지하며, 사용자가 마치 신뢰할 수 있는 전문가와 대화하고 있다고 느낄 수 있도록 하십시오.
        - 정보의 나열이 아닌, 사용자의 실제적인 문제 해결과 의사결정에 도움이 되는 방향으로 내용을 구성하고 전달해야 합니다.
        - 답변 말미에는 사용자가 추가적인 질문이나 명확화 요청을 할 수 있도록 자연스럽게 유도하는 문장을 포함하는 것이 좋습니다. (예: "이 외에도 궁금하신 점이나 더 자세한 설명이 필요한 부분이 있으시면 언제든지 다시 문의해 주십시오.")
    11. fixed_accounting_guideline.txt에 대한 특수처리 
        - 중요한 사안으로, 인플루언서 마케팅 등은 홍보마케팅비로 처리해야 함. 외주용역을 맡기는 것이기 때문에 인스타그램, 쿠팡, 유튜브 등의 광고 마케팅비는 홍보마케팅비가 아닌 전문가 활용비임. 
        - 중요한 사안으로, 사업비/지원금 계획을 검토할 때는 기업의 사용 방식을 한 번 이야기해주어야 합니다. 그리고 이어서 관련 지침을 이야기해주고 네가 판단한 근거를 이야기 해야 합니다. (예: 제미나이는 사업비 계획을 전문가 활용에 50% 이상 활용하고 있습니다. 가이드 상 그러한 사용은 가능/불가능하며 이로 인해서 적합/부적합합니다.)

    [지속적 학습 및 개선 의무]
    귀하는 본 가이드라인 및 관련 규정, 지침 등의 변경 사항에 대해 항상 최신 정보를 유지해야 하며, 사용자와의 상호작용을 통해 수집된 다양한 사례와 피드백을 바탕으로 분석 능력과 답변의 질을 지속적으로 향상시켜야 할 의무가 있습니다.
    
    [관련 지침 내용]
    {{context}}

    [사용자 제공 내용 및 질문]
    {{input}}

    [검토 의견 및 답변]
    """
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    document_chain = create_stuff_documents_chain(_llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


# --- Uploaded Document Specific Resource Builders ---
@st.cache_resource(show_spinner="업로드된 문서 처리 중 (Q&A용)...")
def build_generic_vector_store_for_qa(_docs_for_qa, _embeddings_model, file_name_for_log, cache_key_tuple):
    """
    업로드된 문서로부터 Q&A용 일반 벡터 저장소를 구축합니다.
    _docs_for_qa는 Streamlit 캐싱에서 제외되며, cache_key_tuple이 캐시의 주요 식별자 역할을 합니다.
    """
    if not _embeddings_model:
        st.error("임베딩 모델이 로드되지 않아 Q&A용 벡터 저장소를 구축할 수 없습니다.")
        return None
    if not _docs_for_qa: 
        st.warning(f"'{file_name_for_log}'에서 Q&A를 위한 문서를 로드하지 못했습니다.")
        return None
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len, add_start_index=True
        )
        chunked_qa_docs = text_splitter.split_documents(_docs_for_qa) 
        if not chunked_qa_docs:
            st.warning(f"'{file_name_for_log}' 문서가 Q&A를 위해 청크로 분할되지 않았습니다.")
            return None
        
        st.info(f"업로드 문서 ('{file_name_for_log}') Q&A용: {len(_docs_for_qa)}개 원본 Doc -> {len(chunked_qa_docs)}개 청크 분할.")
        vector_store = FAISS.from_documents(chunked_qa_docs, _embeddings_model)
        st.success(f"'{file_name_for_log}' 문서 기반 Q&A용 벡터 저장소 생성 완료.")
        return vector_store
    except Exception as e:
        st.error(f"'{file_name_for_log}' 문서 Q&A용 벡터 저장소 구축 실패: {e}")
        return None

@st.cache_resource
def get_document_qa_rag_chain(_vector_store, _llm, document_name_for_prompt):
    """업로드된 문서 Q&A용 RAG 체인을 생성합니다."""
    if not _vector_store or not _llm: return None
    retriever = _vector_store.as_retriever(search_kwargs={'k': 3})
    # --- MODIFIED PROMPT FOR EMPHASIS ---
    prompt_template_str = f"""
    당신은 '{document_name_for_prompt}' 문서의 내용을 기반으로 질문에 답변하는 AI 어시스턴트입니다.
    문서 내용을 충실히 사용하여 답변해주세요. 제공된 문서에서 답을 찾을 수 없으면, 문서에서 정보를 찾을 수 없다고 명확히 밝혀주세요.
    당신이 판단한 근거와 txt 상의 관련 조항을 같이 출력해주세요. 
    답변 내용 중 특히 중요하다고 판단되는 부분이나 사용자에게 주의가 필요한 내용은 **굵은 글씨**로 강조하고, 위험도가 높거나 치명적인 문제로 이어질 수 있는 내용은 **<font color='red'>굵은 빨간색 글씨</font>**로 강조하여 사용자가 쉽게 인지할 수 있도록 해주세요.
    특히나 보수적 검토한 시나리오도 꼭 같이 보여줘서 투명한 집행 의사결정에 도움이 될 수 있도록 해주세요. 이 경우, 보수적 시나리오에 따른 잠재적 부정적 결과는 **<font color='red'>굵은 빨간색 글씨</font>**로 강조해주십시오.
    답변은 한국어로 상세하고 친절하게 해주세요.

    [문서 내용 중 관련 부분]
    {{context}}

    [사용자 질문]
    {{input}}

    [답변]
    """
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    document_chain = create_stuff_documents_chain(_llm, prompt)
    return create_retrieval_chain(retriever, document_chain)


# --- Streamlit UI 구성 ---
st.set_page_config(page_title="문서 자동 검토 AI", layout="wide")
st.title("문서 자동 검토 및 질의응답 시스템 🤖")

# SQLite DB 초기화
init_db()

OPENAI_API_KEY_INPUT = st.text_input(
    "OpenAI API 키를 입력하세요:", type="password", key="api_key_main_input",
    placeholder="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)

if not OPENAI_API_KEY_INPUT:
    st.warning("OpenAI API 키를 입력해야 다음 단계를 진행할 수 있습니다.")
    st.stop()

embeddings_model_global = get_embeddings_model(OPENAI_API_KEY_INPUT)
llm_global = get_llm(OPENAI_API_KEY_INPUT)

if not embeddings_model_global or not llm_global:
    st.error("OpenAI 모델(임베딩 또는 LLM) 로딩에 실패했습니다. API 키를 확인해주세요.")
    st.stop()
else:
    st.success(f"OpenAI 모델 로딩 완료 (LLM: {llm_global.model_name}).")


st.markdown("---")

selected_business_name_from_selectbox = st.selectbox( # 변수명 변경으로 명확화
    "검토 기준이 될 사업을 선택하세요:",
    options=list(GUIDELINE_FILES.keys()),
    key="business_selection_selectbox",
    index=list(GUIDELINE_FILES.keys()).index(st.session_state.get("selected_business_name_stored", DEFAULT_BUSINESS_NAME))
)

if selected_business_name_from_selectbox: # 변경된 변수명 사용
    st.session_state.selected_business_name_stored = selected_business_name_from_selectbox
    # current_guideline_name이 실제 사용되는 사업명 (로깅에 이 값을 사용)
    st.session_state.current_guideline_name = selected_business_name_from_selectbox 
    st.session_state.current_guideline_path = GUIDELINE_FILES[selected_business_name_from_selectbox]
else: 
    st.session_state.current_guideline_path = None
    st.session_state.current_guideline_name = None # 사업명이 선택되지 않은 경우 None으로 설정
    st.warning("사업을 선택해주세요.")
    st.stop()

active_guideline_rag_chain = None
if st.session_state.current_guideline_path and st.session_state.current_guideline_name:
    guideline_vector_store = get_cached_guideline_vector_store(
        embeddings_model_global, 
        st.session_state.current_guideline_path, 
        st.session_state.current_guideline_name # 사업명 전달
    )

    if guideline_vector_store:
        active_guideline_rag_chain = get_rag_chain_for_guideline(
            guideline_vector_store, 
            llm_global, 
            st.session_state.current_guideline_name # 사업명 전달
        )
        if active_guideline_rag_chain:
            st.success(f"'{st.session_state.current_guideline_name}' 지침 기반 RAG 체인 준비 완료.")
        else:
            st.error(f"'{st.session_state.current_guideline_name}' 지침 RAG 체인을 초기화할 수 없습니다.")
    else:
        st.error(f"'{st.session_state.current_guideline_name}' 지침 벡터 저장소를 생성할 수 없습니다.")
else:
    st.info("먼저 검토 기준이 될 사업을 선택하고 API 키를 입력해주세요.")

st.markdown("---")
st.subheader("1. 기업 제출 문서 자동 검토")
if st.session_state.current_guideline_name:
    st.caption(f"업로드된 문서의 내용을 바탕으로 '{st.session_state.current_guideline_name}'의 회계 지침에 맞는지 자동 검토 의견을 제시합니다.")
else:
    st.caption("업로드된 문서의 내용을 바탕으로 선택된 회계 지침에 맞는지 자동 검토 의견을 제시합니다.")

uploaded_file_for_review = st.file_uploader(
    "검토할 기업 문서를 업로드하세요 (txt, pdf, xlsx, xls, docx, doc, csv)",
    type=["txt", "pdf", "xlsx", "xls", "docx", "doc", "csv"],
    key="auto_review_uploader"
)

current_review_config_key = (
    uploaded_file_for_review.name if uploaded_file_for_review else None,
    st.session_state.get("current_guideline_path")
)

if "last_review_config_key_processed" not in st.session_state:
    st.session_state.last_review_config_key_processed = None
if "auto_review_output" not in st.session_state:
    st.session_state.auto_review_output = None

if uploaded_file_for_review is not None and active_guideline_rag_chain and st.session_state.current_guideline_name: # 사업명 확인 추가
    if st.session_state.last_review_config_key_processed != current_review_config_key:
        st.session_state.auto_review_output = None 
        st.session_state.last_review_config_key_processed = current_review_config_key

        st.info(f"'{uploaded_file_for_review.name}' 파일 자동 검토를 '{st.session_state.current_guideline_name}' 지침에 따라 시작합니다...")
        temp_file_path_for_upload = save_uploaded_file_to_temp(uploaded_file_for_review)
        
        if temp_file_path_for_upload:
            st.info(f"업로드된 검토용 파일 로딩 시도: {uploaded_file_for_review.name}")
            uploaded_docs_content_list = load_document_from_local_path(temp_file_path_for_upload, uploaded_file_for_review.name)
            
            if uploaded_docs_content_list:
                content_to_review = ""
                for doc_idx, doc in enumerate(uploaded_docs_content_list[:2]): 
                    content_to_review += f"[문서 {doc_idx+1} 시작]\n"
                    content_to_review += str(doc.page_content)[:1000] + "\n" 
                    content_to_review += f"[문서 {doc_idx+1} 끝]\n...\n"
                
                if content_to_review.strip(): 
                    auto_review_question = f"다음은 기업이 제출한 문서의 일부 내용입니다. 이 내용이 '{st.session_state.current_guideline_name}' 지침에 부합하는지, 특별히 주의해야 할 점이나 확인이 필요한 사항이 있는지 검토해주세요:\n\n---\n{content_to_review.strip()}\n---"
                    
                    with st.spinner("자동 검토 의견 생성 중..."):
                        try:
                            with get_openai_callback() as cb:
                                response = active_guideline_rag_chain.invoke({"input": auto_review_question})
                                cost_usd = cb.total_cost
                                total_tokens = cb.total_tokens
                                prompt_tokens = cb.prompt_tokens
                                completion_tokens = cb.completion_tokens
                                
                                log_token_usage(
                                    business_name=st.session_state.current_guideline_name, # 사업명 전달
                                    model_name=llm_global.model_name,
                                    prompt_tokens=prompt_tokens,
                                    completion_tokens=completion_tokens,
                                    total_tokens=total_tokens,
                                    cost_usd=cost_usd,
                                    api_call_tag="auto_review_guideline"
                                )
                                cost_krw = cost_usd * USD_TO_KRW_EXCHANGE_RATE

                            st.session_state.auto_review_output = {
                                "file_name": uploaded_file_for_review.name,
                                "guideline_name": st.session_state.current_guideline_name,
                                "answer": response["answer"],
                                "cost_info": f"총 사용 토큰: {total_tokens}, 예상 비용 (USD): ${cost_usd:.6f}, (KRW): ₩{cost_krw:,.2f}"
                            }
                        except Exception as e:
                            st.error(f"자동 검토 중 오류 발생: {e}")
                            st.session_state.auto_review_output = None 
                else:
                    st.warning("업로드된 문서에서 검토할 내용을 추출하지 못했습니다.")
            else:
                st.error(f"업로드된 문서 내용을 로드하지 못했습니다. ({uploaded_file_for_review.name})")
            
            if os.path.exists(temp_file_path_for_upload):
                try:
                    os.remove(temp_file_path_for_upload)
                except Exception as e:
                    st.warning(f"임시 파일 삭제 실패 ({temp_file_path_for_upload}): {e}")
        else:
            st.error("업로드된 파일을 임시 저장하지 못했습니다.")

    if st.session_state.auto_review_output:
        output = st.session_state.auto_review_output
        if output["file_name"] == uploaded_file_for_review.name and \
           output["guideline_name"] == st.session_state.current_guideline_name:
            st.subheader(f"'{output['file_name']}' 자동 검토 결과 ({output['guideline_name']} 기준):")
            st.markdown(output["answer"], unsafe_allow_html=True) 
            with st.expander("자동 검토 비용 정보 (이번 요청)"):
                st.text(output["cost_info"])
        else: 
            st.info("새로운 파일 또는 지침이 선택되었습니다. 자동 검토가 다시 실행됩니다.")

elif uploaded_file_for_review is None:
    st.info("자동 검토를 위해 기업 문서를 업로드해주세요.")
elif not active_guideline_rag_chain:
    st.warning("자동 검토를 위한 지침 RAG 체인이 준비되지 않았습니다. API키와 사업 선택을 확인해주세요.")
elif not st.session_state.current_guideline_name: # 사업명이 선택되지 않은 경우
    st.warning("자동 검토를 실행하기 전에 먼저 사업을 선택해주세요.")


st.markdown("---")
st.subheader("2. 업로드된 문서에 대해 직접 질의응답")
st.caption("위 '자동 검토' 섹션에서 사용된 동일한 업로드 파일을 대상으로 직접 질문하고 답변을 받을 수 있습니다.")

uploaded_doc_qa_chain = None
# 사업명이 선택되었고, 파일이 업로드 되었을 때만 이 섹션 활성화
if uploaded_file_for_review is not None and st.session_state.current_guideline_name: 
    current_uploaded_file_key_for_qa_tuple = (
        uploaded_file_for_review.name,
        uploaded_file_for_review.size,
        getattr(uploaded_file_for_review, 'last_modified', uploaded_file_for_review.size) 
    )
    
    vs_session_key = f"uploaded_vs_{current_uploaded_file_key_for_qa_tuple}_{OPENAI_API_KEY_INPUT}"
    chain_session_key = f"uploaded_chain_{current_uploaded_file_key_for_qa_tuple}_{OPENAI_API_KEY_INPUT}"

    if chain_session_key not in st.session_state: 
        st.info(f"'{uploaded_file_for_review.name}' 문서 기반 질의응답 준비 중 ({st.session_state.current_guideline_name} 사업 컨텍스트)...")
        temp_file_path_for_qa = save_uploaded_file_to_temp(uploaded_file_for_review) 
        
        if temp_file_path_for_qa:
            st.info(f"업로드된 Q&A용 파일 로딩 시도: {uploaded_file_for_review.name}")
            uploaded_documents_for_qa = load_document_from_local_path(temp_file_path_for_qa, uploaded_file_for_review.name)
            if uploaded_documents_for_qa:
                vector_store_for_qa = build_generic_vector_store_for_qa(
                    _docs_for_qa=uploaded_documents_for_qa, 
                    _embeddings_model=embeddings_model_global,
                    file_name_for_log=uploaded_file_for_review.name,
                    cache_key_tuple=current_uploaded_file_key_for_qa_tuple 
                )
                st.session_state[vs_session_key] = vector_store_for_qa 

                if vector_store_for_qa:
                    chain = get_document_qa_rag_chain(
                        vector_store_for_qa,
                        llm_global,
                        uploaded_file_for_review.name
                    )
                    st.session_state[chain_session_key] = chain 
                    if chain:
                        st.success(f"'{uploaded_file_for_review.name}' 문서 기반 질의응답 준비 완료.")
                    else:
                        st.error(f"'{uploaded_file_for_review.name}' 문서 Q&A 체인 생성 실패.")
                else: 
                    if chain_session_key in st.session_state: del st.session_state[chain_session_key]
            else: 
                st.error(f"업로드된 Q&A용 문서 내용을 로드하지 못했습니다. ({uploaded_file_for_review.name})")
                if vs_session_key in st.session_state: del st.session_state[vs_session_key]
                if chain_session_key in st.session_state: del st.session_state[chain_session_key]

            if os.path.exists(temp_file_path_for_qa):
                try:
                    os.remove(temp_file_path_for_qa)
                except Exception as e:
                    st.warning(f"임시 파일 삭제 실패 ({temp_file_path_for_qa}): {e}")
        else: 
            if vs_session_key in st.session_state: del st.session_state[vs_session_key]
            if chain_session_key in st.session_state: del st.session_state[chain_session_key]
            st.error("Q&A를 위한 업로드 파일 임시 저장 실패.")

    uploaded_doc_qa_chain = st.session_state.get(chain_session_key)

    if uploaded_doc_qa_chain:
        user_question_for_uploaded_doc = st.text_input(
            f"'{uploaded_file_for_review.name}' 내용에 대해 질문하세요 ({st.session_state.current_guideline_name} 사업 관련):",
            key=f"user_question_direct_uploaded_{uploaded_file_for_review.name}_{st.session_state.current_guideline_name}" 
        )
        if user_question_for_uploaded_doc:
            with st.spinner("답변 생성 중..."):
                try:
                    with get_openai_callback() as cb:
                        response = uploaded_doc_qa_chain.invoke({"input": user_question_for_uploaded_doc})
                        cost_usd = cb.total_cost
                        total_tokens = cb.total_tokens
                        prompt_tokens = cb.prompt_tokens
                        completion_tokens = cb.completion_tokens

                        log_token_usage(
                            business_name=st.session_state.current_guideline_name, # 사업명 전달
                            model_name=llm_global.model_name,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            cost_usd=cost_usd,
                            api_call_tag="qa_uploaded_doc"
                        )
                        cost_krw = cost_usd * USD_TO_KRW_EXCHANGE_RATE

                    st.subheader("답변:")
                    st.markdown(response["answer"], unsafe_allow_html=True) 
                    if "context" in response and response["context"]:
                        with st.expander("참고한 업로드 문서 내용 (일부)"):
                            for i, doc_ctx in enumerate(response["context"][:2]): 
                                st.write(f"**출처 {i+1} (소스: {doc_ctx.metadata.get('source', '알 수 없음')})**")
                                st.caption(doc_ctx.page_content[:300] + "...")
                    with st.expander("비용 정보 (이번 요청)"):
                        st.text(f"총 사용 토큰: {total_tokens}, 예상 비용 (USD): ${cost_usd:.6f}, (KRW): ₩{cost_krw:,.2f}")
                except Exception as e:
                    st.error(f"직접 질의응답 중 오류 발생: {e}")
    elif uploaded_file_for_review is not None : 
        st.warning("업로드된 문서에 대한 질의응답 기능이 아직 준비되지 않았거나 초기화 중입니다. 잠시 후 다시 시도해주세요.")

elif not st.session_state.current_guideline_name:
     st.info("문서 질의응답을 위해 먼저 사업을 선택해주세요.")
else: # uploaded_file_for_review is None but business_name is selected
    st.info("문서를 업로드하면 해당 문서에 대한 자동 검토 및 직접 질의응답 기능을 사용할 수 있습니다.")


st.markdown("---")
st.subheader("3. 지침 문서 직접 질의응답")

if active_guideline_rag_chain and st.session_state.current_guideline_name:
    st.caption(f"'{st.session_state.current_guideline_name}' 지침 내용에 대해 직접 질문하고 답변을 받을 수 있습니다.")
    
    user_question_for_guideline = st.text_input(
        f"'{st.session_state.current_guideline_name}' 지침 내용에 대해 질문하세요:",
        key=f"user_question_guideline_direct_{st.session_state.current_guideline_name}"
    )

    if user_question_for_guideline:
        with st.spinner(f"'{st.session_state.current_guideline_name}' 지침 기반 답변 생성 중..."):
            try:
                with get_openai_callback() as cb:
                    input_for_guideline_qna = f"다음은 '{st.session_state.current_guideline_name}' 지침에 대한 직접적인 질문입니다: {user_question_for_guideline}"
                    response = active_guideline_rag_chain.invoke({"input": input_for_guideline_qna})
                    cost_usd = cb.total_cost
                    total_tokens = cb.total_tokens
                    prompt_tokens = cb.prompt_tokens
                    completion_tokens = cb.completion_tokens

                    log_token_usage(
                        business_name=st.session_state.current_guideline_name, # 사업명 전달
                        model_name=llm_global.model_name,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        cost_usd=cost_usd,
                        api_call_tag="qa_guideline_direct"
                    )
                    cost_krw = cost_usd * USD_TO_KRW_EXCHANGE_RATE
                
                st.subheader("답변:")
                st.markdown(response["answer"], unsafe_allow_html=True) 
                
                if "context" in response and response["context"]:
                    with st.expander("참고한 지침 내용 (일부)"):
                        for i, doc_ctx in enumerate(response["context"][:3]): 
                            source_name_display = doc_ctx.metadata.get('source', st.session_state.current_guideline_name)
                            st.write(f"**출처 {i+1} (문서: {source_name_display})**")
                            st.caption(doc_ctx.page_content[:500] + "...") 
                
                with st.expander("비용 정보 (이번 요청)"):
                     st.text(f"총 사용 토큰: {total_tokens}, 예상 비용 (USD): ${cost_usd:.6f}, (KRW): ₩{cost_krw:,.2f}")
            except Exception as e:
                st.error(f"'{st.session_state.current_guideline_name}' 지침 직접 질의응답 중 오류 발생: {e}")
elif not st.session_state.current_guideline_name:
     st.info("먼저 검토 기준이 될 사업을 선택해주세요.")
else: # active_guideline_rag_chain is None (and business_name might be selected)
    st.warning(f"'{st.session_state.current_guideline_name}' 지침에 대한 RAG 체인이 준비되지 않았습니다. API키와 사업 선택을 확인해주세요.")

# --- (선택적) 로깅된 데이터 확인 UI ---
st.sidebar.markdown("---")
st.sidebar.subheader("누적 사용량 확인 (최근 10건)")
show_logs_button = st.sidebar.button("사용량 로그 보기")

if show_logs_button:
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        # business_name 컬럼을 포함하여 조회
        cursor.execute("SELECT timestamp, business_name, api_call_tag, model_name, total_tokens, cost_usd, cost_krw FROM usage_logs ORDER BY timestamp DESC LIMIT 10")
        logs = cursor.fetchall()
        conn.close()
        
        if logs:
            st.sidebar.markdown("`시간 | 사업명 | 호출태그 | 모델 | 토큰 | USD | KRW`")
            for log_entry in logs:
                ts, biz_name, tag, model, tokens, usd, krw = log_entry
                # 소수점 및 None 값 처리 개선
                usd_display = f"${usd:.4f}" if usd is not None else "N/A"
                krw_display = f"₩{krw:,.0f}" if krw is not None else "N/A"
                tag_display = tag if tag else "N/A"
                biz_name_display = biz_name if biz_name else "N/A"

                st.sidebar.text(f"{ts[:19]} | {biz_name_display} | {tag_display} | {model} | T:{tokens} | {usd_display} | {krw_display}")
        else:
            st.sidebar.info("기록된 사용 내역이 없습니다.")
            
    except Exception as e:
        st.sidebar.error(f"로그 조회 중 오류: {e}")