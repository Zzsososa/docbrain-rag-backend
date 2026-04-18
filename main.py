import os
import re
import time
import unicodedata
import mimetypes
from pathlib import Path
import tempfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from dataclasses import dataclass
import requests
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
from docx import Document as DocxDocument
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase.client import create_client, Client

BASE_DIR = Path(__file__).resolve().parent
ENV_CANDIDATES = [
    BASE_DIR.parent / 'docbrain-app' / '.env.local',
    BASE_DIR.parent / 'siged-ia' / '.env.local',
]

for env_path in ENV_CANDIDATES:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        break

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_AI_KEY")
GOOGLE_BONUS_API_KEY = os.getenv("GOOGLE_AI_KEY_BONUS_FREE")
DEFAULT_TEXT_MODEL = "gemini-2.5-flash-lite"
DEFAULT_TEXT_FALLBACK_MODELS = "gemini-2.5-flash,gemini-2.0-flash"
DOC_MODEL = os.getenv("GOOGLE_DOC_MODEL", DEFAULT_TEXT_MODEL)
GENERAL_MODEL = os.getenv("GOOGLE_GENERAL_MODEL", DEFAULT_TEXT_MODEL)
DOC_MODELS = os.getenv("GOOGLE_DOC_MODELS", DOC_MODEL)
GENERAL_MODELS = os.getenv("GOOGLE_GENERAL_MODELS", GENERAL_MODEL)
BONUS_DOC_MODEL = os.getenv("GOOGLE_BONUS_DOC_MODEL", DEFAULT_TEXT_MODEL)
BONUS_GENERAL_MODEL = os.getenv("GOOGLE_BONUS_GENERAL_MODEL", DEFAULT_TEXT_MODEL)
BONUS_DOC_MODELS = os.getenv("GOOGLE_BONUS_DOC_MODELS", BONUS_DOC_MODEL)
BONUS_GENERAL_MODELS = os.getenv("GOOGLE_BONUS_GENERAL_MODELS", BONUS_GENERAL_MODEL)
GOOGLE_MAX_RETRIES = int(os.getenv("GOOGLE_MAX_RETRIES", "2"))
GOOGLE_POLL_SECONDS = int(os.getenv("GOOGLE_POLL_SECONDS", "2"))
GOOGLE_FILE_PROCESS_TIMEOUT = int(os.getenv("GOOGLE_FILE_PROCESS_TIMEOUT", "90"))
WORKER_SECRET = os.getenv("WORKER_SECRET")
EMBEDDING_MODELS = os.getenv(
    "GOOGLE_EMBEDDING_MODELS",
    os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"),
)
EMBEDDING_DIMENSIONS = int(os.getenv("GOOGLE_EMBEDDING_DIMENSIONS", "768"))
INDEX_BATCH_SIZE = int(os.getenv("INDEX_BATCH_SIZE", "24"))
INDEX_MAX_PENDING = int(os.getenv("INDEX_MAX_PENDING", "3"))
EMBEDDING_RATE_LIMIT_PAUSE_SECONDS = int(os.getenv("EMBEDDING_RATE_LIMIT_PAUSE_SECONDS", "35"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "4"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "180"))
LARGE_DOC_PAGE_THRESHOLD = int(os.getenv("LARGE_DOC_PAGE_THRESHOLD", "60"))
LARGE_DOC_CHUNK_SIZE = int(os.getenv("LARGE_DOC_CHUNK_SIZE", "2600"))
LARGE_DOC_CHUNK_OVERLAP = int(os.getenv("LARGE_DOC_CHUNK_OVERLAP", "220"))
USAGE_TIMEZONE = os.getenv("USAGE_TIMEZONE", "America/Santo_Domingo")
MAX_DAILY_PAID_CHAT_REQUESTS = int(os.getenv("MAX_DAILY_PAID_CHAT_REQUESTS", os.getenv("MAX_DAILY_GOOGLE_CHAT_REQUESTS", "50")))
MAX_DAILY_BONUS_CHAT_REQUESTS = int(os.getenv("MAX_DAILY_BONUS_CHAT_REQUESTS", "30"))

SPANISH_STOPWORDS = {
    "a", "al", "algo", "alguna", "alguno", "ante", "bajo", "como", "con", "contra", "cual",
    "cuales", "de", "del", "desde", "donde", "el", "ella", "ellas", "ello", "ellos", "en",
    "entre", "era", "eran", "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta", "estas",
    "este", "esto", "estos", "fue", "ha", "hay", "la", "las", "le", "les", "lo", "los", "mas",
    "me", "mi", "mis", "mucho", "muy", "no", "nos", "o", "otra", "otro", "para", "pero", "por",
    "que", "quien", "se", "si", "sin", "sobre", "su", "sus", "te", "tu", "tus", "un", "una",
    "uno", "y", "ya"
}

CHAT_PATTERNS = (
    "hola", "hello", "helloo", "buenas", "buenos dias", "buenas tardes", "buenas noches",
    "como estas", "quien eres", "que haces", "ayudame", "gracias", "ok", "okay"
)

SEARCH_HINTS = (
    "base de conocimiento", "en algun documento", "en algún documento", "en que documento",
    "en qué documento", "se menciona", "se habla de", "donde se habla", "dónde se habla",
    "donde aparece", "dónde aparece", "buscar", "busca", "aparece", "aparece en",
    "documento", "documentos", "archivo", "archivos", "expediente", "expedientes"
)

GENERAL_KNOWLEDGE_HINTS = (
    "que dia es hoy", "qué día es hoy", "que es", "qué es", "quien es", "quién es",
    "puedes decirme que es", "puedes decirme qué es", "explicame", "explícame", "define"
)

STRUCTURAL_QUERY_HINTS = (
    "indice", "tabla de contenido", "capitulo", "seccion", "pagina",
    "resumen", "partes", "estructura", "apartados", "contenido"
)

PLACEHOLDER_PREFIX = "documento listo para analisis de contexto masivo."

missing_env = [
    name for name, value in {
        "NEXT_PUBLIC_SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_SERVICE_ROLE_KEY": SUPABASE_KEY,
        "GOOGLE_AI_KEY": GOOGLE_API_KEY,
    }.items() if not value
]

if missing_env:
    raise RuntimeError(
        "Faltan variables de entorno requeridas en .env.local: "
        + ", ".join(missing_env)
    )

genai.configure(api_key=GOOGLE_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
embeddings_model_cache: dict[str, GoogleGenerativeAIEmbeddings] = {}

app = FastAPI()


class ChatRequest(BaseModel):
    document_id: Optional[str] = None
    message: str


class JobResponse(BaseModel):
    queued: bool
    document_id: Optional[str] = None
    count: int = 0


class DailyEmbeddingQuotaError(Exception):
    pass


class EmbeddingRateLimitError(Exception):
    pass


@dataclass
class ChatUsageReservation:
    allowed: bool
    tier: str
    usage_day: str
    used_count: int
    limit_count: int


def parse_model_candidates(raw_models: str, fallback: str) -> list[str]:
    candidates = [item.strip() for item in raw_models.split(",") if item.strip()]
    if fallback and fallback not in candidates:
        candidates.append(fallback)
    return candidates


def upgrade_legacy_model_candidates(model_names: list[str], fallback_defaults: list[str]) -> list[str]:
    upgraded: list[str] = []
    legacy_aliases = {
        "gemini-1.5-flash": [DEFAULT_TEXT_MODEL, *fallback_defaults],
        "gemini-1.5-flash-latest": [DEFAULT_TEXT_MODEL, *fallback_defaults],
        "gemini-1.5-pro": ["gemini-2.5-flash", "gemini-2.0-flash", *fallback_defaults],
        "gemini-pro": [DEFAULT_TEXT_MODEL, *fallback_defaults],
    }

    for model_name in model_names:
        replacements = legacy_aliases.get(model_name, [model_name])
        for replacement in replacements:
            if replacement and replacement not in upgraded:
                upgraded.append(replacement)

    for model_name in fallback_defaults:
        if model_name and model_name not in upgraded:
            upgraded.append(model_name)

    return upgraded


DEFAULT_TEXT_CANDIDATES = parse_model_candidates(DEFAULT_TEXT_FALLBACK_MODELS, DEFAULT_TEXT_MODEL)
DOC_MODEL_CANDIDATES = upgrade_legacy_model_candidates(parse_model_candidates(DOC_MODELS, DOC_MODEL), DEFAULT_TEXT_CANDIDATES)
GENERAL_MODEL_CANDIDATES = upgrade_legacy_model_candidates(parse_model_candidates(GENERAL_MODELS, GENERAL_MODEL), DEFAULT_TEXT_CANDIDATES)
BONUS_DOC_MODEL_CANDIDATES = upgrade_legacy_model_candidates(parse_model_candidates(BONUS_DOC_MODELS, BONUS_DOC_MODEL), DEFAULT_TEXT_CANDIDATES)
BONUS_GENERAL_MODEL_CANDIDATES = upgrade_legacy_model_candidates(parse_model_candidates(BONUS_GENERAL_MODELS, BONUS_GENERAL_MODEL), DEFAULT_TEXT_CANDIDATES)
EMBEDDING_MODEL_CANDIDATES = parse_model_candidates(EMBEDDING_MODELS, "models/gemini-embedding-001")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def is_placeholder_content(content: str) -> bool:
    return normalize_text(content).startswith(PLACEHOLDER_PREFIX)


def extract_terms(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-ZáéíóúÁÉÍÓÚñÑ0-9]{3,}", text)
    normalized_terms = []
    seen = set()

    for token in tokens:
        cleaned = normalize_text(token)
        if cleaned in SPANISH_STOPWORDS:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            normalized_terms.append(cleaned)

    return normalized_terms[:8]


def is_quota_error(message: str) -> bool:
    lowered = message.lower()
    return "quota exceeded" in lowered or "429" in lowered or "rate limit" in lowered


def is_model_not_found_error(message: str) -> bool:
    lowered = message.lower()
    return "not found" in lowered or "not_found" in lowered or "404" in lowered


def is_daily_embedding_quota_error(message: str) -> bool:
    lowered = message.lower()
    return (
        "embedcontentrequestsperday" in lowered
        or "requestsperday" in lowered
        or "per day" in lowered
        or "current quota" in lowered and "quotaValue': '1000'" in message
    )


def extract_retry_delay_seconds(message: str) -> Optional[int]:
    retry_match = re.search(r"retryDelay': '(\d+)s'", message)
    if retry_match:
        return int(retry_match.group(1))

    retry_match = re.search(r"retry in ([0-9.]+)s", message, re.IGNORECASE)
    if retry_match:
        return int(float(retry_match.group(1))) + 1

    return None


def looks_like_general_chat(message: str) -> bool:
    normalized = normalize_text(message).strip(" ?!.,")
    return normalized in CHAT_PATTERNS


def looks_like_general_knowledge(message: str) -> bool:
    normalized = normalize_text(message)
    return any(hint in normalized for hint in GENERAL_KNOWLEDGE_HINTS)


def should_search_knowledge_base(message: str) -> bool:
    normalized = normalize_text(message)
    return any(hint in normalized for hint in SEARCH_HINTS)


def looks_like_structural_query(message: str) -> bool:
    normalized = normalize_text(message)
    return any(hint in normalized for hint in STRUCTURAL_QUERY_HINTS)


def generate_once(model_name: str, payload):
    last_error = None

    for attempt in range(GOOGLE_MAX_RETRIES + 1):
        try:
            model = genai.GenerativeModel(model_name)
            return model.generate_content(payload)
        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()
            is_timeout = "timed out" in error_text or "deadline" in error_text or "504" in error_text
            if not is_timeout or attempt >= GOOGLE_MAX_RETRIES:
                break
            time.sleep(2 * (attempt + 1))

    raise last_error


def generate_with_fallback(model_names: list[str], payload):
    last_error = None

    for model_name in model_names:
        try:
            response = generate_once(model_name, payload)
            return response, model_name
        except Exception as exc:
            last_error = exc
            if is_quota_error(str(exc)) or is_model_not_found_error(str(exc)):
                continue
            raise

    raise last_error


def extract_response_text(response) -> str:
    return getattr(response, "text", "") or ""


def build_bonus_notice(slot: ChatUsageReservation) -> Optional[str]:
    if slot.tier == "bonus" and slot.used_count == 1:
        return "Oye, como regalo por haber pagado esta app, tienes 30 solicitudes mas."
    return None


def prepend_bonus_notice(answer: str, slot: ChatUsageReservation) -> str:
    notice = build_bonus_notice(slot)
    return f"{notice}\n\n{answer}" if notice else answer


def can_use_bonus_failover() -> bool:
    return bool(GOOGLE_BONUS_API_KEY)


def is_failover_eligible_error(message: str) -> bool:
    return is_quota_error(message) or is_model_not_found_error(message)


def get_embeddings_model(model_name: str, api_key: Optional[str] = None) -> GoogleGenerativeAIEmbeddings:
    cache_key = f"{api_key or GOOGLE_API_KEY}:{model_name}"
    if cache_key not in embeddings_model_cache:
        embeddings_model_cache[cache_key] = GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key or GOOGLE_API_KEY,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        )
    return embeddings_model_cache[cache_key]


def embed_documents_with_fallback(
    texts: list[str],
    *,
    task_type: str = "RETRIEVAL_DOCUMENT",
    titles: Optional[list[str]] = None,
    api_key: Optional[str] = None,
) -> list[list[float]]:
    last_error = None

    for model_name in EMBEDDING_MODEL_CANDIDATES:
        try:
            return get_embeddings_model(model_name, api_key=api_key).embed_documents(
                texts,
                task_type=task_type,
                titles=titles,
            )
        except Exception as exc:
            last_error = exc
            if "not_found" in str(exc).lower() or "404" in str(exc):
                continue
            raise

    raise last_error


def embed_query_with_fallback(
    text: str,
    api_key: Optional[str] = None,
    *,
    task_type: str = "RETRIEVAL_QUERY",
    title: Optional[str] = None,
) -> list[float]:
    last_error = None

    for model_name in EMBEDDING_MODEL_CANDIDATES:
        try:
            return get_embeddings_model(model_name, api_key=api_key).embed_query(
                text,
                task_type=task_type,
                title=title,
            )
        except Exception as exc:
            last_error = exc
            if "not_found" in str(exc).lower() or "404" in str(exc):
                continue
            raise

    raise last_error


def verify_worker_secret(x_worker_secret: Optional[str]) -> None:
    if WORKER_SECRET and x_worker_secret != WORKER_SECRET:
        raise HTTPException(status_code=401, detail="Worker no autorizado.")


def get_usage_day() -> str:
    try:
        return datetime.now(ZoneInfo(USAGE_TIMEZONE)).date().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


def count_usage_events(event_type: str, usage_day: Optional[str] = None) -> int:
    target_day = usage_day or get_usage_day()
    response = (
        supabase.table("ai_usage_events")
        .select("id", count="exact")
        .eq("event_type", event_type)
        .eq("usage_day", target_day)
        .execute()
    )
    return response.count or 0


def reserve_slot(event_type: str, daily_limit: int, request: ChatRequest) -> ChatUsageReservation:
    response = supabase.rpc(
        "reserve_daily_ai_usage_slot",
        {
            "p_event_type": event_type,
            "p_daily_limit": daily_limit,
            "p_resource_id": request.document_id,
            "p_metadata": {
                "message_length": len(request.message or ""),
                "document_mode": bool(request.document_id),
            },
        },
    ).execute()
    slot = (response.data or [{}])[0]
    return ChatUsageReservation(
        allowed=bool(slot.get("allowed", False)),
        tier="paid" if event_type == "chat_paid_request" else "bonus",
        usage_day=slot.get("usage_day") or get_usage_day(),
        used_count=int(slot.get("used_count") or 0),
        limit_count=int(slot.get("limit_count") or daily_limit),
    )


def reserve_daily_google_request_budget(request: ChatRequest) -> ChatUsageReservation:
    paid_slot = reserve_slot("chat_paid_request", MAX_DAILY_PAID_CHAT_REQUESTS, request)
    if paid_slot.allowed:
        return paid_slot

    if GOOGLE_BONUS_API_KEY:
        bonus_slot = reserve_slot("chat_bonus_request", MAX_DAILY_BONUS_CHAT_REQUESTS, request)
        if bonus_slot.allowed:
            return bonus_slot

    raise HTTPException(
        status_code=429,
        detail=(
            f"Se alcanzo el limite diario de {MAX_DAILY_PAID_CHAT_REQUESTS + (MAX_DAILY_BONUS_CHAT_REQUESTS if GOOGLE_BONUS_API_KEY else 0)} "
            f"consultas con IA para {paid_slot.usage_day}. Intenta de nuevo manana."
        ),
    )


def get_mime_type(file_path: str, file_name: str) -> str:
    guessed, _ = mimetypes.guess_type(file_name or file_path)
    return guessed or "application/octet-stream"


def extract_rest_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
        return payload.get("error", {}).get("message") or response.text
    except Exception:
        return response.text


def generate_bonus_text_with_fallback(model_names: list[str], prompt: str) -> tuple[str, str]:
    if not GOOGLE_BONUS_API_KEY:
        raise RuntimeError("No se configuro GOOGLE_AI_KEY_BONUS_FREE.")

    last_error = None
    for model_name in model_names:
        try:
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                params={"key": GOOGLE_BONUS_API_KEY},
                json={"contents": [{"role": "user", "parts": [{"text": prompt}]}]},
                timeout=120,
            )
            if not response.ok:
                raise RuntimeError(extract_rest_error_message(response))
            payload = response.json()
            parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts if part.get("text"))
            if not text:
                raise RuntimeError("Gemini no devolvio texto utilizable.")
            return text, model_name
        except Exception as exc:
            last_error = exc
            if is_quota_error(str(exc)) or is_model_not_found_error(str(exc)):
                continue

    raise last_error or RuntimeError("No se pudo generar respuesta bonus.")


def upload_bonus_file(file_path: str, display_name: str, mime_type: str) -> dict:
    file_size = os.path.getsize(file_path)
    start_response = requests.post(
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
        params={"key": GOOGLE_BONUS_API_KEY},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        json={"file": {"display_name": display_name}},
        timeout=120,
    )
    if not start_response.ok:
        raise RuntimeError(extract_rest_error_message(start_response))

    upload_url = start_response.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        raise RuntimeError("No se recibio URL de carga para el archivo bonus.")

    with open(file_path, "rb") as handle:
        upload_response = requests.post(
            upload_url,
            headers={
                "Content-Length": str(file_size),
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
            },
            data=handle.read(),
            timeout=120,
        )

    if not upload_response.ok:
        raise RuntimeError(extract_rest_error_message(upload_response))

    return upload_response.json().get("file", {})


def wait_for_bonus_file(file_name: str) -> dict:
    started_at = time.time()
    resource_name = file_name if file_name.startswith("files/") else f"files/{file_name}"

    while True:
        response = requests.get(
            f"https://generativelanguage.googleapis.com/v1beta/{resource_name}",
            params={"key": GOOGLE_BONUS_API_KEY},
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(extract_rest_error_message(response))
        payload = response.json()
        state = (payload.get("state") or {}).get("name", "")
        if state == "ACTIVE":
            return payload
        if state and state != "PROCESSING":
            raise RuntimeError(f"El archivo bonus no quedo listo. Estado: {state}")
        if time.time() - started_at > GOOGLE_FILE_PROCESS_TIMEOUT:
            raise TimeoutError("La carga del documento bonus hacia Gemini tardo demasiado.")
        time.sleep(GOOGLE_POLL_SECONDS)


def generate_bonus_document_with_fallback(file_path: str, display_name: str, prompt: str) -> tuple[str, str]:
    mime_type = get_mime_type(file_path, display_name)
    last_error = None

    for model_name in BONUS_DOC_MODEL_CANDIDATES:
        try:
            uploaded_file = upload_bonus_file(file_path, display_name, mime_type)
            ready_file = wait_for_bonus_file(uploaded_file.get("name", ""))
            response = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                params={"key": GOOGLE_BONUS_API_KEY},
                json={
                    "contents": [
                        {
                            "role": "user",
                            "parts": [
                                {"file_data": {"mime_type": mime_type, "file_uri": ready_file.get("uri")}},
                                {"text": prompt},
                            ],
                        }
                    ]
                },
                timeout=180,
            )
            if not response.ok:
                raise RuntimeError(extract_rest_error_message(response))
            payload = response.json()
            parts = payload.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts if part.get("text"))
            if not text:
                raise RuntimeError("Gemini bonus no devolvio texto utilizable.")
            return text, model_name
        except Exception as exc:
            last_error = exc
            if is_quota_error(str(exc)) or is_model_not_found_error(str(exc)):
                continue

    raise last_error or RuntimeError("No se pudo generar respuesta bonus de documento.")


def update_index_status(document_id: str, **updates) -> None:
    supabase.table("documents").update(updates).eq("id", document_id).execute()


def download_document_to_temp(doc_data: dict) -> str:
    file_path_in_storage = (doc_data.get("file_path") or "").replace("\\", "/")
    if not file_path_in_storage:
        raise ValueError("El documento no tiene archivo asociado en Storage.")

    file_url = supabase.storage.from_("documents").get_public_url(file_path_in_storage)
    response = requests.get(file_url, timeout=120)
    response.raise_for_status()

    temp_ext = os.path.splitext(doc_data.get("name", ""))[1] or ".bin"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=temp_ext)
    try:
        temp_file.write(response.content)
        return temp_file.name
    finally:
        temp_file.close()


def extract_pdf_pages(file_path: str) -> list[dict]:
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    extracted_pages = []

    for index, page in enumerate(pages):
        content = (page.page_content or "").strip()
        if not content:
            continue
        metadata = dict(page.metadata or {})
        metadata["page"] = metadata.get("page", index)
        extracted_pages.append({"content": content, "metadata": metadata})

    return extracted_pages


def extract_docx_pages(file_path: str) -> list[dict]:
    document = DocxDocument(file_path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    content = "\n\n".join(paragraphs)
    return [{"content": content, "metadata": {"page": 0, "source": "docx"}}] if content else []


def extract_document_pages(file_path: str, doc_name: str) -> list[dict]:
    lower_name = doc_name.lower()
    if lower_name.endswith(".pdf"):
        return extract_pdf_pages(file_path)
    if lower_name.endswith(".docx"):
        return extract_docx_pages(file_path)
    return []


def get_chunk_settings(page_count: Optional[int]) -> tuple[int, int]:
    if page_count and page_count >= LARGE_DOC_PAGE_THRESHOLD:
        return LARGE_DOC_CHUNK_SIZE, LARGE_DOC_CHUNK_OVERLAP
    return CHUNK_SIZE, CHUNK_OVERLAP


def split_document_pages(pages: list[dict], page_count: Optional[int] = None) -> list[dict]:
    chunk_size, chunk_overlap = get_chunk_settings(page_count)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["content"])
        for index, content in enumerate(page_chunks):
            clean_content = content.strip()
            if not clean_content:
                continue
            metadata = dict(page["metadata"])
            metadata["chunk_index"] = index
            chunks.append({"content": clean_content, "metadata": metadata})

    return chunks


def build_embedding_ready_text(chunk: dict, doc_name: str) -> str:
    metadata = chunk.get("metadata") or {}
    page_value = metadata.get("page")
    page_label = ""
    if isinstance(page_value, int):
        page_label = f" | Pagina {page_value + 1}"
    elif page_value is not None:
        page_label = f" | Pagina {page_value}"

    content = (chunk.get("content") or "").strip()
    return f"Titulo del documento: {doc_name}{page_label}\n\n{content}"


def build_document_preview_text(chunks: list[dict], max_chars: int = 60000) -> str:
    preview_parts = []
    current_length = 0

    for chunk in chunks:
        content = chunk["content"]
        if current_length + len(content) > max_chars:
            break
        preview_parts.append(content)
        current_length += len(content)

    return "\n\n".join(preview_parts)


def insert_chunk_batch(document_id: str, doc_name: str, chunks: list[dict], start: int, batch_size: int) -> int:
    batch = chunks[start:start + batch_size]
    texts = [build_embedding_ready_text(chunk, doc_name) for chunk in batch]
    titles = [doc_name for _ in batch]

    for attempt in range(EMBEDDING_MAX_RETRIES + 1):
        try:
            vectors = embed_documents_with_fallback(
                texts,
                task_type="RETRIEVAL_DOCUMENT",
                titles=titles,
            )
            break
        except Exception as exc:
            if is_daily_embedding_quota_error(str(exc)):
                raise DailyEmbeddingQuotaError(str(exc)) from exc

            if not is_quota_error(str(exc)):
                raise

            if attempt >= EMBEDDING_MAX_RETRIES:
                raise EmbeddingRateLimitError(str(exc)) from exc

            wait_seconds = extract_retry_delay_seconds(str(exc)) or EMBEDDING_RATE_LIMIT_PAUSE_SECONDS
            update_index_status(
                document_id,
                index_message=f"Limite temporal de embeddings. Reintento en {wait_seconds} segundos.",
            )
            time.sleep(wait_seconds)

    rows = [
        {
            "document_id": document_id,
            "content": chunk["content"],
            "embedding": vector,
            "metadata": chunk["metadata"],
        }
        for chunk, vector in zip(batch, vectors)
    ]
    supabase.table("document_chunks").insert(rows).execute()
    return len(rows)


def index_document_sync(document_id: str) -> None:
    temp_path = None

    try:
        res = supabase.table("documents").select("*").eq("id", document_id).single().execute()
        doc_data = res.data
        if not doc_data:
            raise ValueError(f"Documento {document_id} no encontrado.")

        doc_name = doc_data.get("name", "")
        lower_name = doc_name.lower()

        if lower_name.endswith((".jpg", ".jpeg", ".png")):
            update_index_status(
                document_id,
                index_status="skipped",
                index_progress=100,
                index_message="Imagen lista para consultas directas con @ en el chat.",
                index_error=None,
            )
            return

        update_index_status(
            document_id,
            index_status="processing",
            index_progress=5,
            index_message="Descargando archivo para indexacion.",
            index_error=None,
        )

        temp_path = download_document_to_temp(doc_data)

        update_index_status(
            document_id,
            index_progress=20,
            index_message="Extrayendo texto del documento.",
        )
        pages = extract_document_pages(temp_path, doc_name)
        page_count = doc_data.get("page_count") or (len(pages) if pages else None)

        if not pages:
            update_index_status(
                document_id,
                index_status="error",
                index_progress=0,
                index_message="No se pudo extraer texto indexable del documento.",
                index_error="Documento sin texto extraible. Si es escaneado, requiere OCR especializado.",
                page_count=page_count,
            )
            return

        update_index_status(
            document_id,
            index_progress=35,
            index_message=(
                "Dividiendo contenido en fragmentos amplios."
                if page_count and page_count >= LARGE_DOC_PAGE_THRESHOLD
                else "Dividiendo contenido en fragmentos."
            ),
            page_count=page_count,
        )
        chunks = split_document_pages(pages, page_count)

        if not chunks:
            update_index_status(
                document_id,
                index_status="error",
                index_progress=0,
                index_message="No se generaron fragmentos indexables.",
                index_error="El documento no produjo chunks validos.",
                page_count=page_count,
            )
            return

        existing_chunks_res = (
            supabase.table("document_chunks")
            .select("id", count="exact")
            .eq("document_id", document_id)
            .execute()
        )
        existing_chunk_count = existing_chunks_res.count or 0

        total_chunks = len(chunks)
        inserted = min(existing_chunk_count, total_chunks)

        if inserted >= total_chunks:
            preview_text = build_document_preview_text(chunks)
            update_index_status(
                document_id,
                content=preview_text or doc_data.get("content"),
                status="ready",
                index_status="indexed",
                index_progress=100,
                index_message="Documento indexado y disponible para busqueda semantica.",
                chunk_count=total_chunks,
                indexed_at=datetime.now(timezone.utc).isoformat(),
                index_error=None,
                page_count=page_count,
            )
            return

        if inserted > 0:
            update_index_status(
                document_id,
                index_progress=40 + int((inserted / total_chunks) * 55),
                index_message=f"Reanudando embeddings desde {inserted} de {total_chunks} fragmentos.",
                chunk_count=inserted,
            )

        for start in range(inserted, total_chunks, INDEX_BATCH_SIZE):
            inserted += insert_chunk_batch(document_id, doc_name, chunks, start, INDEX_BATCH_SIZE)
            progress = 40 + int((inserted / total_chunks) * 55)
            update_index_status(
                document_id,
                index_progress=min(progress, 95),
                index_message=f"Generando embeddings: {inserted} de {total_chunks} fragmentos.",
                chunk_count=inserted,
            )

        preview_text = build_document_preview_text(chunks)
        update_index_status(
            document_id,
            content=preview_text or doc_data.get("content"),
            status="ready",
            index_status="indexed",
            index_progress=100,
            index_message="Documento indexado y disponible para busqueda semantica.",
            chunk_count=total_chunks,
            indexed_at=datetime.now(timezone.utc).isoformat(),
            index_error=None,
            page_count=page_count,
        )
    except DailyEmbeddingQuotaError as exc:
        current_progress = 0
        current_chunk_count = 0
        try:
            current_doc = (
                supabase.table("documents")
                .select("index_progress,chunk_count")
                .eq("id", document_id)
                .single()
                .execute()
            )
            current_progress = int((current_doc.data or {}).get("index_progress") or 0)
            current_chunk_count = int((current_doc.data or {}).get("chunk_count") or 0)
        except Exception:
            current_progress = 0

        update_index_status(
            document_id,
            status="ready",
            index_status="paused",
            index_progress=current_progress,
            index_message="Indexacion pausada por limite diario de embeddings. Se reanudara cuando la cuota se restablezca.",
            index_error=str(exc),
            chunk_count=current_chunk_count,
        )
    except EmbeddingRateLimitError as exc:
        current_progress = 0
        current_chunk_count = 0
        try:
            current_doc = (
                supabase.table("documents")
                .select("index_progress,chunk_count")
                .eq("id", document_id)
                .single()
                .execute()
            )
            current_progress = int((current_doc.data or {}).get("index_progress") or 0)
            current_chunk_count = int((current_doc.data or {}).get("chunk_count") or 0)
        except Exception:
            current_progress = 0

        update_index_status(
            document_id,
            status="ready",
            index_status="paused",
            index_progress=current_progress,
            index_message="Indexacion pausada temporalmente por limite de embeddings. Se reintentara mas tarde.",
            index_error=str(exc),
            chunk_count=current_chunk_count,
        )
    except Exception as exc:
        current_progress = 0
        try:
            current_doc = supabase.table("documents").select("index_progress").eq("id", document_id).single().execute()
            current_progress = int((current_doc.data or {}).get("index_progress") or 0)
        except Exception:
            current_progress = 0

        update_index_status(
            document_id,
            status="error",
            index_status="error",
            index_progress=current_progress,
            index_message="Error durante la indexacion.",
            index_error=str(exc),
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def index_pending_sync(limit: int = INDEX_MAX_PENDING) -> int:
    res = (
        supabase.table("documents")
        .select("id")
        .eq("index_status", "uploaded")
        .order("uploaded_at", desc=False)
        .limit(limit)
        .execute()
    )
    docs = res.data or []

    for doc in docs:
        index_document_sync(doc["id"])

    return len(docs)


def get_or_upload_to_google(document_id: str):
    res = supabase.table("documents").select("*").eq("id", document_id).single().execute()
    doc_data = res.data
    if not doc_data:
        raise Exception(f"Documento {document_id} no encontrado.")

    google_uri = doc_data.get("google_file_uri")
    if google_uri:
        try:
            file_name = google_uri.split("/")[-1]
            google_file = genai.get_file(f"files/{file_name}")
            if google_file.state.name == "ACTIVE":
                return google_file
        except Exception:
            pass

    file_path_in_storage = (doc_data.get("file_path") or "").replace("\\", "/")
    file_url = supabase.storage.from_("documents").get_public_url(file_path_in_storage)

    response = requests.get(file_url, timeout=60)
    response.raise_for_status()

    temp_ext = os.path.splitext(doc_data.get("name", ""))[1] or ".pdf"
    temp_filename = f"temp_{document_id}{temp_ext}"

    with open(temp_filename, "wb") as f:
        f.write(response.content)

    try:
        new_google_file = genai.upload_file(path=temp_filename, display_name=doc_data.get("name", "Documento"))
        started_at = time.time()

        while new_google_file.state.name == "PROCESSING":
            if time.time() - started_at > GOOGLE_FILE_PROCESS_TIMEOUT:
                raise TimeoutError("La carga del documento hacia Gemini tardo demasiado.")
            time.sleep(GOOGLE_POLL_SECONDS)
            new_google_file = genai.get_file(new_google_file.name)

        if new_google_file.state.name != "ACTIVE":
            raise Exception(f"El archivo no quedo listo en Gemini. Estado: {new_google_file.state.name}")

        supabase.table("documents").update({"google_file_uri": new_google_file.uri}).eq("id", document_id).execute()
        return new_google_file
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


def build_excerpt(content: str, terms: list[str], radius: int = 260) -> str:
    normalized_content = normalize_text(content)
    match_index = -1

    for term in terms:
        idx = normalized_content.find(term)
        if idx != -1:
            match_index = idx
            break

    if match_index == -1:
        excerpt = content[:radius * 2]
    else:
        start = max(0, match_index - radius)
        end = min(len(content), match_index + radius)
        excerpt = content[start:end]

    excerpt = re.sub(r"\s+", " ", excerpt).strip()
    return f"...{excerpt}..." if excerpt else ""


def search_documents_by_text(query: str) -> list[dict]:
    terms = extract_terms(query)
    if not terms:
        return []

    res = supabase.table("documents").select("id,name,content").limit(100).execute()
    docs = res.data or []
    ranked = []

    for doc in docs:
        content = doc.get("content") or ""
        normalized_name = normalize_text(doc.get("name", ""))
        normalized_content = normalize_text(content)
        score = 0
        content_score = 0
        name_score = 0

        for term in terms:
            name_hits = normalized_name.count(term)
            content_hits = normalized_content.count(term)
            name_score += name_hits * 8
            content_score += content_hits

        score = name_score + content_score

        if score > 0:
            has_real_content = bool(content) and not is_placeholder_content(content)
            ranked.append({
                "id": doc["id"],
                "name": doc["name"],
                "score": score,
                "excerpt": build_excerpt(content, terms) if has_real_content else "",
                "content_match": content_score > 0 and has_real_content,
                "name_match": name_score > 0,
                "has_real_content": has_real_content,
            })

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:4]


def enrich_chunk_matches(matches: list[dict]) -> list[dict]:
    document_ids = list({match["document_id"] for match in matches if match.get("document_id")})
    if not document_ids:
        return []

    docs_res = (
        supabase.table("documents")
        .select("id,name,index_status")
        .in_("id", document_ids)
        .execute()
    )
    docs_by_id = {doc["id"]: doc for doc in (docs_res.data or [])}
    enriched = []

    for match in matches:
        doc = docs_by_id.get(match.get("document_id"))
        if not doc:
            continue
        enriched.append({
            "id": match.get("id"),
            "document_id": match.get("document_id"),
            "name": doc.get("name", "Documento sin nombre"),
            "content": match.get("content") or "",
            "similarity": float(match.get("similarity") or 0),
        })

    return enriched


def search_documents_semantic(query: str, match_count: int = 8, api_key: Optional[str] = None) -> list[dict]:
    query_embedding = embed_query_with_fallback(
        query,
        api_key=api_key,
        task_type="RETRIEVAL_QUERY",
    )
    response = supabase.rpc(
        "match_document_chunks_hybrid",
        {
            "query_text": query,
            "query_embedding": query_embedding,
            "match_threshold": 0.12,
            "match_count": match_count,
            "is_structural_query": looks_like_structural_query(query),
        },
    ).execute()
    return enrich_chunk_matches(response.data or [])


def search_documents_semantic_with_failover(query: str, chat_slot: ChatUsageReservation, match_count: int = 8) -> tuple[list[dict], str]:
    if chat_slot.tier == "bonus":
        return search_documents_semantic(query, match_count=match_count, api_key=GOOGLE_BONUS_API_KEY), "bonus"

    try:
        return search_documents_semantic(query, match_count=match_count), "paid"
    except Exception as exc:
        if can_use_bonus_failover() and is_failover_eligible_error(str(exc)):
            return search_documents_semantic(query, match_count=match_count, api_key=GOOGLE_BONUS_API_KEY), "bonus_failover"
        raise


def build_semantic_evidence_blocks(matches: list[dict], max_chars: int = 9000) -> str:
    blocks = []
    current_length = 0

    for index, match in enumerate(matches, start=1):
        content = re.sub(r"\s+", " ", match["content"]).strip()
        block = (
            f"Fuente {index}\n"
            f"Documento: {match['name']}\n"
            f"Similitud: {match['similarity']:.3f}\n"
            f"Extracto: {content}"
        )
        if current_length + len(block) > max_chars:
            break
        blocks.append(block)
        current_length += len(block)

    return "\n\n".join(blocks)


def dedupe_sources(matches: list[dict]) -> list[dict]:
    seen = set()
    sources = []

    for match in matches:
        document_id = match["document_id"]
        if document_id in seen:
            continue
        seen.add(document_id)
        sources.append({"id": document_id, "name": match["name"]})

    return sources


def build_text_search_fallback_answer(matches: list[dict], query: str) -> str:
    lines = [f"En la base documental si se encontraron coincidencias para: \"{query}\"."]

    for item in matches[:3]:
        excerpt = item["excerpt"] or "Coincidencia detectada por nombre de documento."
        lines.append(f"- {item['name']}: {excerpt}")

    lines.append("Respuesta generada en modo de respaldo por limite de cuota de Gemini.")
    return "\n".join(lines)


def build_name_only_guidance(matches: list[dict]) -> str:
    top_doc = matches[0]["name"]
    return (
        f"Encontre un documento relacionado por nombre: \"{top_doc}\", pero no tengo suficiente contenido indexado "
        "en la busqueda global para responder con precision. Si quieres analizar ese archivo en detalle, "
        "mencionalo con @ en el chat para abrir el modo de documento."
    )


def general_chat_fallback(message: str) -> str:
    normalized = normalize_text(message).strip(" ?!.,")
    if normalized in ("hola", "hello", "helloo", "buenas", "buenos dias", "buenas tardes", "buenas noches"):
        return "Hola. Soy SIGED-IA, tu asistente juridico. Puedo ayudarte con preguntas generales o buscar informacion dentro de tus documentos."
    if normalized == "quien eres":
        return "Soy SIGED-IA, un asistente juridico para consultar documentos y responder preguntas basadas en tu base documental."
    return "Puedo ayudarte con consultas juridicas generales y con la busqueda de informacion dentro de los documentos que has subido."


def no_documentary_evidence_answer() -> str:
    return (
        "No encontre evidencia suficiente en la base documental para responder esa consulta. "
        "Prueba con otro nombre, palabra clave o menciona el archivo con @ para analizarlo directamente."
    )


def generate_text_answer_with_failover(prompt: str, chat_slot: ChatUsageReservation, mode_label: str) -> tuple[str, str]:
    if chat_slot.tier == "bonus":
        answer, model_name = generate_bonus_text_with_fallback(BONUS_GENERAL_MODEL_CANDIDATES, prompt)
        return prepend_bonus_notice(answer, chat_slot), f"{model_name} (Bonus {mode_label})"

    try:
        response, model_name = generate_with_fallback(GENERAL_MODEL_CANDIDATES, prompt)
        return extract_response_text(response), f"{model_name} ({mode_label})"
    except Exception as exc:
        if can_use_bonus_failover() and is_failover_eligible_error(str(exc)):
            answer, model_name = generate_bonus_text_with_fallback(BONUS_GENERAL_MODEL_CANDIDATES, prompt)
            return answer, f"{model_name} (API Secundaria {mode_label})"
        raise


def generate_document_answer_with_failover(document_id: str, prompt: str, chat_slot: ChatUsageReservation) -> tuple[str, str]:
    if chat_slot.tier == "bonus":
        doc_record = supabase.table("documents").select("id,name,file_path").eq("id", document_id).single().execute()
        doc_data = doc_record.data or {}
        temp_path = download_document_to_temp(doc_data)
        try:
            answer, model_name = generate_bonus_document_with_fallback(temp_path, doc_data.get("name", "Documento"), prompt)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return prepend_bonus_notice(answer, chat_slot), f"{model_name} (Bonus Doc Mode)"

    try:
        google_file = get_or_upload_to_google(document_id)
        response, model_name = generate_with_fallback(DOC_MODEL_CANDIDATES, [google_file, prompt])
        return extract_response_text(response), f"{model_name} (Doc Mode)"
    except Exception as exc:
        if not (can_use_bonus_failover() and is_failover_eligible_error(str(exc))):
            raise

        doc_record = supabase.table("documents").select("id,name,file_path").eq("id", document_id).single().execute()
        doc_data = doc_record.data or {}
        temp_path = download_document_to_temp(doc_data)
        try:
            answer, model_name = generate_bonus_document_with_fallback(temp_path, doc_data.get("name", "Documento"), prompt)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        return answer, f"{model_name} (API Secundaria Doc Mode)"


@app.post("/jobs/index-document/{document_id}", response_model=JobResponse, status_code=202)
async def queue_index_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    x_worker_secret: Optional[str] = Header(default=None),
):
    verify_worker_secret(x_worker_secret)
    background_tasks.add_task(index_document_sync, document_id)
    return JobResponse(queued=True, document_id=document_id, count=1)


@app.post("/jobs/index-pending", response_model=JobResponse, status_code=202)
async def queue_index_pending(
    background_tasks: BackgroundTasks,
    limit: int = INDEX_MAX_PENDING,
    x_worker_secret: Optional[str] = Header(default=None),
):
    verify_worker_secret(x_worker_secret)
    safe_limit = max(1, min(limit, 10))
    background_tasks.add_task(index_pending_sync, safe_limit)
    return JobResponse(queued=True, count=safe_limit)


@app.post("/ask")
async def ask_document(request: ChatRequest):
    try:
        chat_slot = reserve_daily_google_request_budget(request)

        if request.document_id:
            try:
                prompt = (
                    "Eres SIGED-IA, asistente legal experto. "
                    "Responde basandote unicamente en este documento. "
                    "Si la pregunta es general, resume el contenido principal del documento en lenguaje claro.\n"
                    f"Pregunta: {request.message}"
                )
                answer, model_name = generate_document_answer_with_failover(request.document_id, prompt, chat_slot)
                return {"answer": answer, "model": model_name}
            except Exception as exc:
                if is_quota_error(str(exc)):
                    raise HTTPException(
                        status_code=429,
                        detail="Se agotaron temporalmente las cuotas de los modelos configurados para analizar documentos completos.",
                    )
                raise

        if looks_like_general_chat(request.message):
            return {"answer": general_chat_fallback(request.message), "model": "Saludo", "sources": []}

        try:
            semantic_matches, semantic_mode = search_documents_semantic_with_failover(request.message, chat_slot)
            if semantic_matches:
                evidence_blocks = build_semantic_evidence_blocks(semantic_matches)
                prompt = (
                    "Eres SIGED-IA, un asistente juridico experto. "
                    "Responde usando solo la evidencia documental suministrada. "
                    "No inventes datos fuera de los extractos. "
                    "Si la evidencia es parcial o insuficiente, dilo claramente. "
                    "Menciona los documentos fuente cuando sustenten la respuesta.\n\n"
                    f"Pregunta del usuario: {request.message}\n\n"
                    f"Evidencia documental recuperada por busqueda semantica/hibrida:\n{evidence_blocks}"
                )
                answer, model_label = generate_text_answer_with_failover(
                    prompt,
                    chat_slot,
                    "RAG Semantico" if semantic_mode != "bonus_failover" else "RAG Semantico con API Secundaria",
                )
                return {
                    "answer": answer,
                    "model": model_label,
                    "sources": dedupe_sources(semantic_matches),
                }
        except Exception as exc:
            if is_quota_error(str(exc)):
                matches = search_documents_by_text(request.message)
                return {
                    "answer": build_text_search_fallback_answer(matches, request.message) if matches else no_documentary_evidence_answer(),
                    "model": "Busqueda Textual (Respaldo local)",
                    "sources": [{"id": item["id"], "name": item["name"]} for item in matches],
                }

            print(f"Error en busqueda semantica, usando respaldo textual: {exc}")

        matches = search_documents_by_text(request.message)
        if matches:
            if not any(match["content_match"] for match in matches):
                return {
                    "answer": build_name_only_guidance(matches),
                    "model": "Busqueda Textual",
                    "sources": [{"id": item["id"], "name": item["name"]} for item in matches],
                }

            try:
                evidence_blocks = "\n\n".join(
                    [f"Documento: {item['name']}\nExtracto: {item['excerpt']}" for item in matches if item["content_match"]]
                )
                prompt = (
                    "Eres SIGED-IA, un asistente juridico experto. "
                    "Responde usando solo la evidencia documental suministrada. "
                    "Si la evidencia no es suficiente, dilo claramente. "
                    "Indica de forma clara si la informacion aparece o no en la base documental.\n\n"
                    f"Pregunta del usuario: {request.message}\n\n"
                    f"Evidencia documental:\n{evidence_blocks}"
                )
                answer, model_label = generate_text_answer_with_failover(prompt, chat_slot, "Busqueda Global")
                return {
                    "answer": answer,
                    "model": model_label,
                    "sources": [{"id": item["id"], "name": item["name"]} for item in matches],
                }
            except Exception as exc:
                if is_quota_error(str(exc)):
                    return {
                        "answer": build_text_search_fallback_answer(matches, request.message),
                        "model": "Busqueda Textual (Respaldo local)",
                        "sources": [{"id": item["id"], "name": item["name"]} for item in matches],
                    }
                raise

        return {
            "answer": no_documentary_evidence_answer(),
            "model": "Busqueda Documental",
            "sources": [],
        }

    except TimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=f"{str(e)} Intenta una pregunta mas puntual o vuelve a intentar en unos segundos.",
        )
    except HTTPException:
        raise
    except Exception as e:
        message = str(e)
        if "timed out" in message.lower() or "504" in message.lower():
            raise HTTPException(
                status_code=504,
                detail="Gemini excedio el tiempo limite al analizar el documento. Intenta una pregunta mas especifica o vuelve a intentar.",
            )
        if is_quota_error(message):
            raise HTTPException(
                status_code=429,
                detail="Se agotaron temporalmente las cuotas de los modelos configurados. Espera un momento y vuelve a intentar.",
            )
        raise HTTPException(status_code=500, detail=message)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
