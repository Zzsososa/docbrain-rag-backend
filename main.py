import os
import re
import time
import unicodedata
from pathlib import Path
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai
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
DOC_MODEL = os.getenv("GOOGLE_DOC_MODEL", "gemini-2.5-flash")
GENERAL_MODEL = os.getenv("GOOGLE_GENERAL_MODEL", "gemini-2.5-flash")
DOC_MODELS = os.getenv("GOOGLE_DOC_MODELS", DOC_MODEL)
GENERAL_MODELS = os.getenv("GOOGLE_GENERAL_MODELS", GENERAL_MODEL)
GOOGLE_MAX_RETRIES = int(os.getenv("GOOGLE_MAX_RETRIES", "2"))
GOOGLE_POLL_SECONDS = int(os.getenv("GOOGLE_POLL_SECONDS", "2"))
GOOGLE_FILE_PROCESS_TIMEOUT = int(os.getenv("GOOGLE_FILE_PROCESS_TIMEOUT", "90"))

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

app = FastAPI()


class ChatRequest(BaseModel):
    document_id: Optional[str] = None
    message: str


def parse_model_candidates(raw_models: str, fallback: str) -> list[str]:
    candidates = [item.strip() for item in raw_models.split(",") if item.strip()]
    if fallback and fallback not in candidates:
        candidates.append(fallback)
    return candidates


DOC_MODEL_CANDIDATES = parse_model_candidates(DOC_MODELS, DOC_MODEL)
GENERAL_MODEL_CANDIDATES = parse_model_candidates(GENERAL_MODELS, GENERAL_MODEL)


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


def looks_like_general_chat(message: str) -> bool:
    normalized = normalize_text(message).strip(" ?!.,")
    return normalized in CHAT_PATTERNS


def looks_like_general_knowledge(message: str) -> bool:
    normalized = normalize_text(message)
    return any(hint in normalized for hint in GENERAL_KNOWLEDGE_HINTS)


def should_search_knowledge_base(message: str) -> bool:
    normalized = normalize_text(message)
    return any(hint in normalized for hint in SEARCH_HINTS)


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
            if is_quota_error(str(exc)):
                continue
            raise

    raise last_error


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


@app.post("/ask")
async def ask_document(request: ChatRequest):
    try:
        if request.document_id:
            try:
                google_file = get_or_upload_to_google(request.document_id)
                response, model_name = generate_with_fallback(
                    DOC_MODEL_CANDIDATES,
                    [
                        google_file,
                        (
                            "Eres SIGED-IA, asistente legal experto. "
                            "Responde basandote unicamente en este documento. "
                            "Si la pregunta es general, resume el contenido principal del documento en lenguaje claro.\n"
                            f"Pregunta: {request.message}"
                        ),
                    ],
                )
                return {"answer": response.text, "model": f"{model_name} (Doc Mode)"}
            except Exception as exc:
                if is_quota_error(str(exc)):
                    raise HTTPException(
                        status_code=429,
                        detail="Se agotaron temporalmente las cuotas de los modelos configurados para analizar documentos completos.",
                    )
                raise

        if looks_like_general_chat(request.message):
            try:
                response, model_name = generate_with_fallback(
                    GENERAL_MODEL_CANDIDATES,
                    f"Eres SIGED-IA, un asistente juridico experto. Responde de forma breve y natural a esto: {request.message}",
                )
                return {"answer": response.text, "model": f"{model_name} (General Mode)", "sources": []}
            except Exception as exc:
                if is_quota_error(str(exc)):
                    return {"answer": general_chat_fallback(request.message), "model": "Modo General (Respaldo local)", "sources": []}
                raise

        if should_search_knowledge_base(request.message):
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
                    response, model_name = generate_with_fallback(
                        GENERAL_MODEL_CANDIDATES,
                        (
                            "Eres SIGED-IA, un asistente juridico experto. "
                            "Responde usando solo la evidencia documental suministrada. "
                            "Si la evidencia no es suficiente, dilo claramente. "
                            "Indica de forma clara si la informacion aparece o no en la base documental.\n\n"
                            f"Pregunta del usuario: {request.message}\n\n"
                            f"Evidencia documental:\n{evidence_blocks}"
                        ),
                    )
                    return {
                        "answer": response.text,
                        "model": f"{model_name} (Busqueda Global)",
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
                "answer": (
                    "No se encontraron coincidencias textuales en la base documental para esa consulta. "
                    "Puedes intentar con otro nombre, palabra clave o usar @ para preguntar por un documento especifico."
                ),
                "model": "Busqueda Textual",
                "sources": [],
            }

        if looks_like_general_knowledge(request.message):
            try:
                response, model_name = generate_with_fallback(
                    GENERAL_MODEL_CANDIDATES,
                    f"Eres SIGED-IA, un asistente juridico experto. Responde de forma clara y breve a esto: {request.message}",
                )
                return {"answer": response.text, "model": f"{model_name} (General Mode)", "sources": []}
            except Exception as exc:
                if is_quota_error(str(exc)):
                    return {"answer": general_chat_fallback(request.message), "model": "Modo General (Respaldo local)", "sources": []}
                raise

        try:
            response, model_name = generate_with_fallback(
                GENERAL_MODEL_CANDIDATES,
                f"Eres SIGED-IA, un asistente juridico experto. Responde de forma breve y util a esto: {request.message}",
            )
            return {"answer": response.text, "model": f"{model_name} (General Mode)", "sources": []}
        except Exception as exc:
            if is_quota_error(str(exc)):
                return {"answer": general_chat_fallback(request.message), "model": "Modo General (Respaldo local)", "sources": []}
            raise

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
