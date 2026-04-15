import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

# Create Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Configure Embeddings
embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=GOOGLE_API_KEY
)

def process_chunk(chunk_data):
    """Función para procesar un solo fragmento (generar embedding)"""
    text, doc_id, metadata = chunk_data
    try:
        vector = embeddings_model.embed_query(text)
        return {
            "document_id": doc_id,
            "content": text,
            "embedding": vector,
            "metadata": metadata
        }
    except Exception as e:
        print(f"Error procesando fragmento: {e}")
        return None

def ingest_pdf(file_path: str, document_id: str):
    start_time = time.time()
    print(f"--- Iniciando Ingesta Turbo V2 ---")
    print(f"Archivo: {file_path}")
    
    try:
        # 1. Carga del PDF
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # 2. Fragmentación
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)
        total_chunks = len(chunks)
        print(f"Documento dividido en {total_chunks} fragmentos.")

        # 3. Procesamiento en Paralelo (Multi-threading)
        # Preparamos los datos para los hilos
        chunks_data = [(c.page_content, document_id, c.metadata) for c in chunks]
        
        processed_chunks = []
        # Usamos 10 hilos para acelerar las peticiones a la API de Google
        print(f"Generando embeddings en paralelo (10 hilos)...")
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(process_chunk, chunks_data))
            processed_chunks = [r for r in results if r is not None]

        # 4. Inserción Masiva en Supabase (Lotes de 50)
        print(f"Insertando {len(processed_chunks)} vectores en Supabase...")
        batch_size = 50
        for i in range(0, len(processed_chunks), batch_size):
            batch = processed_chunks[i : i + batch_size]
            supabase.table("document_chunks").insert(batch).execute()
            print(f"Progreso: {min(100, int((i + batch_size)/len(processed_chunks) * 100))}%")

        # 5. Finalización
        supabase.table("documents").update({
            "status": "ready"
        }).eq("id", document_id).execute()
        
        end_time = time.time()
        print(f"✅ Ingesta completada en {int(end_time - start_time)} segundos.")

    except Exception as e:
        print(f"❌ ERROR CRÍTICO EN INGESTA: {str(e)}")
        supabase.table("documents").update({"status": "error"}).eq("id", document_id).execute()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(1)
    ingest_pdf(sys.argv[1], sys.argv[2])
