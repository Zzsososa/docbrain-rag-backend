# DocBrain RAG Backend

FastAPI backend for DocBrain AI chat, document search, and Gemini document analysis.

## Render Deployment

Use these settings when creating a Render Web Service:

```bash
Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

Required environment variables:

```bash
NEXT_PUBLIC_SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
GOOGLE_AI_KEY=
```

Optional environment variables:

```bash
GOOGLE_DOC_MODEL=gemini-2.5-flash
GOOGLE_GENERAL_MODEL=gemini-2.5-flash
GOOGLE_MAX_RETRIES=2
GOOGLE_POLL_SECONDS=2
GOOGLE_FILE_PROCESS_TIMEOUT=90
```

After deployment, configure the frontend with:

```bash
PYTHON_BACKEND_URL=https://your-render-service.onrender.com
```
