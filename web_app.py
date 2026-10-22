import os
import sys
import re
import json
import uuid
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
import anyio

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv
from google import genai

from security_manager import SecurityManager
from memory_manager import MemoryManager

# ==========================================
# 1. SETUP ENVIRONMENT & MANAGERS
# ==========================================

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("⚠️ Warning: GEMINI_API_KEY not found in environment. Please configure it in Hugging Face Space Settings or .env file.")

security = SecurityManager(enable_encryption=False, privacy_mode=False)
client = genai.Client(api_key=api_key) if api_key else None
memory = MemoryManager(gemini_client=client, security_manager=security, privacy_mode=False)

app = FastAPI(title="Nova AI Voice Assistant", description="Modern Web UI for Nova AI Voice Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_SYSTEM_INSTRUCTION = """
You are Nova, an ultra-fast, intelligent, friendly, and privacy-conscious AI voice assistant.

Rules:
- Keep answers concise, natural, and conversational (1-2 sentences unless details are requested).
- Use clear and simple spoken language without markdown formatting or asterisks.
- Address the user naturally if you know their name.
- Recall stored user facts seamlessly without explicitly reciting lists.
- If the user asks what you remember about them, give a warm and crisp summary.
"""

# WebSocket connection manager
active_websockets: List[WebSocket] = []

async def broadcast_event(event_type: str, data: Any):
    """Broadcasts real-time events to all connected UI clients."""
    message = json.dumps({"type": event_type, "data": data})
    for ws in list(active_websockets):
        try:
            await ws.send_text(message)
        except Exception:
            if ws in active_websockets:
                active_websockets.remove(ws)


# ==========================================
# 2. API ENDPOINTS
# ==========================================

@app.get("/api/status")
async def get_status():
    """Returns overall Nova status, wake word details, and security indicators."""
    profile = memory.get_user_profile()
    ltm = memory.get_all_long_term_memories()
    episodes = memory.get_episodic_history(limit=5)
    last_id = memory.get_last_interaction_id()

    return {
        "status": "online",
        "assistant_name": "Nova",
        "wake_word": "Hey Nova",
        "user_name": profile.get("name"),
        "total_memories": len(ltm),
        "total_episodes": len(episodes),
        "last_interaction_id": last_id,
        "api_key_configured": bool(api_key),
        "security": {
            "masked_key": security.mask_api_key(api_key) if (security and api_key) else "Not Configured",
            "encryption": security.enable_encryption,
            "privacy_mode": security.privacy_mode,
            "local_wake_word": True
        }
    }


@app.post("/api/chat")
async def process_chat(data: Dict[str, Any]):
    """Handles text input, evaluates direct memory commands or sends to Gemini."""
    user_text = data.get("message", "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if not client:
        return {
            "user": user_text,
            "reply": "Nova is running on Hugging Face Spaces, but GEMINI_API_KEY is not configured yet. Please add GEMINI_API_KEY under Space Settings -> Variables and Secrets.",
            "type": "configuration_needed"
        }

    previous_id = memory.get_last_interaction_id()

    # Handle standalone wake words or prefixes
    clean_lower = user_text.lower().strip()
    if clean_lower in ["hey nova", "nova", "hi nova", "hello nova", "hey nova!"]:
        ack = "I'm listening! What can I help you with?"
        return {"user": user_text, "reply": ack, "type": "greeting"}

    user_text_clean = re.sub(r"^(?:hey\s+nova|nova|hello\s+nova)[,\s]+", "", user_text, flags=re.IGNORECASE).strip()
    if not user_text_clean:
        user_text_clean = user_text

    # 1. Direct memory command check
    memory_reply = memory.handle_memory_command(user_text_clean)
    if memory_reply:
        memory.log_turn(user_text_clean, memory_reply, previous_id)
        await broadcast_event("memory_updated", {})
        return {
            "user": user_text,
            "reply": memory_reply,
            "type": "memory_command"
        }

    # 2. Build dynamically grounded system instruction
    active_instruction = memory.build_system_instruction(
        BASE_SYSTEM_INSTRUCTION,
        current_query=user_text_clean
    )

    # 3. Guard against prompt injection and scrub PII
    guarded_instruction, guarded_input = security.wrap_safe_prompt(
        active_instruction,
        user_text_clean
    )

    # 4. Request Gemini 3.8 Flash response via threadpool to prevent event loop blocking
    def _call_gemini(m_id, g_input, g_instr):
        if m_id is None:
            return client.interactions.create(
                model="gemini-3.8-flash",
                input=g_input,
                system_instruction=g_instr
            )
        return client.interactions.create(
            model="gemini-3.8-flash",
            input=g_input,
            previous_interaction_id=m_id,
            system_instruction=g_instr
        )

    try:
        response = await anyio.to_thread.run_sync(_call_gemini, previous_id, guarded_input, guarded_instruction)
    except Exception as e:
        memory.clear_session_id()
        try:
            response = await anyio.to_thread.run_sync(_call_gemini, None, guarded_input, guarded_instruction)
        except Exception as err:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {err}")

    new_interaction_id = response.id
    ai_text = response.output_text.strip()

    # Log turn to memory and trigger background learning
    memory.log_turn(user_text_clean, ai_text, new_interaction_id)
    await broadcast_event("chat_turn", {"user": user_text, "ai": ai_text})

    return {
        "user": user_text,
        "reply": ai_text,
        "type": "ai_response",
        "interaction_id": new_interaction_id
    }


@app.post("/api/audio")
async def process_audio(file: UploadFile = File(...)):
    """Receives audio recorded from browser microphone, transcribes via Gemini, and generates reply."""
    if not client:
        return {
            "transcript": "",
            "reply": "Nova is running on Hugging Face Spaces, but GEMINI_API_KEY is not configured yet. Please add GEMINI_API_KEY under Space Settings -> Variables and Secrets.",
            "type": "configuration_needed"
        }

    temp_dir = tempfile.gettempdir()
    temp_audio = os.path.join(temp_dir, f"browser_mic_{uuid.uuid4().hex[:8]}.webm")

    try:
        content = await file.read()
        with open(temp_audio, "wb") as f:
            f.write(content)

        # Upload audio to Gemini Transcribe via threadpool
        def _upload_and_transcribe(path):
            uploaded = client.files.upload(file=path)
            return client.interactions.create(
                model="gemini-3.5-transcribe",
                input=[
                    {
                        "type": "audio",
                        "uri": uploaded.uri,
                        "mime_type": uploaded.mime_type,
                    }
                ],
            )

        transcription = await anyio.to_thread.run_sync(_upload_and_transcribe, temp_audio)
        user_text = transcription.output_text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {e}")
    finally:
        # Privacy: Secure shred of temporary audio file
        security.secure_delete_file(temp_audio)

    if not user_text:
        return {"transcript": "", "reply": "I couldn't hear any clear speech. Please try speaking again."}

    # Process conversational text
    result = await process_chat({"message": user_text})
    return {
        "transcript": user_text,
        "reply": result["reply"],
        "type": result.get("type", "voice_response")
    }


@app.get("/api/memory")
async def get_memory_bank():
    """Returns the full 5-tier memory bank for UI inspection."""
    return {
        "profile": memory.get_user_profile(),
        "long_term_memories": memory.get_all_long_term_memories(),
        "episodic_history": memory.get_episodic_history(limit=10),
        "recent_history": memory.get_recent_history(limit=10)
    }


@app.post("/api/memory/add")
async def add_memory(data: Dict[str, Any]):
    """Manually adds a fact or preference to Nova's memory."""
    content = data.get("content", "").strip()
    category = data.get("category", "personal").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Memory content required")

    if category == "preference":
        memory.add_preference(content)
    elif category == "work_project":
        memory.add_project(content)
    else:
        memory.add_fact(content, category=category)

    await broadcast_event("memory_updated", {})
    return {"status": "success", "message": f"Added memory: {content}"}


@app.post("/api/memory/forget")
async def forget_memory_item(data: Dict[str, Any]):
    """Removes a specific memory item by query or ID."""
    query = data.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query or ID required")

    count = memory.forget_memory(query)
    await broadcast_event("memory_updated", {})
    return {"status": "success", "removed_count": count}


@app.post("/api/memory/reset")
async def reset_memory_bank():
    """Clears all persistent memory files."""
    memory.clear_all_memory()
    await broadcast_event("memory_updated", {})
    return {"status": "success", "message": "All memory cleared"}


@app.get("/api/memory/export")
async def export_memory_report():
    """Generates and serves a formatted markdown memory report."""
    export_path = memory.export_memory()
    return FileResponse(
        path=export_path,
        filename="nova_memory_export.md",
        media_type="text/markdown"
    )


@app.post("/api/security/privacy-mode")
async def toggle_privacy_mode(data: Dict[str, bool]):
    """Toggles ephemeral incognito mode on/off."""
    active = data.get("active", False)
    security.privacy_mode = active
    memory.privacy_mode = active
    await broadcast_event("status_changed", {"privacy_mode": active})
    return {"privacy_mode": active}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time bi-directional WebSocket connection for UI live events."""
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo or heartbeat
            await websocket.send_text(json.dumps({"type": "pong", "payload": data}))
    except WebSocketDisconnect:
        if websocket in active_websockets:
            active_websockets.remove(websocket)


# Mount static files
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the main single page web application."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Nova UI loading...</h1>")


if __name__ == "__main__":
    default_port = 7860 if os.getenv("SPACE_ID") else 8000
    port = int(os.getenv("PORT", default_port))
    print(f"\n🚀 Launching Nova Web UI at http://0.0.0.0:{port}")
    uvicorn.run("web_app:app", host="0.0.0.0", port=port, reload=False)
