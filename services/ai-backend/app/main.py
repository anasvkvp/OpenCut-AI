"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routes import eenoki, analyze, audio, background, broll, command, dub, engagement, export, factcheck, generate, llm, podcast, sarvam, search, setup, smallest, template, transcribe, transcribe_ws, translate, tts, turboquant, video, youtube

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    logger.info("OpenCut AI Backend starting on %s:%d", settings.HOST, settings.PORT)
    logger.info(
        "Microservice URLs: whisper=%s  tts=%s  image=%s  ollama=%s",
        settings.WHISPER_SERVICE_URL,
        settings.TTS_SERVICE_URL,
        settings.IMAGE_SERVICE_URL,
        settings.OLLAMA_URL,
    )
    logger.info(
        "Sarvam AI: %s (key %s)",
        settings.SARVAM_API_BASE_URL,
        "configured" if settings.SARVAM_API_KEY else "NOT configured",
    )
    logger.info(
        "Smallest AI: %s (key %s)",
        settings.SMALLEST_API_BASE_URL,
        "configured" if settings.SMALLEST_API_KEY else "NOT configured",
    )
    logger.info(
        "TurboQuant: service=%s  kv_bits=%d  memory_budget=%s  model_tier=%s",
        settings.TURBOQUANT_SERVICE_URL,
        settings.KV_CACHE_BITS,
        settings.AI_MEMORY_BUDGET,
        settings.AI_MODEL_TIER,
    )
    logger.info("CLIP embeddings: %s", settings.CLIP_SERVICE_URL)
    logger.info("NLLB translate: %s", settings.TRANSLATE_SERVICE_URL)
    logger.info(
        "Seedance: %s (key %s)",
        settings.SEEDANCE_API_BASE_URL,
        "configured" if settings.SEEDANCE_API_KEY else "NOT configured",
    )
    yield
    logger.info("Shutdown complete.")


app = FastAPI(
    title="OpenCut AI Backend",
    description="AI-powered backend for the OpenCut AI video editor",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for generated content
app.mount("/generated", StaticFiles(directory=settings.GENERATED_DIR), name="generated")

# Register routers
app.include_router(transcribe.router)
app.include_router(transcribe_ws.router)
app.include_router(llm.router)
app.include_router(command.router)
app.include_router(analyze.router)
app.include_router(tts.router)
app.include_router(generate.router)
app.include_router(export.router)
app.include_router(audio.router)
app.include_router(setup.router)
app.include_router(factcheck.router)
app.include_router(podcast.router)
app.include_router(sarvam.router)
app.include_router(smallest.router)
app.include_router(template.router)
app.include_router(turboquant.router)
app.include_router(video.router)
app.include_router(youtube.router)
app.include_router(engagement.router)
app.include_router(search.router)
app.include_router(translate.router)
app.include_router(dub.router)
app.include_router(background.router)
app.include_router(broll.router)
app.include_router(eenoki.router)


@app.get("/health")
async def health() -> dict:
    import asyncio
    import httpx
    import psutil

    def check_url(name: str, url: str):
        try:
            with httpx.Client(timeout=5) as client:
                r = client.get(url)
                if r.status_code == 200:
                    return name, True, r
        except Exception:
            pass
        return name, False, None

    checks = [
        ("ollama", f"{settings.OLLAMA_URL}/api/tags"),
        ("whisper", f"{settings.WHISPER_SERVICE_URL}/health"),
        ("clip", f"{settings.CLIP_SERVICE_URL}/health"),
    ]

    results = await asyncio.gather(
        *[asyncio.to_thread(check_url, name, url) for name, url in checks]
    )

    services = []
    models = []

    for name, ok, resp in results:
        if ok:
            services.append(name)
            if name == "ollama" and resp is not None:
                try:
                    models = [m.get("name", "") for m in resp.json().get("models", [])]
                except Exception:
                    pass

    mem = psutil.virtual_memory()

    return {
        "available": True,
        "models": models,
        "services": services,
        "gpuAvailable": False,
        "memoryUsage": {
            "ram": {
                "usedMb": round(mem.used / 1024 / 1024),
                "totalMb": round(mem.total / 1024 / 1024),
                "percent": mem.percent,
            },
            "gpu": None,
        },
        "version": "0.1.0",
    }

@app.get("/services/health")
async def services_health() -> dict:
    """Detailed health check for all downstream services.

    Returns per-service status including model info so the frontend
    doesn't need to make cross-origin requests to each service.
    """
    import asyncio
    import httpx

    results: dict[str, dict] = {}

    async def _check(name: str, url: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{url}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    entry: dict = {"status": "running"}

                    # Extract model info from nested structures
                    if isinstance(data.get("model"), dict):
                        entry["model_loaded"] = data["model"].get("loaded") is True
                        entry["model_installed"] = data["model"].get("installed") is not False
                        entry["model_name"] = data["model"].get("model_name") or data["model"].get("model_size")
                        entry["model_size"] = data["model"].get("model_size") or data["model"].get("model_name")
                    if isinstance(data.get("models"), dict) and isinstance(data["models"].get("diffusion"), dict):
                        diff = data["models"]["diffusion"]
                        entry["model_loaded"] = diff.get("loaded") is True
                        entry["model_installed"] = diff.get("installed") is not False
                        entry["model_name"] = diff.get("model_name")

                    if data.get("install_command"):
                        entry["install_command"] = data["install_command"]
                    if data.get("supported_languages"):
                        entry["supported_languages"] = data["supported_languages"]
                    if data.get("version"):
                        entry["version"] = data["version"]
                    if data.get("service"):
                        entry["detail"] = data["service"]

                    results[name] = entry
                else:
                    results[name] = {"status": "error", "detail": f"HTTP {resp.status_code}"}
        except Exception:
            results[name] = {"status": "stopped", "detail": "Not reachable"}

    async def _check_ollama() -> None:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(settings.OLLAMA_URL)
                if resp.status_code != 200:
                    results["ollama"] = {"status": "stopped", "detail": "Not reachable"}
                    return
                tags_resp = await client.get(f"{settings.OLLAMA_URL}/api/tags")
                models = []
                if tags_resp.status_code == 200:
                    models = tags_resp.json().get("models", [])
                results["ollama"] = {"status": "running", "models": models}
        except Exception:
            results["ollama"] = {"status": "stopped", "detail": "Not reachable"}

    await asyncio.gather(
        _check("backend", f"http://localhost:{settings.PORT}"),
        _check_ollama(),
        _check("whisper", settings.WHISPER_SERVICE_URL),
        _check("tts", settings.TTS_SERVICE_URL),
        _check("image", settings.IMAGE_SERVICE_URL),
        _check("speaker", settings.SPEAKER_SERVICE_URL),
        _check("face", settings.FACE_SERVICE_URL),
        _check("turboquant", settings.TURBOQUANT_SERVICE_URL),
        _check("clip", settings.CLIP_SERVICE_URL),
        _check("translate", settings.TRANSLATE_SERVICE_URL),
    )

    # Backend is always running if we're responding
    results["backend"] = {"status": "running", "version": "0.1.0"}

    return results
