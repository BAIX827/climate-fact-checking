"""FastAPI app: POST /verify, GET /health, and a static demo page at /."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import config

logging.basicConfig(level=logging.INFO)
STATIC = config.ROOT / "app" / "static"


class VerifyRequest(BaseModel):
    claim: str = Field(min_length=3, max_length=500)
    top_k: int = Field(config.TOP_K, ge=1, le=20)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "pipeline", None) is None:  # tests inject a fake pipeline
        from app.pipeline import Pipeline

        app.state.pipeline = Pipeline.from_config()
    yield


app = FastAPI(title="Climate Fact-Checking", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    p = request.app.state.pipeline
    return {"status": "ok", "passages": len(p.retriever.text)}


@app.post("/verify")
def verify(body: VerifyRequest, request: Request):
    claim = body.claim.strip()
    if not claim:
        raise HTTPException(422, "claim is empty")
    return request.app.state.pipeline.verify(claim, top_k=body.top_k)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
