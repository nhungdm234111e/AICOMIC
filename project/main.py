"""
FastAPI backend for generating comic-style panels with OpenAI gpt-image-1.
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from utils.file_saver import save_image
from utils.prompt_builder import build_prompt

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY) if API_KEY else None

BASE_DIR = Path(__file__).resolve().parent
GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="AI Comic Image Generator",
    description="Generate comic-style panels using OpenAI's image model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Scene description for the comic panel")


class GenerateResponse(BaseModel):
    status: str
    image_url: str
    file_path: str
    prompt_used: str


class ImageItem(BaseModel):
    file_path: str
    image_url: str


def _data_url_from_file(file_path: Path) -> str:
    data = file_path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.post("/generate", response_model=GenerateResponse, summary="Generate a new comic panel")
async def generate_image(payload: GenerateRequest) -> Dict[str, Any]:
    text = payload.text.strip()
    if len(text) < 10:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Text must be at least 10 characters long."},
        )
    if client is None:
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": "OPENAI_API_KEY is not configured in environment or .env file."},
        )

    prompt = build_prompt(text)

    try:
        response = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size="1024x1024",
        )
        image_base64 = response.data[0].b64_json
        if not image_base64:
            raise ValueError("No image data returned from OpenAI.")

        file_path = save_image(image_base64)
        data_url = f"data:image/png;base64,{image_base64}"

        return {
            "status": "success",
            "image_url": data_url,
            "file_path": file_path,
            "prompt_used": prompt,
        }
    except HTTPException:
        raise
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": f"Image generation failed: {exc}"},
        ) from exc


@app.get("/images", response_model=List[ImageItem], summary="List recently generated images")
async def list_images() -> List[Dict[str, str]]:
    files = sorted(
        GENERATED_DIR.glob("*.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    items: List[Dict[str, str]] = []
    for file in files[:20]:
        relative_path = str(file.relative_to(BASE_DIR))
        items.append({"file_path": relative_path, "image_url": _data_url_from_file(file)})
    return items


@app.get("/health", summary="Health check")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}
