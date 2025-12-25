from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from typing import Iterable
import os
from pathlib import Path


if os.environ.get("MODE") == "dev":
    assert "OPENAI_API_KEY_FROM" in os.environ, f"Missing OPENAI_API_KEY_FROM in dev mode"
    secret_path:Path = Path(os.environ["OPENAI_API_KEY_FROM"])
    with open(secret_path, "r") as f:
        os.environ["OPENAI_API_KEY"] = f.read().strip()

MODEL:str = os.environ["OPENAI_MODEL"]

def chat(messages:Iterable[ChatCompletionMessageParam]) -> ChatCompletion:
    _client = OpenAI()
    resp = _client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=messages
    )
    return resp

async def async_chat(messages:Iterable[ChatCompletionMessageParam]) -> ChatCompletion:
    _client = AsyncOpenAI()
    resp = await _client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        messages=messages
    )
    return resp
