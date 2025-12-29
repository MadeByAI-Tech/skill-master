import json
import re
from typing import Any
from pipeline.phase1.function.var import MASTER_SYSTEM_PROMPT
from llm import chat

def extract_all_components(
        job_title: str,
        job_function: str,
        job_description: str,
        ) -> dict[str, Any]:
    
    """
    Pipeline to extract job information using LLM

    - Sends job detail to LLM using `system_prompt`
    - Parses JSON response and do processing which includes:
        - Clean and validate `job_level`
        - Normalize skill titles
        - Ensure valid skill types
        - Truncate long evidence fields
    - Returns:
        dict[str, Any]: Structured extraction result containing:
        job_title_cleaned, job_level, skills, cwf_items, etc.
    """

    system_prompt = MASTER_SYSTEM_PROMPT   # defined earlier
    user_prompt   = f"""
Extract all required components.

job_title:
{job_title}

job_function:
{job_function}

job_description:
{job_description}

Return ONLY the JSON object defined in the system prompt.
""".strip()
    
    resp = chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": user_prompt},
        ]
    )

    content = resp.choices[0].message.content.strip() # type: ignore
    data = _extract_json_object(content)

    # --------------------------
    # POST-PROCESSING
    # --------------------------
    if "skills" in data and isinstance(data["skills"], list):
        s: dict[str,str]
        for s in data["skills"]: # type: ignore
            title:str = s.get("skill_title", "") # type: ignore
            s["skill_title"] = _translate_if_needed(title) # type: ignore
            if s.get("skill_type") not in {"TSC", "GSC", "Cert."}: # type: ignore
                s["skill_type"] = "TSC"
            if "evidence" in s:
                s["evidence"] = s["evidence"][:200]

    # ensure job_level integer
    jl = data.get("job_level")
    if not isinstance(jl, int):
        data["job_level"] = None

    return data


# ------------------------------------
# JSON Safe Extractor
# ------------------------------------
def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}

    start = text.find("{")
    end   = text.rfind("}")

    if start == -1 or end == -1:
        return {}

    block = text[start:end+1]

    try:
        return json.loads(block)
    except:
        try:
            return json.loads(block.replace("\n", " "))
        except:
            return {}
        
# ------------------------------------
# English Detector + Fallback Translator
# ------------------------------------
_ASCII_PATTERN = re.compile(r"^[A-Za-z0-9 ,.\-/()]+$")

def _is_english(text: str) -> bool:
    return bool(_ASCII_PATTERN.fullmatch(text))

def _translate_if_needed(text: str) -> str:
    if _is_english(text):
        return text
    
    resp = chat(
        messages=[{"role": "user", "content": f"Translate to concise professional English:\n{text}"}]
    )
    return resp.choices[0].message.content.strip() # type: ignore