# ==============================================================
# standard_jd_generator.py
# - Generate Standardized Job Description using:
#     job_title_normed + job_function
#     + skillset (with PL level)
#     + CWF topics (with key tasks)
# - Output: fixed-format JSON, 1–2 paragraphs, <= max_chars
# ==============================================================

import os, time, json, re
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============ OpenAI Client =========
os.environ.setdefault("OPENAI_API_KEY", "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")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# -------------------------------------------------
# SYSTEM PROMPT (fixed)
# -------------------------------------------------
SYSTEM_PROMPT = """
You are an expert in job role standardization.

Your task is to write a clean, unified, and standardized job description
using the provided job title, job function, core skillset (with proficiency levels implied),
and CWF topics with key tasks.

RULES:
1. Write in professional international English.
2. Length must NOT exceed the character limit.
3. Use a narrative continuous style (no bullets).
4. Use 1–2 paragraphs only.
5. Skill proficiency must be implied through natural wording
   (e.g., “applies advanced analysis”, “executes independently”).
6. Integrate CWF and key tasks into a cohesive role description.
7. Do NOT mention salary, benefits, company info, working hours,
   or unrelated administrative content.
8. The tone must be consistent and standardized.

Your output must be ONLY the job description text. No JSON, no commentary.
""".strip()


# -------------------------------------------------
# Helper: Format skillset + CWF into readable prompt blocks
# -------------------------------------------------
def _format_skillset_for_prompt(skillset: List[Dict[str, Any]]) -> str:
    """
    Convert skillset list → readable block for prompting.
    Example:
        - Excel (PL4)
        - SAP (PL3)
    """
    lines = []
    for s in skillset:
        name = s.get("skillname", "")
        pl = s.get("skill_pl", "")
        if name:
            lines.append(f"- {name} (PL{pl})")
    return "\n".join(lines)


def _format_cwf_for_prompt(cwf_data: List[Dict[str, Any]]) -> str:
    """
    Convert CWF list → readable block.
    Example:
        Financial Reporting:
            * prepare monthly closing
            * support audit queries
    """
    lines = []
    for cwf_block in cwf_data:
        cwf_name = cwf_block.get("cwf", "")
        keytasks = cwf_block.get("keytasks", [])
        if cwf_name:
            lines.append(f"{cwf_name}:")
            for kt in keytasks:
                lines.append(f"  - {kt}")
    return "\n".join(lines)


# -------------------------------------------------
# Main function
# -------------------------------------------------
def generate_standard_job_description(
    job_title_normed: str,
    job_function: str,
    skillset: List[Dict[str, Any]],
    cwf_data: List[Dict[str, Any]],
    max_chars: int = 2000,
    model_name: str = "gpt-4o-mini",
) -> Dict[str, Any]:

    skill_block = _format_skillset_for_prompt(skillset)
    cwf_block = _format_cwf_for_prompt(cwf_data)

    # ----------------------------
    # USER PROMPT
    # ----------------------------
    user_prompt = f"""
Job Title: {job_title_normed}
Job Function: {job_function}

Character Limit: {max_chars}

Core Skillset (with proficiency levels):
{skill_block}

CWF Topics + Key Tasks:
{cwf_block}

Write the standardized job description now.
""".strip()

    # ----------------------------
    # LLM Call
    # ----------------------------
    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    jd_text = resp.choices[0].message.content.strip()
    jd_text = jd_text[:max_chars]

    # ----------------------------
    # JSON Output
    # ----------------------------
    return {
        "job_title_normed": job_title_normed,
        "job_function": job_function,
        "max_characters": max_chars,
        "standard_job_description": jd_text,
        "length": len(jd_text),
        "source": {
            "skillset_count": len(skillset),
            "cwf_count": len(cwf_data)
        }
    }
