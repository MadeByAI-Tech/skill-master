import os, time, json, re
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============ OpenAI Client =========
os.environ.setdefault("OPENAI_API_KEY", "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")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

PL_VERSION = "pl_v1"

# =====================================================================
# LLM–1: PL Descriptor Builder (Step 3)
# =====================================================================

SYSTEM_LLM1 = """
You build a 6-level Proficiency Ladder (PL1–PL6) for ONE skill.

Your input:
- skill_title_norm (canonical skill name)
- skill_type (TSC / GSC / Cert.)
- many evidence sentences gathered across jobposts

Rules:
1. The PL ladder must cover the full range from beginner to expert.
2. PL1 = basic exposure, limited independence.
   PL2 = simple tasks under supervision.
   PL3 = standard tasks independently.
   PL4 = moderate complexity, adapts and optimizes.
   PL5 = leads complex/strategic tasks and mentors others.
   PL6 = expert-level mastery, sets strategy/standards for org.

3. Ladder must be generalizable (NOT tied to any company).
4. Use evidence to scale difficulty, independence, complexity.
5. If skill_type == "Cert." → pl_descriptors = "N/A".

OUTPUT FORMAT (PURE JSON ONLY):
{
  "skill_title_norm": "...",
  "skill_type": "...",
  "pl_descriptors": {
      "PL1": "...",
      "PL2": "...",
      ...
      "PL6": "..."
  } OR "N/A",
  "evidence_span": <int>,
  "version": "<string>"
}
"""

def build_pl_descriptors_for_cluster(
    skill_title_norm: str,
    skill_type: str,
    evidence_sentences: List[str],
    model_name: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Input:
      - canonical skill name
      - type (TSC/GSC/Cert.)
      - evidence sentences (list)
    Output:
      dict as PL descriptor pack
    """

    # ---------------------
    # Cert. = N/A ปิดจ๊อบ
    # ---------------------
    if skill_type == "Cert.":
        return {
            "skill_title_norm": skill_title_norm,
            "skill_type": skill_type,
            "pl_descriptors": "N/A",
            "evidence_span": len(evidence_sentences),
            "version": PL_VERSION
        }

    # ---------------------
    # Limit evidence to avoid token explosion
    # ---------------------
    MAX_EVID = 60
    evid_sample = evidence_sentences[:MAX_EVID]

    evidence_text = "\n".join(f"- {e}" for e in evid_sample)

    user_prompt = f"""
Skill: {skill_title_norm}
Type: {skill_type}

Evidence sentences (multi-jobposts):
{evidence_text}

Return JSON exactly as specified.
""".strip()

    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_LLM1},
            {"role": "user", "content": user_prompt}
        ]
    )

    raw = resp.choices[0].message.content.strip()

    # Robust JSON extraction
    start = raw.find("{")
    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{": depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                json_text = raw[start:i + 1]
                data = json.loads(json_text)
                return data

    raise ValueError("Failed to parse LLM JSON output in PL builder.")
