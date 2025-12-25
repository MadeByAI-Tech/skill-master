# skills_llm2_pl_assigner.py

import os, time, json, re
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============ OpenAI Client =========
os.environ.setdefault("OPENAI_API_KEY", "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")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_LLM2 = """
You assign skill proficiency levels (PL) for ONE jobpost.

Input:
For each skill:
   • canonical skill title
   • skill_type (TSC/GSC/Cert.)
   • one evidence sentence from THIS jobpost
   • PL1–PL6 descriptors generated from multi-jobpost analysis (LLM-1)

Rules:
1. If skill_type == "Cert." → skill_pl = "N/A".
2. Otherwise:
   - Compare evidence sentence against PL descriptors.
   - Choose the BEST FIT among PL1..PL6.
3. rationale_short = 1 lines explaining
   why the PL level was chosen (link evidence → descriptor).

Output must be PURE JSON:
{
  "skills_assessed": [
     {
       "skill_title_norm": "<str>",
       "skill_type": "<str>",
       "skill_pl": <int or 'N/A'>,
       "evidence_used": "<str>",
       "rationale_short": "<str>"
     }
  ]
}
NO commentary. NO markdown.
"""


def assign_pl_for_jobpost(
    skills_for_post: List[Dict[str, Any]],
    model_name: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """
    Assign proficiency levels for all skills of ONE jobpost.

    skills_for_post format:
      [
        {
          "skill_title_norm": "...",
          "skill_type": "TSC"|"GSC"|"Cert.",
          "evidence_used": "...",
          "pl_descriptors": {
              "PL1": "...",
              "PL2": "...",
              ...
              "PL6": "..."
          } OR "N/A"
        },
        ...
      ]
    """

    user_payload = {
        "skills": skills_for_post
    }

    resp = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_LLM2},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ]
    )

    raw = resp.choices[0].message.content.strip()

    # --- Robust JSON extraction ---
    start = raw.find("{")
    if start == -1:
        raise ValueError(f"Invalid model response: {raw[:200]}")

    depth = 0
    for i in range(start, len(raw)):
        if raw[i] == "{":
            depth += 1
        elif raw[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])

    raise ValueError("Failed to extract JSON in LLM-2 assigner.")
