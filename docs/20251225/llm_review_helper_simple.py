# llm_review_helper_simple.py
import os, time, json, re
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ============== Config ==============
MODEL        = "gpt-4o-mini"
TEMP         = 0.0
MAX_RETRIES  = 3
RETRY_SLEEP  = 1.0  # sec

IN_FILE      = "job_title_last_check.csv"      # คอลัมน์: rare_item, best_match, similarity
OUT_FILE     = "llm_decisions_simple.csv"
ERR_FILE     = "llm_errors_simple.csv"

# ============ OpenAI Client =========
os.environ.setdefault("OPENAI_API_KEY", "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")
from openai import OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ============ Small utils ===========
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8")

def write_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

def safe_float(x, default=0.0) -> float:
    try:
        if x is None: return default
        if isinstance(x, str) and not x.strip():
            return default
        v = float(x)
        return v if v == v else default  # NaN guard
    except Exception:
        return default

def _extract_json(text: str) -> str:
    """ดึงก้อน JSON แรกด้วยการนับวงเล็บปีกกา (กันโมเดลพูดเกิน)"""
    if not text:
        return ""
    start = text.find("{")
    if start == -1:
        return text.strip()
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return text[start:].strip()

# ========= Prompts (ยึด 3 อินพุต) ========
SYSTEM_PROMPT = (
    "You normalize job titles in a labor-market dataset. "
    "You will receive:\n"
    "- rare_item: a rare/raw job title string\n"
    "- best_match: the best canonical candidate title string\n"
    "- similarity: numeric score in [0,1]\n\n"
    "Task: Decide whether the two titles refer to the SAME role (MERGE) or a DIFFERENT role (NEW_CANONICAL).\n"
    "Guidelines:\n"
    "• MERGE when differences are only word order, connectors, spelling variants, or synonyms that don't change the role.\n"
    "• NEW_CANONICAL when seniority or responsibility scope differs (e.g., Manager vs Officer), or different job families (e.g., Auditor vs Accountant).\n"
    "• Consider the similarity score as one signal, not an absolute rule.\n"
    "• IMPORTANT: Only two outcomes are allowed: MERGE or NEW_CANONICAL. No manual review.\n"
    "Return ONLY JSON matching the schema. No extra text."
)

# few-shot
FEWSHOTS = [
    {
        "rare_item": "Accounting And Finance Officer",
        "best_match": "Finance & Accounting Officer",
        "similarity": 0.78,
        "expect": {
            "decision": "MERGE",
            "confidence": 0.90,
            "reason": "Same role; wording/connector re-order only"
        }
    },
    {
        "rare_item": "Accounting Manager",
        "best_match": "Accounting Officer",
        "similarity": 0.82,
        "expect": {
            "decision": "NEW_CANONICAL",
            "confidence": 0.90,
            "reason": "Different seniority (Manager vs Officer)"
        }
    },
    {
        "rare_item": "Internal Auditor",
        "best_match": "Accountant",
        "similarity": 0.73,
        "expect": {
            "decision": "NEW_CANONICAL",
            "confidence": 0.85,
            "reason": "Different job families (Auditor vs Accountant)"
        }
    },
    {
        "rare_item": "Finance Staff",
        "best_match": "Finance Officer",
        "similarity": 0.71,
        "expect": {
            "decision": "NEW_CANONICAL",
            "confidence": 0.60,
            "reason": "Likely different scope/seniority; ambiguous titles → treat as different"
        }
    },
]

USER_TEMPLATE = """rare_item: "{rare_item}"
best_match: "{best_match}"
similarity: {similarity}

Return JSON exactly in this schema:
{{
  "decision": "MERGE" | "NEW_CANONICAL",
  "confidence": 0-1,
  "reason": "<short>"
}}"""

def build_messages(rare_item: str, best_match: str, similarity: float) -> List[Dict[str, Any]]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEWSHOTS:
        msgs.append({"role": "user", "content": USER_TEMPLATE.format(**ex)})
        msgs.append({"role": "assistant", "content": json.dumps(ex["expect"], ensure_ascii=False)})
    payload = {
        "rare_item": rare_item or "",
        "best_match": best_match or "",
        "similarity": similarity
    }
    msgs.append({"role": "user", "content": USER_TEMPLATE.format(**payload)})
    msgs.append({
        "role": "system",
        "content": "Return ONLY valid minified JSON. No backticks, no extra text."
    })
    return msgs

def decide_pair(rare_item: str, best_match: str, similarity: float) -> Dict[str, Any]:
    """เรียก LLM เพื่อให้ผล MERGE / NEW_CANONICAL + confidence + reason"""
    messages = build_messages(rare_item, best_match, similarity)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                temperature=TEMP,
                messages=messages,
            )
            content = resp.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Empty content from model")

            payload = _extract_json(content)
            if not payload.startswith("{"):
                raise ValueError(f"Non-JSON content: {content[:200]}")

            data = json.loads(payload)
            decision = str(data.get("decision", "")).upper()
            if decision not in {"MERGE", "NEW_CANONICAL"}:
                raise ValueError(f"Invalid decision: {decision}")

            return {
                "decision": decision,
                "confidence": safe_float(data.get("confidence", 0.0), 0.0),
                "reason": str(data.get("reason", "")),
            }
        except Exception as e:
            if attempt >= MAX_RETRIES:
                # ตัด manual review: ถ้า error ให้ default เป็น NEW_CANONICAL (ระมัดระวัง)
                return {
                    "decision": "NEW_CANONICAL",
                    "confidence": 0.0,
                    "reason": f"LLM_error: {type(e).__name__}: {e}"
                }
            time.sleep(RETRY_SLEEP)

# ============== Runner ==============
def run_llm_review_simple(
    rare_item: str,
    best_match: str,
    similarity: float
) -> dict:
    """
    Run LLM decision logic for a single (rare_item, best_match, similarity) pair.

    Returns:
        {
            "rare_item": str,
            "best_match": str,
            "similarity": float,
            "decision": str,
            "confidence": float,
            "reason": str
        }
    """

    # Normalize inputs
    rare = str(rare_item) if rare_item is not None else ""
    best = str(best_match) if best_match is not None else ""
    sim = safe_float(similarity, 0.0)

    # LLM decision
    res = decide_pair(rare, best, sim)

    return {
        "rare_item": rare,
        "best_match": best,
        "similarity": sim,
        "decision": res.get("decision"),
        "confidence": res.get("confidence"),
        "reason": res.get("reason"),
    }