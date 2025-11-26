# agent.py  -- SDR agent (Day 5) - Razorpay 
import logging
import os
import json
import datetime
import uuid
from dataclasses import dataclass
from typing import Dict, Any

from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, deepgram, noise_cancellation, google, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -----------------------
# Config & logging
# -----------------------
logger = logging.getLogger("sdr_agent")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
load_dotenv(".env.local")

BASE_DIR = os.path.dirname(__file__)
CONTENT_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "shared-data", "day5_sdr_content.json"))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "shared-data", "leads"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

logger.info("📄 Using content file: %s", CONTENT_FILE)
logger.info("📁 Lead output dir: %s", OUTPUT_DIR)

# -----------------------
# Content loader (NO AUTO CREATE)
# -----------------------
def read_content() -> Dict[str, Any]:
    if not os.path.exists(CONTENT_FILE):
        raise FileNotFoundError(f"Missing required content file: {CONTENT_FILE}")
    with open(CONTENT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------
# Simple FAQ lookup (token overlap + fuzzy)
# -----------------------
import re
from difflib import SequenceMatcher

def faq_lookup(user_question: str, faq_entries: list):
    uq = (user_question or "").lower()
    uq_tokens = set(re.findall(r"\w+", uq))
    best = None
    best_score = 0.0
    for entry in faq_entries:
        text = (entry.get("q","") + " " + entry.get("a","")).lower()
        etokens = set(re.findall(r"\w+", text))
        token_overlap = len(uq_tokens & etokens)
        overlap_score = token_overlap / max(1, len(uq_tokens)) if uq_tokens else 0.0
        fuzz = SequenceMatcher(None, uq, entry.get("q","").lower()).ratio() if uq else 0.0
        score = 0.6 * overlap_score + 0.4 * fuzz
        if score > best_score:
            best_score = score
            best = entry
    if best_score >= 0.25 and best is not None:
        return best.get("a"), best.get("q"), best_score
    return None, None, 0.0

# -----------------------
# State classes
# -----------------------
@dataclass
class SDRState:
    lead: Dict[str, Any]
    transcript: list
    faq: list
    company: dict

    @classmethod
    def create_from_content(cls, content: dict):
        lead = {
            "lead_id": str(uuid.uuid4()),
            "name": "",
            "company": "",
            "email": "",
            "role": "",
            "use_case": "",
            "team_size": "",
            "timeline": "",
            "notes": "",
            "timestamp": ""
        }
        return cls(lead=lead, transcript=[], faq=content.get("faq", []), company=content.get("company", {}))

# -----------------------
# Function tools (exposed to LLM)
# -----------------------
@function_tool
async def get_faq_answer(ctx: RunContext[SDRState], question: str = Field(description="User question")) -> Dict:
    """
    Return the best FAQ answer (or signal unknown).
    """
    state: SDRState = ctx.userdata
    ans, matched_q, score = faq_lookup(question, state.faq)
    if ans:
        state.transcript.append({
            "from":"agent",
            "type":"faq_answer",
            "question":question,
            "matched_q":matched_q,
            "answer":ans,
            "score":score,
            "time":datetime.datetime.utcnow().isoformat()+"Z"
        })
        return {"answer": ans, "matched_question": matched_q, "score": score}
    else:
        state.transcript.append({
            "from":"agent",
            "type":"faq_miss",
            "question":question,
            "time":datetime.datetime.utcnow().isoformat()+"Z"
        })
        return {"answer": None, "note": "I don't have that detail in the FAQ. Offer to connect to sales."}

@function_tool
async def capture_lead_field(ctx: RunContext[SDRState], field: str = Field(description="Lead field name"), value: str = Field(description="Value provided by user")) -> str:
    """
    Capture a single lead field (name/company/email/role/use_case/team_size/timeline)
    """
    state: SDRState = ctx.userdata
    if field not in state.lead:
        return f"Field '{field}' not supported."
    state.lead[field] = value.strip()
    ts = datetime.datetime.utcnow().isoformat() + "Z"
    state.lead["notes"] += f"[{ts}] Captured {field}: {value}\n"
    state.transcript.append({"from":"agent","type":"capture","field":field,"value":value,"time":ts})
    return f"Saved {field}."

@function_tool
async def summarize_and_save(ctx: RunContext[SDRState]) -> Dict:
    """
    Save lead JSON and transcript and return a short summary for the agent to speak.
    """
    state: SDRState = ctx.userdata
    state.lead["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    lead_fname = os.path.join(OUTPUT_DIR, f"lead_{state.lead['lead_id']}.json")
    transcript_fname = os.path.join(OUTPUT_DIR, f"transcript_{state.lead['lead_id']}.json")
    with open(lead_fname, "w", encoding="utf-8") as f:
        json.dump(state.lead, f, indent=2, ensure_ascii=False)
    with open(transcript_fname, "w", encoding="utf-8") as f:
        json.dump(state.transcript, f, indent=2, ensure_ascii=False)
    summary = (
        f"Summary: {state.lead.get('name') or 'Unknown'} from {state.lead.get('company') or 'Unknown'} "
        f"({state.lead.get('role') or 'Unknown'}). Email: {state.lead.get('email') or 'Unknown'}. "
        f"Use case: {state.lead.get('use_case') or 'Not specified'}. Timeline: {state.lead.get('timeline') or 'Not set'}."
    )
    state.transcript.append({"from":"agent","type":"summary","text":summary,"time":datetime.datetime.utcnow().isoformat()+"Z"})
    return {"summary": summary, "lead_file": lead_fname, "transcript_file": transcript_fname}

# -----------------------
# LiveKit Agent
# -----------------------
class SDRAgent(Agent):
    def __init__(self, company_name: str, faq_count: int):
        super().__init__(
            instructions=f"""
            You are Alex — an SDR assistant for {company_name}.
            Use the provided tools to fetch FAQ answers and capture lead fields.
            Keep conversations friendly, ask for name/company/email/role/use_case/team_size/timeline,
            confirm contact info before ending, and call summarize_and_save() when the user ends the call.
            """,
            tools=[get_faq_answer, capture_lead_field, summarize_and_save],
        )
        self._brief = f"SDR for {company_name} — {faq_count} FAQ items loaded."

# -----------------------
# Prewarm & entrypoint
# -----------------------
def prewarm(proc: JobProcess):
    # prewarm VAD model (silero) so VAD is ready quickly when session starts
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("Prewarmed silero VAD.")
    except Exception:
        proc.userdata["vad"] = None
        logger.warning("Failed to prewarm silero VAD (continuing without).")

async def entrypoint(ctx: JobContext):
    logger.info("Starting SDR agent entrypoint.")
    content = read_content()  # raise if missing
    sdr_state = SDRState.create_from_content(content)

    # create session (using Deepgram STT, Google LLM, Murf TTS, turn detector, and prewarmed VAD)
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Promo", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=sdr_state,
        preemptive_generation=True,
    )

    # Start the session; the Agent will be the SDR agent using the provided tools
    await session.start(
        agent=SDRAgent(company_name=sdr_state.company.get("name","<company>"), faq_count=len(sdr_state.faq)),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
