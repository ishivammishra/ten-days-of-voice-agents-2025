# agent.py  -- Fraud Alert Voice Agent (Day 6)
import logging
import os
import sqlite3
import threading
import datetime
import uuid
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional

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
from livekit.plugins import murf, deepgram, noise_cancellation, silero, google
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -----------------------
# Config & logging
# -----------------------
logger = logging.getLogger("fraud_agent")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
load_dotenv(".env.local")

BASE_DIR = os.path.dirname(__file__)
DB_FILE = os.path.join(BASE_DIR, "fraud_cases.db")
logger.info("Using SQLite DB at %s", DB_FILE)

# -----------------------
# DB helpers (thread-safe)
# -----------------------
_db_lock = threading.Lock()

def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create DB and seed with fake entries if missing (DEMO ONLY)."""
    if os.path.exists(DB_FILE):
        logger.info("DB exists, skipping seed.")
        return
    with _db_lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE fraud_cases (
            id TEXT PRIMARY KEY,
            userName TEXT,
            securityIdentifier TEXT,
            cardEnding TEXT,
            transactionAmount TEXT,
            merchantName TEXT,
            location TEXT,
            timestamp TEXT,
            transactionCategory TEXT,
            transactionSource TEXT,
            securityQuestion TEXT,
            securityAnswer TEXT,
            status TEXT,
            notes TEXT
        )
        """)
        seed = [
            (
                "case-001", "Shivam", "SID-12345", "**** 4242", "₹3,250.00",
                "ABC Industry", "Bengaluru, IN", "2025-11-20T14:32:00+05:30",
                "e-commerce", "alibaba.com", "What is your favorite color?", "blue",
                "pending_review", "[]"
            ),
            (
                "case-002", "John", "SID-54321", "**** 1234", "₹9,999.00",
                "Zeta Electronics", "Delhi, IN", "2025-11-24T09:12:00+05:30",
                "electronics", "zetastore.fake", "What city were you born in?", "delhi",
                "pending_review", "[]"
            ),
            (
                "case-003", "Sneha", "SID-67890", "**** 7788", "₹5,499.00",
                "Fashion Hub", "Mumbai, IN", "2025-11-26T19:45:00+05:30",
                "fashion", "fashionhub.fake",
                "What is your pet’s name?", "duster",
                "pending_review", "[]"
            )
        ]
        cur.executemany("""
        INSERT INTO fraud_cases (
          id, userName, securityIdentifier, cardEnding, transactionAmount,
          merchantName, location, timestamp, transactionCategory, transactionSource,
          securityQuestion, securityAnswer, status, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, seed)
        conn.commit()
        conn.close()
        logger.info("Seeded demo DB with fake fraud cases.")

def find_pending_case_by_username(username: str) -> Optional[sqlite3.Row]:
    with _db_lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("""
          SELECT * FROM fraud_cases
          WHERE LOWER(userName) = LOWER(?) AND status = 'pending_review'
          ORDER BY timestamp ASC
          LIMIT 1
        """, (username,))
        row = cur.fetchone()
        conn.close()
        return row

def get_case(case_id: str) -> Optional[sqlite3.Row]:
    with _db_lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM fraud_cases WHERE id = ?", (case_id,))
        row = cur.fetchone()
        conn.close()
        return row

def update_case_status_and_append_note(case_id: str, new_status: str, note: str):
    with _db_lock:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT notes FROM fraud_cases WHERE id = ?", (case_id,))
        r = cur.fetchone()
        notes = r["notes"] if r and r["notes"] else "[]"
        # notes is stored as a JSON array string for demo; append safely
        try:
            nlist = json.loads(notes)
        except Exception:
            nlist = []
        nlist.append({"time": datetime.datetime.utcnow().isoformat() + "Z", "note": note})
        cur.execute("UPDATE fraud_cases SET status = ?, notes = ? WHERE id = ?", (new_status, json.dumps(nlist), case_id))
        conn.commit()
        conn.close()

# initialize DB
init_db()

# -----------------------
# Agent state dataclass
# -----------------------
@dataclass
class FraudState:
    caseId: Optional[str]
    caseSummary: Optional[Dict[str, Any]]  # non-sensitive details for TTS
    transcript: list

    @classmethod
    def empty(cls):
        return cls(caseId=None, caseSummary=None, transcript=[])

# -----------------------
# Helper text builders (safe: never include securityAnswer)
# -----------------------
def make_intro_text(user_name: str) -> str:
    return (f"Hello {user_name}. This is the Fraud Prevention team from State Bank of India. "
            "We are contacting you about a suspicious transaction on your account. "
            "For your security I will ask a verification question before we proceed.")

def make_transaction_text(row: sqlite3.Row) -> str:
    ts = row["timestamp"]
    # present the row details in a TTS friendly way (no sensitive data)
    return (f"We detected a transaction at {row['merchantName']} for {row['transactionAmount']} "
            f"on {ts}. The card used ends with {row['cardEnding']}. "
            "Was this transaction made by you? Please answer yes or no.")

# -----------------------
# Exposed function tools (callable by LLM)
# -----------------------
@function_tool
async def load_case_for_user(ctx: RunContext[FraudState], user_name: str = Field(description="Customer name to lookup")) -> Dict:
    """
    Find a pending fraud case for the user. Returns non-sensitive details (caseId, introText, securityQuestion, masked card).
    """
    state: FraudState = ctx.userdata
    row = find_pending_case_by_username(user_name.strip())
    if not row:
        text = (f"Hello {user_name.strip()}. We do not have any pending suspicious transactions at this time. "
                "If you believe this is an error, please contact the bank's support line.")
        state.transcript.append({"from": "agent", "type": "no_case_found", "user": user_name, "time": datetime.datetime.utcnow().isoformat() + "Z"})
        return {"found": False, "text": text}

    # populate state with safe summary (do not store securityAnswer)
    summary = {
        "caseId": row["id"],
        "userName": row["userName"],
        "merchantName": row["merchantName"],
        "transactionAmount": row["transactionAmount"],
        "maskedCard": row["cardEnding"],
        "timestamp": row["timestamp"],
        "transactionCategory": row["transactionCategory"],
        "transactionSource": row["transactionSource"]
    }
    state.caseId = row["id"]
    state.caseSummary = summary
    state.transcript.append({"from": "agent", "type": "case_loaded", "caseId": row["id"], "time": datetime.datetime.utcnow().isoformat() + "Z"})
    return {
        "found": True,
        "caseId": row["id"],
        "introText": make_intro_text(row["userName"]),
        "securityQuestion": row["securityQuestion"],
        "maskedCard": row["cardEnding"]
    }

@function_tool
async def verify_security_answer(ctx: RunContext[FraudState], case_id: str = Field(description="Case ID"), answer: str = Field(description="Answer provided by customer")) -> Dict:
    """
    Verify security answer server-side. If correct, return transactionText for reading; otherwise mark verification_failed.
    """
    state: FraudState = ctx.userdata
    r = get_case(case_id)
    if not r:
        state.transcript.append({"from": "agent", "type": "verify_error", "caseId": case_id, "note": "case not found", "time": datetime.datetime.utcnow().isoformat() + "Z"})
        return {"verified": False, "text": "Case not found."}

    correct = (r["securityAnswer"] or "").strip().lower()
    given = (answer or "").strip().lower()
    if correct and given == correct:
        # success
        tx_text = make_transaction_text(r)
        state.transcript.append({"from": "agent", "type": "verified", "caseId": case_id, "time": datetime.datetime.utcnow().isoformat() + "Z"})
        return {"verified": True, "transactionText": tx_text}
    else:
        # failure -> update DB
        update_case_status_and_append_note(case_id, "verification_failed", "Verification failed via agent session")
        state.transcript.append({"from": "agent", "type": "verification_failed", "caseId": case_id, "time": datetime.datetime.utcnow().isoformat() + "Z"})
        return {"verified": False, "text": ("I'm sorry, I could not verify your identity. For your security, I cannot proceed. "
                                             "Please contact the bank's support line to continue.")}

@function_tool
async def resolve_case(ctx: RunContext[FraudState], case_id: str = Field(description="Case ID"), confirmed: bool = Field(description="true if customer says transaction was made by them")) -> Dict:
    """
    Resolve case: confirmed True -> confirmed_safe; False -> confirmed_fraud with mock actions.
    """
    state: FraudState = ctx.userdata
    r = get_case(case_id)
    if not r:
        state.transcript.append({"from": "agent", "type": "resolve_error", "caseId": case_id, "note": "case not found", "time": datetime.datetime.utcnow().isoformat() + "Z"})
        return {"ok": False, "text": "Case not found."}

    if r["status"] == "verification_failed":
        state.transcript.append({"from": "agent", "type": "resolve_blocked", "caseId": case_id, "time": datetime.datetime.utcnow().isoformat() + "Z"})
        return {"ok": False, "text": "This case is locked due to failed verification."}

    if confirmed:
        update_case_status_and_append_note(case_id, "confirmed_safe", "Customer confirmed transaction as legitimate.")
        state.transcript.append({"from": "agent", "type": "resolved_safe", "caseId": case_id, "time": datetime.datetime.utcnow().isoformat() + "Z"})
        return {"ok": True, "text": "Thank you. We have marked the transaction as legitimate and no further action is required. Have a nice day."}
    else:
        update_case_status_and_append_note(case_id, "confirmed_fraud", "Customer denied transaction; flagged as fraudulent. Mock actions: card blocked, dispute opened.")
        state.transcript.append({"from": "agent", "type": "resolved_fraud", "caseId": case_id, "time": datetime.datetime.utcnow().isoformat() + "Z"})
        return {"ok": True, "text": f"Thank you for confirming. We have blocked the card ending {r['cardEnding']} and opened a dispute. Our fraud team will contact you for next steps."}

# -----------------------
# Agent class for LLM-driven flow
# -----------------------
class FraudAgentLLM(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are a calm, professional fraud prevention representative for State Bank of India.
            Flow:
             1) Ask for customer's name to locate a pending fraud case.
             2) Use load_case_for_user(user_name) to fetch case. Do NOT ask for or reveal secrets.
             3) Ask the security question returned by load_case_for_user. Collect the spoken answer.
             4) Call verify_security_answer(case_id, answer). If verified, read transactionText from response.
             5) Ask the user 'Did you make this transaction? Please answer yes or no.'
             6) Call resolve_case(case_id, confirmed) using user's yes/no.
             7) End the call by repeating the outcome and next (mock) actions.
            Always avoid asking for full card numbers, PINs, passwords, OTPs, or any sensitive credentials.
            """,
            tools=[load_case_for_user, verify_security_answer, resolve_case],
        )

# -----------------------
# Prewarm & entrypoint
# -----------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
        logger.info("Prewarmed silero VAD.")
    except Exception as e:
        proc.userdata["vad"] = None
        logger.warning("Failed to prewarm silero VAD: %s", e)

async def entrypoint(ctx: JobContext):
    logger.info("Starting Fraud agent entrypoint.")
    # create initial state for each session
    fraud_state = FraudState.empty()

    # create session like Day 5: Deepgram STT, Google LLM, Murf TTS, turn detector, VAD
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Calm", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=fraud_state,
        preemptive_generation=True,
    )

    await session.start(
        agent=FraudAgentLLM(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

# -----------------------
# CLI entry for worker
# -----------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
