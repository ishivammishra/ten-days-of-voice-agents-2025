# agent.py  -- Teach-the-Tutor (Day 4) 
import logging
import os
import json
import datetime
from dataclasses import dataclass
from typing import Optional, Annotated, Literal

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
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -----------------------
# Configuration & logging
# -----------------------
logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)
load_dotenv(".env.local")

BASE_DIR = os.path.dirname(__file__)
# file MUST exist at backend/shared-data/day4_tutor_content.json (relative to agent.py location)
CONTENT_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "shared-data", "day4_tutor_content.json"))

VOICE_BY_MODE = {
    "learn": "Matthew",
    "quiz": "Alicia",
    "teach_back": "Ken",
}

print("📄 Using content file:", CONTENT_FILE)

# -----------------------
# Content loader (NO DEFAULTS)
# -----------------------
def read_content():
    """
    Load the user-provided content file.
    DO NOT auto-create or overwrite - raise if missing.
    """
    if not os.path.exists(CONTENT_FILE):
        raise FileNotFoundError(
            f"Missing required content file: {CONTENT_FILE}\n"
            "Create this file with your course content (an array of concept objects)."
        )
    with open(CONTENT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------
# Simple heuristics (scoring only in-memory)
# -----------------------
def simple_keyword_check(answer: str, summary: str) -> bool:
    if not summary or not answer:
        return False
    strip_chars = ".,()[]!?\"'`"
    words = [w.strip(strip_chars).lower() for w in summary.split() if len(w) > 4]
    keywords = words[:3]
    hits = sum(1 for k in keywords if k in answer.lower())
    return hits >= 1

def qualitative_score(answer: str, summary: str) -> int:
    if not summary or not answer:
        return 0
    strip_chars = ".,?():;!\"'`"
    ans_words = set(w.strip(strip_chars).lower() for w in answer.split() if w.strip())
    summ_words = set(w.strip(strip_chars).lower() for w in summary.split() if w.strip())
    if not summ_words:
        return 0
    overlap = len(ans_words & summ_words)
    score = int(min(100, (overlap / max(1, len(summ_words))) * 100))
    return max(5, score)

def qualitative_feedback(score:int) -> str:
    if score >= 80:
        return "Excellent — you explained the concept clearly!"
    if score >= 50:
        return "Good — you covered the main ideas; try adding a short example next."
    return "Keep practicing — focus on the key idea and give an example."

# -----------------------
# State classes
# -----------------------
@dataclass
class TutorState:
    current_topic_id: Optional[str] = None
    current_topic_data: Optional[dict] = None
    mode: Literal["learn", "quiz", "teach_back"] = "learn"

    def set_topic(self, topic_id: str):
        topics = read_content()
        topic = next((item for item in topics if item.get("id") == topic_id), None)
        if topic:
            self.current_topic_id = topic_id
            self.current_topic_data = topic
            return True
        return False

@dataclass
class Userdata:
    tutor_state: TutorState
    agent_session: Optional[AgentSession] = None

# -----------------------
# Module-level function tools
# -----------------------
@function_tool
async def select_topic(
    ctx: RunContext[Userdata],
    topic_id: Annotated[str, Field(description="The ID of the topic to study (e.g., 'variables')")]
) -> str:
    state = ctx.userdata.tutor_state
    ok = state.set_topic(topic_id.lower())
    if ok:
        return f"Topic set to {state.current_topic_data.get('title','<no title>')}. Ask the user if they want to 'learn', 'quiz', or 'teach_back'."
    else:
        try:
            available = ", ".join([t.get("id","<no-id>") for t in read_content()])
        except FileNotFoundError:
            available = "<content file missing>"
        return f"Topic not found. Available topics are: {available}"

@function_tool
async def set_learning_mode(
    ctx: RunContext[Userdata],
    mode: Annotated[str, Field(description="The mode to switch to: 'learn', 'quiz', or 'teach_back'")]
) -> str:
    state = ctx.userdata.tutor_state
    state.mode = mode.lower()
    agent_session = ctx.userdata.agent_session

    topic = state.current_topic_data or {}
    title = topic.get("title", "<no topic selected>")
    summary = topic.get("summary", "No topic selected — please choose a topic.")
    question = topic.get("sample_question", "No question available.")

    if agent_session:
        if state.mode == "learn":
            try:
                agent_session.tts.update_options(voice="en-US-matthew", style="Promo")
            except Exception:
                agent_session.tts = murf.TTS(voice="en-US-matthew", style="Promo", text_pacing=True)
            instruction = f"Mode: LEARN. Topic: {title}. Explain: {summary}"
        elif state.mode == "quiz":
            try:
                agent_session.tts.update_options(voice="en-US-alicia", style="Conversational")
            except Exception:
                agent_session.tts = murf.TTS(voice="en-US-alicia", style="Conversational", text_pacing=True)
            instruction = f"Mode: QUIZ. Topic: {title}. Ask: {question}"
        elif state.mode == "teach_back":
            try:
                agent_session.tts.update_options(voice="en-US-ken", style="Promo")
            except Exception:
                agent_session.tts = murf.TTS(voice="en-US-ken", style="Promo", text_pacing=True)
            instruction = f"Mode: TEACH_BACK. Topic: {title}. Ask the user to explain: {question}"
        else:
            return "Invalid mode."
    else:
        instruction = "Voice switch failed (Session not found)."

    logger.info(f"Switched mode -> {state.mode}")
    return f"Switched to {state.mode} mode. {instruction}"

@function_tool
async def evaluate_teaching(
    ctx: RunContext[Userdata],
    user_explanation: Annotated[str, Field(description="The explanation given by the user during teach-back")]
):
    """
    Score the user's teach-back and return feedback.
    (NO persistence — no files are written.)
    """
    state = ctx.userdata.tutor_state
    topic = state.current_topic_data or {}
    concept_id = state.current_topic_id or topic.get("id", "unknown")
    summary = topic.get("summary", "")

    score = qualitative_score(user_explanation, summary)
    feedback = qualitative_feedback(score)

    return {"score": score, "feedback": feedback, "concept_id": concept_id}

# -----------------------
# Agent definition
# -----------------------
class TutorAgent(Agent):
    def __init__(self):
        # build topic list for prompt; read_content() will raise if missing
        topics = read_content()
        topic_list = ", ".join([f"{t.get('id','')} ({t.get('title','')})" for t in topics])

        super().__init__(
            instructions=f"""
            You are a Tutor designed to help users master programming concepts.

            📚 AVAILABLE TOPICS: {topic_list}

            You support three modes: learn (voice: Matthew), quiz (voice: Alicia), teach_back (voice: Ken).
            Use the provided tools to select topic, switch mode, and evaluate teach_back explanations.
            """,
            tools=[select_topic, set_learning_mode, evaluate_teaching],
        )

# -----------------------
# Prewarm & entrypoint
# -----------------------
def prewarm(proc: JobProcess):
    # only load VAD; do NOT create any data files
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # THIS WILL RAISE FileNotFoundError if CONTENT_FILE missing (intentional)
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"Starting Teach-the-Tutor session, reading content from {CONTENT_FILE}")

    # initialize userdata
    userdata = Userdata(tutor_state=TutorState())

    # create session
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Promo", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        userdata=userdata,
    )

    userdata.agent_session = session

    # start
    await session.start(
        agent=TutorAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
