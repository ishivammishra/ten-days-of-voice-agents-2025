import logging
import os
import json
import datetime
import tempfile
import threading
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# File persistence settings
DATA_DIR = os.path.join(os.getcwd(), "backend")
DATA_FILE = os.path.join(DATA_DIR, "wellness_log.json")
_file_lock = threading.Lock()

def ensure_data_file():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)

def read_all_entries():
    ensure_data_file()
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return []

def atomic_write_all(entries):
    ensure_data_file()
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="tmp_", dir=DATA_DIR)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        os.replace(tmp, DATA_FILE)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly, grounded Health & Wellness voice companion. "
                "Your role is supportive: run a short daily check-in asking about the user's mood, energy, "
                "stress, and 1–3 practical intentions for the day. Do NOT give medical diagnoses or treatment. "
                "Keep suggestions small and actionable (e.g., 'take a 5-minute walk', 'try a 2-minute stretch', "
                "'break the task into a 10-minute starter'). Ask one question at a time. "
                "At the end, summarize today's mood and objectives and ask 'Does this sound right?'. "
                "After confirmation, call the function save_checkin with a payload that includes at least: "
                "timestamp, mood (text), mood_score (optional numeric 1-10), energy (low/medium/high), "
                "objectives (list), and agent_summary (one-line). "
                "On session start, call get_last_checkin to retrieve the previous session and include one brief "
                "reference to it (for example: 'Last time you said you were low energy. How is today different?'). "
                "Keep each turn short; aim for 6-12 turns. Be empathetic, concise, and non-judgmental."
            )
        )
        # Temporary in-memory state during a session (not required but helpful)
        self.current_session_state = {
            "mood": "",
            "mood_score": None,
            "energy": "",
            "objectives": [],
            "agent_summary": "",
        }

    @function_tool
    async def save_checkin(self, context: RunContext, payload: dict):
        """
        Persist a check-in to backend/wellness_log.json.
        Expects payload keys: mood (str), mood_score (int|None), energy (str), objectives (list[str]), agent_summary (str), meta (optional dict)
        Returns a confirmation string.
        """
        # Basic validation and normalization
        mood = payload.get("mood", "").strip()
        mood_score = payload.get("mood_score", None)
        energy = payload.get("energy", "").strip()
        objectives = payload.get("objectives", []) or []
        if isinstance(objectives, str):
            # Accept comma-separated string too
            objectives = [o.strip() for o in objectives.split(",") if o.strip()]
        agent_summary = payload.get("agent_summary", "").strip()
        meta = payload.get("meta", {})

        entry = {
            "id": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(),
            "mood": mood,
            "mood_score": mood_score,
            "energy": energy,
            "objectives": objectives,
            "agent_summary": agent_summary,
            "meta": meta,
        }

        # Append safely
        with _file_lock:
            entries = read_all_entries()
            entries.append(entry)
            try:
                atomic_write_all(entries)
            except Exception as e:
                logger.exception("Failed to save checkin")
                return f"Failed to save check-in: {e}"

        return f"Check-in saved. ({len(entry['objectives'])} objectives recorded)"

    @function_tool
    async def get_last_checkin(self, context: RunContext):
        """
        Return the most recent check-in entry or None if none exists.
        """
        with _file_lock:
            entries = read_all_entries()
        if not entries:
            return None
        return entries[-1]

def prewarm(proc: JobProcess):
    # prewarm VAD model as before
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with the Assistant
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to the room and user
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
