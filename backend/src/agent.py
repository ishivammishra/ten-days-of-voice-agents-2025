# src/agent.py -- Day 8: The Lost Lantern 
"""
Simple Day 8 voice Game Master: "The Lost Lantern"
Tool-backed (start_adventure, get_scene, player_action, show_journal, restart_adventure)
Structure matches the short friend-style Day-8 agent.
"""

import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Annotated

from dotenv import load_dotenv
from pydantic import Field

load_dotenv(".env.local")

# livekit imports (same pattern as prior short agent)
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

# logging
logger = logging.getLogger("lost_lantern")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# -------------------------
# Simple WORLD (tiny 4-scene model)
# -------------------------
WORLD = {
    "intro": {
        "title": "The Missing Lantern",
        "desc": (
            "Sunlight thins and the guardian fairy flutters before you in a tiny forest clearing. "
            "“Our lantern is gone,” she whispers. Without it, night will swallow the woods. "
            "A trail of tiny glowing pawprints leads toward the fox den."
        ),
        "choices": {
            "inspect_clearing": {"desc": "Look around the clearing for clues.", "result_scene": "fox_den"},
            "ask_fairy": {"desc": "Ask the fairy what happened.", "result_scene": "fairy_info"},
            "follow_pawprints": {"desc": "Follow the glowing pawprints.", "result_scene": "fox_den"},
        },
    },

    "fairy_info": {
        "title": "The Fairy's Fear",
        "desc": (
            "The fairy trembles. She says the lantern wandered off at dusk and a fox was seen near the old bridge. "
            "She points toward a mossy path leading that way."
        ),
        "choices": {
            "follow_path": {"desc": "Follow the mossy path toward the bridge.", "result_scene": "bridge"},
            "check_den": {"desc": "Check the fox den nearby.", "result_scene": "fox_den"},
        },
    },

    "fox_den": {
        "title": "Fox Den",
        "desc": (
            "A small den under a root. Tiny pawprints glow faintly. You see the prints continue toward an old wooden bridge."
        ),
        "choices": {
            "follow_prints": {"desc": "Follow the pawprints to the bridge.", "result_scene": "bridge"},
            "search_den": {"desc": "Search the den for signs of the lantern.", "result_scene": "bridge"},
        },
    },

    "bridge": {
        "title": "Old Bridge",
        "desc": (
            "Under the bridge you hear a jingle like glass on stone. A fox curls around something that glows softly."
        ),
        "choices": {
            "approach": {"desc": "Approach the fox carefully.", "result_scene": "under_bridge"},
            "call_out": {"desc": "Call out to the fox in a friendly voice.", "result_scene": "under_bridge"},
        },
    },

    "under_bridge": {
        "title": "Lantern Found",
        "desc": (
            "The fox guards a small golden lantern, its light warm and gentle. It watches you with curious eyes."
        ),
        "choices": {
            "take_lantern": {
                "desc": "Gently take the lantern from the fox.",
                "result_scene": "return_fairy",
                "effects": {"add_journal": "Recovered the lost lantern."},
            },
            "befriend_fox": {
                "desc": "Offer food and try to befriend the fox.",
                "result_scene": "return_fairy",
                "effects": {"add_journal": "Befriended fox; retrieved lantern peacefully."},
            },
        },
    },

    "return_fairy": {
        "title": "Light Restored",
        "desc": (
            "You bring the lantern back to the clearing. The fairy sets it upon the pedestal, and golden light spills through the trees."
        ),
        "choices": {
            "end_session": {"desc": "Finish the little quest and bask in the glow.", "result_scene": "reward"},
        },
    },

    "reward": {
        "title": "Forest Saved",
        "desc": (
            "The woods breathe again. The fairy thanks you with a small charm. The lantern hums gently — safe for now."
        ),
        "choices": {
            "end_session": {"desc": "End the session.", "result_scene": "reward"},
        },
    },
}

# -------------------------
# Per-session Userdata
# -------------------------
@dataclass
class Userdata:
    player_name: Optional[str] = None
    current_scene: str = "intro"
    history: List[Dict] = field(default_factory=list)
    journal: List[str] = field(default_factory=list)
    inventory: List[str] = field(default_factory=list)
    named_npcs: Dict[str, str] = field(default_factory=dict)
    choices_made: List[str] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

# -------------------------
# Helpers
# -------------------------
def scene_text(scene_key: str, userdata: Userdata) -> str:
    scene = WORLD.get(scene_key)
    if not scene:
        return "You stand in an empty clearing. What do you do?"
    desc = scene["desc"]
    
    if scene_key == "reward":
        return desc
    
    if not desc.endswith("What do you do?"):
        desc += "\n\nWhat do you do?"
    return desc

def apply_effects(effects: dict, userdata: Userdata):
    if not effects:
        return
    if "add_journal" in effects:
        userdata.journal.append(effects["add_journal"])
    if "add_inventory" in effects:
        userdata.inventory.append(effects["add_inventory"])

def summarize_scene_transition(old_scene: str, action_key: str, result_scene: str, userdata: Userdata):
    entry = {"from": old_scene, "action": action_key, "to": result_scene, "time": datetime.utcnow().isoformat() + "Z"}
    userdata.history.append(entry)
    userdata.choices_made.append(action_key)
    return f"You chose '{action_key}'."

# -------------------------
# Tools (function_tool)
# -------------------------
@function_tool
async def start_adventure(ctx: RunContext[Userdata], player_name: Annotated[Optional[str], Field(description="Player name", default=None)] = None) -> str:
    userdata = ctx.userdata
    if player_name:
        userdata.player_name = player_name
    userdata.current_scene = "intro"
    userdata.history = []
    userdata.journal = []
    userdata.inventory = []
    userdata.named_npcs = {}
    userdata.choices_made = []
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"

    opening = f"Welcome {userdata.player_name or 'traveler'} to the forest.\n\n" + scene_text("intro", userdata)
    return opening

@function_tool
async def get_scene(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    return scene_text(userdata.current_scene or "intro", userdata)

@function_tool
async def player_action(ctx: RunContext[Userdata], action: Annotated[str, Field(description="Player spoken action or the short action code")]) -> str:
    userdata = ctx.userdata
    current = userdata.current_scene or "intro"
    scene = WORLD.get(current)
    action_text = (action or "").strip().lower()

    chosen_key = None
    # exact key match
    if action_text in (scene.get("choices") or {}):
        chosen_key = action_text

    # fuzzy match by words in desc
    if not chosen_key:
        for cid, cmeta in (scene.get("choices") or {}).items():
            desc = cmeta.get("desc", "").lower()
            if cid in action_text or any(w in action_text for w in desc.split()[:4]):
                chosen_key = cid
                break

    # keyword match
    if not chosen_key:
        for cid, cmeta in (scene.get("choices") or {}).items():
            for kw in cmeta.get("desc", "").lower().split():
                if kw and kw in action_text:
                    chosen_key = cid
                    break
            if chosen_key:
                break

    if not chosen_key:
        return "Try: 'follow pawprints', 'check the den', or 'approach the fox'. What do you do?"

    choice_meta = scene["choices"].get(chosen_key)
    result_scene = choice_meta.get("result_scene", current)
    effects = choice_meta.get("effects", None)

    apply_effects(effects or {}, userdata)
    summarize_scene_transition(current, chosen_key, result_scene, userdata)
    userdata.current_scene = result_scene

    return scene_text(result_scene, userdata)

@function_tool
async def show_journal(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    lines = []
    lines.append(f"Traveler: {userdata.player_name or 'Anonymous'}")
    if userdata.journal:
        lines.append("Journal:")
        for j in userdata.journal:
            lines.append(f"• {j}")
    if userdata.inventory:
        lines.append("Inventory:")
        for it in userdata.inventory:
            lines.append(f"• {it}")
    else:
        lines.append("Inventory: Empty")
    lines.append("\nWhat do you do?")
    return "\n".join(lines)

@function_tool
async def restart_adventure(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    userdata.current_scene = "intro"
    userdata.history = []
    userdata.journal = []
    userdata.inventory = []
    userdata.named_npcs = {}
    userdata.choices_made = []
    userdata.session_id = str(uuid.uuid4())[:8]
    userdata.started_at = datetime.utcnow().isoformat() + "Z"

    greeting = "The forest resets. You stand at the clearing once more.\n\n" + scene_text("intro", userdata)
    if not greeting.endswith("What do you do?"):
        greeting += "\nWhat do you do?"
    return greeting

# -------------------------
# Agent & Entrypoint
# -------------------------
class GameMasterAgent(Agent):
    def __init__(self):
        instructions = """
        You are a gentle fantasy Game Master for a short forest quest called 'The Lost Lantern'.
        Tone: cozy, short, helpful.
        Rules:
        - Keep responses short (1-3 sentences).
        - Never list choices.
        - Always end with: "What do you do?"
        - Remember journal entries and inventory across turns.
        """
        super().__init__(instructions=instructions, tools=[start_adventure, get_scene, player_action, show_journal, restart_adventure])

def prewarm(proc: JobProcess):
    # optional VAD prewarm
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": getattr(ctx.room, "name", "local")}
    logger.info("🚀 STARTING THE LOST LANTERN ADVENTURE")

    userdata = Userdata()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-marcus", style="Conversational", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad") if hasattr(ctx, "proc") else None,
        userdata=userdata,
    )

    await session.start(
        agent=GameMasterAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
