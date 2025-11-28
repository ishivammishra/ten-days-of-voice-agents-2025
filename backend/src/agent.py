# agent.py -- Day 7 Food & Grocery Ordering Voice Agent 
import os
import json
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sys
import re

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

# -------------------------------------------------
#  CONFIG
# -------------------------------------------------
BASE_DIR = Path(__file__).parent
CATALOG_FILE = BASE_DIR / "catalog.json"
ORDERS_DIR = BASE_DIR / "orders"

# Ensure catalog.json exists
if not CATALOG_FILE.exists():
    sys.exit(f"ERROR: catalog.json not found at {CATALOG_FILE}. Please create it before running the agent.")

# Ensure orders/ directory exists
if not ORDERS_DIR.exists():
    sys.exit(f"ERROR: orders/ directory not found at {ORDERS_DIR}. Please create it before running the agent.")

load_dotenv(".env.local")

logger = logging.getLogger("shop_agent")
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

_file_lock = threading.Lock()


# -------------------------------------------------
#  LOAD CATALOG (required)
# -------------------------------------------------
def load_catalog():
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {item["id"]: item for item in items}


CATALOG = load_catalog()

# For simple matching
NAME_TO_ID = {}
for cid, item in CATALOG.items():
    NAME_TO_ID[item["name"].lower()] = cid
    for token in item["name"].lower().split():
        NAME_TO_ID.setdefault(token, cid)


# -------------------------------------------------
#  RECIPES (ingredients for X)
# -------------------------------------------------
RECIPES = {
    "peanut butter sandwich": ["bread_whole_wheat", "peanut_butter_200g"],
    "pasta for two": ["pasta_500g", "pasta_sauce_400g", "olive_oil_500ml"],
    "simple omelette": ["eggs_6", "butter_250g"],
    "jam toast": ["bread_whole_wheat", "jam_200g", "butter_250g"]
}


def find_item_id(text: str):
    t = text.lower().strip()
    if t in NAME_TO_ID:
        return NAME_TO_ID[t]
    for token in re.findall(r"[a-zA-Z0-9]+", t):
        if token in NAME_TO_ID:
            return NAME_TO_ID[token]
    return None


# -------------------------------------------------
#  CART STATE
# -------------------------------------------------
@dataclass
class CartState:
    user_name: str = ""
    cart: Dict[str, int] = field(default_factory=dict)
    transcript: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, cid: str, qty: int = 1):
        if cid not in CATALOG:
            return {"ok": False, "error": "item_not_found"}

        self.cart[cid] = self.cart.get(cid, 0) + qty
        return {"ok": True, "qty": self.cart[cid]}

    def remove(self, cid: str):
        if cid not in self.cart:
            return {"ok": False, "error": "not_in_cart"}
        del self.cart[cid]
        return {"ok": True}

    def update(self, cid: str, qty: int):
        if cid not in CATALOG:
            return {"ok": False, "error": "item_not_found"}

        if qty <= 0:
            return self.remove(cid)

        self.cart[cid] = qty
        return {"ok": True}

    def list_cart(self):
        items = []
        total = 0.0
        for cid, qty in self.cart.items():
            item = CATALOG[cid]
            line_total = item["price"] * qty
            items.append({
                "id": cid,
                "name": item["name"],
                "unit_price": item["price"],
                "quantity": qty,
                "line_total": line_total
            })
            total += line_total
        return {"ok": True, "items": items, "total": total}


# -------------------------------------------------
#  SAVE ORDER
# -------------------------------------------------
def save_order(order_obj: Dict[str, Any]) -> str:
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    fname = ORDERS_DIR / f"order_{ts}.json"
    with _file_lock:
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(order_obj, f, indent=2, ensure_ascii=False)
    return str(fname)


# -------------------------------------------------
#  FUNCTION TOOLS
# -------------------------------------------------
@function_tool
async def add_item(ctx: RunContext[CartState],
                   item_text: str = Field(description="Item name or id"),
                   quantity: int = Field(default=1)):

    state = ctx.userdata
    cid = item_text.strip()

    if cid not in CATALOG:
        found = find_item_id(item_text)
        if not found:
            return {"ok": False, "text": "I couldn't find that item."}
        cid = found

    res = state.add(cid, quantity)
    if not res["ok"]:
        return res

    return {
        "ok": True,
        "text": f"Added {quantity} x {CATALOG[cid]['name']} to your cart."
    }


@function_tool
async def remove_item(ctx: RunContext[CartState],
                      item_text: str = Field(description="Item name or id")):

    state = ctx.userdata
    cid = find_item_id(item_text)
    if not cid or cid not in state.cart:
        return {"ok": False, "text": "That item is not in your cart."}

    state.remove(cid)
    return {"ok": True, "text": f"Removed {CATALOG[cid]['name']} from your cart."}


@function_tool
async def update_quantity(ctx: RunContext[CartState],
                          item_text: str = Field(description="Item"),
                          quantity: int = Field(description="New quantity")):

    state = ctx.userdata
    cid = find_item_id(item_text)
    if not cid:
        return {"ok": False, "text": "I couldn't find that item."}

    res = state.update(cid, quantity)
    if not res["ok"]:
        return res
    return {"ok": True, "text": f"Updated quantity of {CATALOG[cid]['name']} to {quantity}."}


@function_tool
async def list_cart(ctx: RunContext[CartState]):
    return ctx.userdata.list_cart()


@function_tool
async def ingredients_for(ctx: RunContext[CartState],
                          dish: str = Field(description="Dish name")):

    state = ctx.userdata
    key = dish.lower().strip()

    recipe = RECIPES.get(key)
    if not recipe:
        return {"ok": False, "text": "I don't know that recipe yet."}

    added_items = []
    for cid in recipe:
        state.add(cid, 1)
        added_items.append(CATALOG[cid]["name"])

    return {
        "ok": True,
        "text": f"Added {', '.join(added_items)} for {dish}."
    }


@function_tool
async def place_order_tool(ctx: RunContext[CartState],
                           customer_name: Optional[str] = None,
                           address: Optional[str] = None):

    state = ctx.userdata
    cart = state.list_cart()

    if not cart["items"]:
        return {"ok": False, "text": "Your cart is empty."}

    order_obj = {
        "timestamp": datetime.utcnow().isoformat(),
        "customer_name": customer_name or "",
        "address": address or "",
        "items": cart["items"],
        "order_total": cart["total"]
    }

    fname = save_order(order_obj)
    state.cart = {}
    return {"ok": True, "text": f"Order placed and saved to {fname}", "file": fname}


# -------------------------------------------------
#  AGENT INSTRUCTIONS
# -------------------------------------------------
INSTRUCTIONS = """
You are Swiggy Assistant — a friendly voice-based grocery ordering helper.
You can:
- Add items to the cart
- Remove items
- Update quantities
- Handle "ingredients for X"
- List the cart
- Place the final order as a JSON file

Never invent items. Always confirm unclear quantities.
"""

class ShopAgent(Agent):
    def __init__(self):
        super().__init__(instructions=INSTRUCTIONS,
                         tools=[add_item, remove_item, update_quantity, list_cart,
                                ingredients_for, place_order_tool])


# -------------------------------------------------
#  ENTRYPOINT
# -------------------------------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except:
        proc.userdata["vad"] = None


async def entrypoint(ctx: JobContext):
    state = CartState()

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Friendly", text_pacing=True),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=state,
    )

    await session.start(
        agent=ShopAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()



if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
