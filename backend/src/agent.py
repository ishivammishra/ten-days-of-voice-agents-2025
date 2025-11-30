# src/agent.py -- Day 9: E-commerce Voice Agent 
"""
Day 9 — E-commerce Voice Agent (ACP-lite)
This single-file implementation is designed for the LiveKit-style environment used in Days 7-8.
It contains:
 - a compact in-process merchant layer (catalog + orders persistence)
 - robust product lookup/resolution helpers
 - function_tool wrappers exposed to the LLM:
     show_catalog, add_to_cart, show_cart, clear_cart, place_order, last_order
 - an Agent class wired to these tools (friendly shopkeeper persona)
Notes:
 - Orders are appended to orders.json in the working directory.
 - For a production / ACP-like separation, move the merchant layer into a small FastAPI app
   and call it from the tools via httpx.
"""
import os
import re
import json
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Annotated

from dotenv import load_dotenv
from pydantic import Field, BaseModel

# livekit / plugins (same pattern as your Day 8 / friend code)
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

load_dotenv(".env.local")

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("day9_shop")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# -------------------------
# Catalog (compact, Indian-flavored sample)
# -------------------------
CATALOG = [
    {
        "id": "mug-001",
        "name": "Stoneware Chai Mug",
        "description": "Hand-glazed ceramic mug perfect for masala chai.",
        "price": 299,
        "currency": "INR",
        "category": "mug",
        "color": "blue",
        "sizes": [],
    },
    {
        "id": "tee-001",
        "name": "Roshan Tee (Cotton)",
        "description": "Comfort-fit cotton t-shirt with subtle logo.",
        "price": 799,
        "currency": "INR",
        "category": "tshirt",
        "color": "black",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "hoodie-001",
        "name": "Cozy Hoodie",
        "description": "Warm pullover hoodie, fleece-lined.",
        "price": 1499,
        "currency": "INR",
        "category": "hoodie",
        "color": "grey",
        "sizes": ["M", "L", "XL"],
    },
    {
        "id": "mug-002",
        "name": "Insulated Travel Mug",
        "description": "Keeps chai warm on your way to work.",
        "price": 599,
        "currency": "INR",
        "category": "mug",
        "color": "white",
        "sizes": [],
    },
    {
        "id": "hoodie-002",
        "name": "Black Zip Hoodie",
        "description": "Lightweight zip-up hoodie, black.",
        "price": 1299,
        "currency": "INR",
        "category": "hoodie",
        "color": "black",
        "sizes": ["S", "M", "L"],
    },
    # T-shirts (expanded)
    {
        "id": "tee-002",
        "name": "Casual Cotton Tee",
        "description": "Everyday cotton t-shirt, breathable and soft.",
        "price": 299,
        "currency": "INR",
        "category": "tshirt",
        "color": "white",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "tee-003",
        "name": "Graphic Tee",
        "description": "Printed graphic t-shirt with vibrant design.",
        "price": 499,
        "currency": "INR",
        "category": "tshirt",
        "color": "navy",
        "sizes": ["S", "M", "L", "XL"],
    },
    {
        "id": "tee-004",
        "name": "Premium Polo Tee",
        "description": "Polo-style t-shirt with premium stitching.",
        "price": 999,
        "currency": "INR",
        "category": "tshirt",
        "color": "maroon",
        "sizes": ["M", "L", "XL"],
    },
    {
        "id": "tee-005",
        "name": "Summer V-neck Tee",
        "description": "Lightweight V-neck tee for hot days.",
        "price": 350,
        "currency": "INR",
        "category": "tshirt",
        "color": "sky",
        "sizes": ["S", "M", "L"],
    },
    {
        "id": "tee-006",
        "name": "Henley Tee",
        "description": "Smart casual henley style t-shirt.",
        "price": 699,
        "currency": "INR",
        "category": "tshirt",
        "color": "olive",
        "sizes": ["M", "L", "XL"],
    },
    # Raincoats / Outerwear
    {
        "id": "rain-001",
        "name": "Light Raincoat",
        "description": "Waterproof light raincoat, packable.",
        "price": 1299,
        "currency": "INR",
        "category": "raincoat",
        "color": "yellow",
        "sizes": ["M", "L", "XL"],
    },
    {
        "id": "rain-002",
        "name": "Heavy Duty Raincoat",
        "description": "Heavy-duty rainproof coat for monsoon.",
        "price": 2499,
        "currency": "INR",
        "category": "raincoat",
        "color": "navy",
        "sizes": ["L", "XL"],
    },
    # Laptops
    {
        "id": "laptop-001",
        "name": "Generic Laptop (50k)",
        "description": "A reliable laptop suitable for everyday use.",
        "price": 50000,
        "currency": "INR",
        "category": "laptop",
        "color": "silver",
        "sizes": [],
    },
    {
        "id": "laptop-002",
        "name": "Dell Inspiron (Budget)",
        "description": "Compact Dell laptop for students and professionals.",
        "price": 27800,
        "currency": "INR",
        "category": "laptop",
        "color": "black",
        "sizes": [],
    },
    {
        "id": "laptop-003",
        "name": "Lenovo ThinkPad",
        "description": "Durable Lenovo laptop with strong performance.",
        "price": 60000,
        "currency": "INR",
        "category": "laptop",
        "color": "black",
        "sizes": [],
    },
    {
        "id": "laptop-004",
        "name": "HP Pavilion",
        "description": "High-performance HP laptop for creators.",
        "price": 100000,
        "currency": "INR",
        "category": "laptop",
        "color": "silver",
        "sizes": [],
    },
    # Storage
    {
        "id": "storage-001",
        "name": "External Hard Disk 1TB",
        "description": "Portable external hard disk for backups.",
        "price": 50000,
        "currency": "INR",
        "category": "storage",
        "color": "black",
        "sizes": [],
    },
    # Mobile phones (10k - 50k examples)
    {
        "id": "phone-001",
        "name": "Redmi Note (Entry)",
        "description": "Affordable Redmi smartphone with solid features.",
        "price": 12000,
        "currency": "INR",
        "category": "mobile",
        "color": "blue",
        "sizes": [],
    },
    {
        "id": "phone-002",
        "name": "Oppo A-Series",
        "description": "Stylish Oppo phone with good camera.",
        "price": 18000,
        "currency": "INR",
        "category": "mobile",
        "color": "green",
        "sizes": [],
    },
    {
        "id": "phone-003",
        "name": "Samsung M-Series",
        "description": "Mid-range Samsung phone for everyday use.",
        "price": 25000,
        "currency": "INR",
        "category": "mobile",
        "color": "black",
        "sizes": [],
    },
    {
        "id": "phone-004",
        "name": "iPhone (Standard)",
        "description": "Apple iPhone model example (price varies by config).",
        "price": 50000,
        "currency": "INR",
        "category": "mobile",
        "color": "white",
        "sizes": [],
    },
    {
        "id": "phone-005",
        "name": "Oppo Reno",
        "description": "Higher-end Oppo phone with premium features.",
        "price": 35000,
        "currency": "INR",
        "category": "mobile",
        "color": "black",
        "sizes": [],
    },
    {
        "id": "phone-006",
        "name": "Redmi Pro",
        "description": "Redmi higher-tier phone with improved camera and battery.",
        "price": 22000,
        "currency": "INR",
        "category": "mobile",
        "color": "grey",
        "sizes": [],
    },
]

ORDERS_FILE = "orders.json"
# ensure orders storage exists
if not os.path.exists(ORDERS_FILE):
    with open(ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)


# -------------------------
# Persistence helpers
# -------------------------
def _load_all_orders() -> List[Dict[str, Any]]:
    try:
        with open(ORDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_order(order: Dict[str, Any]) -> None:
    orders = _load_all_orders()
    orders.append(order)
    with open(ORDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2, default=str)


def _get_most_recent_order() -> Optional[Dict[str, Any]]:
    orders = _load_all_orders()
    return orders[-1] if orders else None


# -------------------------
# Merchant-layer logic (filters, create order)
# -------------------------
def list_products(filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Naive but practical filtering supporting category, max_price, min_price, color, size, q (text)."""
    f = (filters or {}).copy()
    q = f.get("q")
    category = f.get("category")
    max_price = f.get("max_price") or f.get("to") or f.get("max")
    min_price = f.get("min_price") or f.get("from") or f.get("min")
    color = f.get("color")
    size = f.get("size")

    # normalize some synonyms
    if category:
        c = category.lower()
        if c in ("phone", "phones", "mobile", "mobiles"):
            category = "mobile"
        elif c in ("tshirt", "t-shirts", "tees", "tee"):
            category = "tshirt"
        else:
            category = c

    results = []
    for p in CATALOG:
        ok = True
        pcat = p.get("category", "").lower()
        if category:
            if pcat != category and category not in pcat and pcat not in category:
                ok = False
        if max_price:
            try:
                if p.get("price", 0) > int(max_price):
                    ok = False
            except Exception:
                pass
        if min_price:
            try:
                if p.get("price", 0) < int(min_price):
                    ok = False
            except Exception:
                pass
        if color:
            if not p.get("color") or p.get("color").lower() != str(color).lower():
                ok = False
        if size:
            if not p.get("sizes") or size not in p.get("sizes"):
                ok = False
        if q:
            ql = q.lower()
            # treat 'phone' queries as mobile category preference
            if any(w in ql for w in ("phone", "mobile")):
                if p.get("category") != "mobile":
                    ok = False
            else:
                if ql not in p.get("name", "").lower() and ql not in p.get("description", "").lower():
                    ok = False
        if ok:
            results.append(p)
    return results


def _lookup_product_by_id(pid: str) -> Dict[str, Any]:
    for p in CATALOG:
        if p["id"].lower() == pid.lower():
            return p
    raise KeyError(f"product not found: {pid}")


def create_order_object(line_items: List[Dict[str, Any]], currency: str = "INR") -> Dict[str, Any]:
    """line_items: [{product_id, quantity, attrs}] -> persisted order dict"""
    items = []
    total = 0
    for li in line_items:
        pid = li.get("product_id")
        qty = int(li.get("quantity", 1))
        prod = None
        # allow numeric references? Not at merchant layer — resolution happens in tools
        try:
            prod = _lookup_product_by_id(pid)
        except KeyError:
            raise ValueError(f"Product {pid} not found")
        line_total = prod["price"] * qty
        total += line_total
        items.append({
            "product_id": prod["id"],
            "name": prod["name"],
            "unit_price": prod["price"],
            "quantity": qty,
            "line_total": line_total,
            "attrs": li.get("attrs", {}),
        })
    order = {
        "id": f"order-{str(uuid.uuid4())[:8]}",
        "items": items,
        "total": total,
        "currency": currency,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    _save_order(order)
    return order


# -------------------------
# Reference resolution helpers
# -------------------------
ORDINALS = {"first": 0, "second": 1, "third": 2, "fourth": 3, "fifth": 4}


def find_product_by_ref(ref_text: str, candidates: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """
    Resolve a user's spoken ref like 'second hoodie', 'black hoodie', 'mug-001', '2' into a product dict.
    Heuristics:
     - ordinal words (first/second/...)
     - exact id match
     - color + category match
     - name substring / token matching
     - numeric index into 'candidates' if provided
    """
    if not ref_text:
        return None
    ref = ref_text.lower().strip()
    cand = candidates if candidates is not None else CATALOG

    # numeric token -> index
    for tok in ref.split():
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(cand):
                return cand[idx]

    # ordinals
    for word, idx in ORDINALS.items():
        if word in ref:
            if idx < len(cand):
                return cand[idx]

    # exact id
    for p in cand:
        if p["id"].lower() == ref:
            return p

    # color + category
    for p in cand:
        color = p.get("color", "").lower()
        cat = p.get("category", "").lower()
        if color and color in ref and cat and cat in ref:
            return p

    # name tokens match (all tokens of reasonable length present in name)
    tokens = [t for t in re.split(r"\W+", ref) if len(t) > 2]
    if tokens:
        for p in cand:
            name = p.get("name", "").lower()
            if all(tok in name for tok in tokens):
                return p

    # partial token heuristic
    for p in cand:
        for tok in tokens:
            if tok in p.get("name", "").lower():
                return p

    return None


# -------------------------
# Per-session Userdata
# -------------------------
@dataclass
class Userdata:
    customer_name: Optional[str] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    cart: List[Dict[str, Any]] = field(default_factory=list)      # list of {product_id, quantity, attrs}
    orders: List[Dict[str, Any]] = field(default_factory=list)    # orders placed this session
    last_browse: List[Dict[str, Any]] = field(default_factory=list)  # last catalog results shown


# -------------------------
# Pydantic models for function_tool inputs
# -------------------------
class CatalogFilterModel(BaseModel):
    q: Optional[str] = None
    category: Optional[str] = None
    max_price: Optional[int] = None
    color: Optional[str] = None
    size: Optional[str] = None


class AddToCartModel(BaseModel):
    product_ref: str
    quantity: Optional[int] = 1
    size: Optional[str] = None


class PlaceOrderModel(BaseModel):
    confirm: Optional[bool] = True


# -------------------------
# Tools exposed to the LLM
# -------------------------
@function_tool
async def show_catalog(
    ctx: RunContext[Userdata],
    filters: Annotated[Optional[CatalogFilterModel], Field(description="Optional filters: q, category, max_price, color, size")] = None,
) -> str:
    """
    Return a short spoken summary of matching products (name, price, id).
    Accepts a single optional CatalogFilterModel so function-calling validation
    works when the LLM sends {} or a partial object.
    Also saves the result in ctx.userdata.last_browse for reference resolution.
    """
    # normalize filters object
    f = (filters.dict() if filters else {}) if hasattr(filters, "dict") else (filters or {})
    # Extract individual values (defaulting to None)
    q = f.get("q")
    category = f.get("category")
    max_price = f.get("max_price")
    color = f.get("color")
    size = f.get("size")

    # Normalize category synonyms
    if category:
        cat = category.lower()
        if cat in ("phone", "phones", "mobile", "mobiles"):
            category = "mobile"
        elif cat in ("tshirt", "t-shirts", "tees", "tee"):
            category = "tshirt"
        else:
            category = cat

    filters_dict = {"q": q, "category": category, "max_price": max_price, "color": color, "size": size}
    prods = list_products({k: v for k, v in filters_dict.items() if v is not None})
    ctx.userdata.last_browse = prods

    if not prods:
        return "Sorry — I couldn't find any items that match. Would you like to try another search?"

    # Summarize top 6 for voice clarity
    lines = [f"Here are the top {min(6, len(prods))} items I found:"]
    for idx, p in enumerate(prods[:6], start=1):
        size_info = f" (sizes: {', '.join(p['sizes'])})" if p.get("sizes") else ""
        lines.append(f"{idx}. {p['name']} — {p['price']} {p['currency']} (id:{p['id']}){size_info}")
    lines.append("You can say 'add item 2 to my cart' or 'add mug-001, quantity 2'.")
    if any(p.get("category") == "mobile" for p in prods):
        lines.append("For phones you can say: 'I want the second phone' or use the product id.")
    return "\n".join(lines)



@function_tool
async def add_to_cart(
    ctx: RunContext[Userdata],
    product_ref: Annotated[str, Field(description="Reference to product: id, spoken ref, or ordinal")] ,
    quantity: Annotated[int, Field(description="Quantity", default=1)] = 1,
    size: Annotated[Optional[str], Field(description="Size (optional)", default=None)] = None,
) -> str:
    """
    Resolve product_ref using last_browse candidates if present; otherwise search whole catalog.
    Adds an item to the session cart.
    """
    userdata = ctx.userdata
    candidates = userdata.last_browse if getattr(userdata, "last_browse", None) else CATALOG
    prod = find_product_by_ref(product_ref, candidates)
    # fallback to global catalog
    if not prod:
        prod = find_product_by_ref(product_ref, CATALOG)
    if not prod:
        return "I couldn't resolve which product you meant. Try using the item id or say 'show catalog' to hear options."

    # size validation if provided
    if size:
        sizes = prod.get("sizes") or []
        if sizes and size not in sizes:
            return f"Sorry, size {size} isn't available for {prod['name']}. Available sizes: {', '.join(sizes)}."

    userdata.cart.append({"product_id": prod["id"], "quantity": int(quantity), "attrs": {"size": size} if size else {}})
    return f"Added {quantity} × {prod['name']} to your cart. Would you like to do anything else?"


@function_tool
async def show_cart(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    if not userdata.cart:
        return "Your cart is empty. Ask me to 'show catalog' to browse items."
    lines = ["Items in your cart:"]
    total = 0
    for li in userdata.cart:
        p = next((x for x in CATALOG if x["id"] == li["product_id"]), None)
        if not p:
            continue
        line_total = p["price"] * li.get("quantity", 1)
        total += line_total
        sz = li.get("attrs", {}).get("size")
        sz_text = f", size {sz}" if sz else ""
        lines.append(f"- {p['name']} x{li['quantity']}{sz_text}: {line_total} {p.get('currency','INR')}")
    lines.append(f"Cart total: {total} INR. Say 'place my order' to checkout or 'clear cart' to empty it.")
    return "\n".join(lines)


@function_tool
async def clear_cart(ctx: RunContext[Userdata]) -> str:
    userdata = ctx.userdata
    userdata.cart = []
    return "Your cart has been cleared. What would you like to do next?"


@function_tool
async def place_order(ctx: RunContext[Userdata], confirm: Annotated[bool, Field(description="Confirm", default=True)] = True) -> str:
    userdata = ctx.userdata
    if not userdata.cart:
        return "Your cart is empty — nothing to place. Would you like to browse items?"
    if not confirm:
        return "Order cancelled. Your cart is untouched."

    # build line_items for merchant layer
    line_items = []
    for li in userdata.cart:
        line_items.append({"product_id": li["product_id"], "quantity": li.get("quantity", 1), "attrs": li.get("attrs", {})})
    try:
        order = create_order_object(line_items)
    except Exception as e:
        logger.exception("create order failed")
        return f"Failed to create order: {e}"

    userdata.orders.append(order)
    userdata.cart = []  # clear cart after order
    return f"Order placed. Order ID {order['id']}. Total {order['total']} {order['currency']}. What would you like to do next?"


@function_tool
async def last_order(ctx: RunContext[Userdata]) -> str:
    ordc = _get_most_recent_order()
    if not ordc:
        return "You have no past orders yet."
    lines = [f"Most recent order: {ordc['id']} — {ordc['created_at']}"]
    for it in ordc["items"]:
        lines.append(f"- {it['name']} x{it['quantity']}: {it['line_total']} {ordc['currency']}")
    lines.append(f"Total: {ordc['total']} {ordc['currency']}")
    return "\n".join(lines)


# -------------------------
# Agent: persona and tooling
# -------------------------
class EcomAgent(Agent):
    def __init__(self):
        instructions = """
        You are 'Ramu Kaka', a friendly neighbourhood shopkeeper and voice shopping assistant.
        Tone: warm, helpful, concise. Speak in short sentences suitable for TTS.
        Use the provided tools for all commerce actions:
          - show_catalog to browse products (do not invent product lists yourself)
          - add_to_cart to add items using product ids or spoken refs (e.g., 'second hoodie')
          - show_cart, clear_cart, place_order, last_order as needed
        When placing an order, confirm order id and total in your reply.
        Keep replies short and always end with a question asking what the user wants next.
        """
        super().__init__(instructions=instructions, tools=[show_catalog, add_to_cart, show_cart, clear_cart, place_order, last_order])


# -------------------------
# Entrypoint & prewarm
# -------------------------
def prewarm(proc: JobProcess):
    # try to load VAD (non-fatal)
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": getattr(ctx.room, "name", "local")}
    logger.info("🚀 STARTING DAY 9 E-COMMERCE AGENT (Ramu Kaka)")

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
        agent=EcomAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
