# agent.py
# --------------------------------------------------------------
# LogisticsAgent – MeTTa-powered routing, pricing, risk & LLM polish
# --------------------------------------------------------------

from __future__ import annotations
import re
import json
import math
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
from uuid import uuid4
from collections import defaultdict

from pydantic import BaseModel as PydanticBaseModel
import aiohttp

# ---------- MeTTa ----------
from hyperon import MeTTa

# ---------- uAgents ----------
from uagents import Agent, Context, Model
from uagents.mailbox import MailboxClient

from uagents_core.contrib.protocols.chat import (
    chat_protocol_spec,
    ChatMessage,
    ChatAcknowledgement,
    TextContent,
    EndSessionContent,
    StartSessionContent,
)

# --------------------------------------------------------------
# Hard-coded configuration (no .env)
# --------------------------------------------------------------
class Settings:
    ASI_API_KEY = ""
    ASI_BASE_URL = "https://api.asi1.ai/v1"
    LLM_MODEL = "asi1-mini"
    LLM_TIMEOUT_SECONDS = 20

    ORS_API_KEY = ""
    ORS_BASE = "https://api.openrouteservice.org/v2/directions/driving-car"

    PARTNER_AGENT_ADDRESS = ""  # fill if you have a partner

    BASE_RATE_PER_MILE = 0.09
    INSURANCE_PER_KG = 0.005
    BASE_HANDLING_FEE = 10.0
    FUEL_SURCHARGE_PCT_BASE = 0.06
    PERISHABLE_SURCHARGE_PCT = 0.20

    LOG_LEVEL = "INFO"
    AGENT_NAME = "LogisticsRoute"
    AGENT_VERSION = "1.1.1"

settings = Settings()
# --------------------------------------------------------------
# FastAPI (ASI-1.ai OpenAI-compatible endpoint)
# --------------------------------------------------------------
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel as FastAPIBaseModel
import uvicorn
import threading

app = FastAPI(title="LogisticsAgent – ASI-1.ai Direct")

# -----------------------------------------------------------------
# 1. Your secret API key – give ONLY this to ASI-1.ai
# -----------------------------------------------------------------
ASI_AGENT_API_KEY = "sk_logistics_8f3e2d1c9a7b6e5d4c3b2a1f0e9d8c7b6a5f4e3d2c1b0a9"   # ← CHANGE THIS

# -----------------------------------------------------------------
# 2. Security – validate Bearer token
# -----------------------------------------------------------------
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != ASI_AGENT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

# -----------------------------------------------------------------
# 3. OpenAI-compatible request models
# -----------------------------------------------------------------
class ChatCompletionMessage(FastAPIBaseModel):
    role: str
    content: str

class ChatCompletionRequest(FastAPIBaseModel):
    model: str = "logistics-agent"
    messages: List[ChatCompletionMessage]
    temperature: float = 0.2
    max_tokens: int = 300
# --------------------------------------------------------------
# Agent definition
# --------------------------------------------------------------
agent = Agent(
    name=settings.AGENT_NAME,
    seed="logistics_seed_metta",
    port=8001,
    mailbox=True,
)

mailbox_client = MailboxClient(
    identity="agent1qg9xyclyvxcmkrjp0qm6lcjdsh23wm86dcampwm7gucqjrdjyyxpg36trgx",
    agentverse="https://agentverse.ai"
)

# -----------------------------------------------------------------
# 4. Pending replies (session → future)
# -----------------------------------------------------------------
_pending: dict[str, asyncio.Future] = {}

# -----------------------------------------------------------------
# 5. When the agent replies → fulfill the HTTP request
# -----------------------------------------------------------------
@agent.on_message(model=ChatMessage)
async def _asi_reply_handler(ctx: Context, sender: str, msg: ChatMessage):
    if not isinstance(msg.content, TextContent):
        return
    text = msg.content.text
    # fulfil first waiting future
    for sid, fut in list(_pending.items()):
        if not fut.done():
            fut.set_result(text)
            del _pending[sid]
            break

# -----------------------------------------------------------------
# 6. /v1/chat/completions – ASI-1.ai calls this
# -----------------------------------------------------------------
@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key)
):
    # take last user message
    user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
    if not user_msg:
        raise HTTPException(status_code=400, detail="No user message")

    session_id = str(hash(user_msg))          # simple session
    future = asyncio.Future()
    _pending[session_id] = future

    # forward to our own agent
    await agent.ctx.send(
        agent.address,
        ChatMessage(content=TextContent(text=user_msg)),
        protocol=chat_protocol_spec,
    )

    try:
        reply = await asyncio.wait_for(future, timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Agent timeout")

    # OpenAI format
    return {
        "id": f"chatcmpl-{uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(user_msg.split()),
            "completion_tokens": len(reply.split()),
            "total_tokens": len(user_msg.split()) + len(reply.split())
        }
    }



# --------------------------------------------------------------
# Logging
# --------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(settings.AGENT_NAME)

# --------------------------------------------------------------
# Constants
# --------------------------------------------------------------
CONGESTION_MULTIPLIER: Dict[str, float] = {
    "NY": 1.12, "NJ": 1.08, "CA": 1.15, "IL": 1.10, "TX": 1.05, "FL": 1.04
}

MODE_RATES = {
    "truck": {"per_mile_per_ton": 2.20, "co2_kg_per_ton_mile": 0.27},
    "rail": {"per_mile_per_ton": 0.50, "co2_kg_per_ton_mile": 0.11},
    "air": {"per_kg_per_km": 0.0015, "co2_kg_per_kg_km": 0.50},
}

STATE_CAPITALS = {  # (lat, lon)
    "AL": (32.377716, -86.300568), "AK": (58.301598, -134.420212), "AZ": (33.448143, -112.096962),
    "AR": (34.746613, -92.288986), "CA": (38.576668, -121.493629), "CO": (39.739227, -104.984856),
    "CT": (41.758611, -72.674333), "DE": (39.157307, -75.519722), "FL": (30.438118, -84.281296),
    "GA": (33.749027, -84.388229), "HI": (21.307442, -157.857376), "ID": (43.617775, -116.199722),
    "IL": (39.798363, -89.654961), "IN": (39.768623, -86.162643), "IA": (41.586835, -93.624964),
    "KS": (39.048191, -95.677956), "KY": (38.186722, -84.875374), "LA": (30.457069, -91.187393),
    "ME": (44.307167, -69.781693), "MD": (38.978764, -76.490936), "MA": (42.358162, -71.063698),
    "MI": (42.733635, -84.555328), "MN": (44.955097, -93.102211), "MS": (32.29869, -90.180489),
    "MO": (38.579201, -92.172935), "MT": (46.585709, -112.018417), "NE": (40.808075, -96.699654),
    "NV": (39.163914, -119.766121), "NH": (43.206898, -71.537994), "NJ": (40.220596, -74.769913),
    "NM": (35.68224, -105.939728), "NY": (42.652843, -73.757874), "NC": (35.78043, -78.639099),
    "ND": (46.82085, -100.783318), "OH": (39.961346, -82.999069), "OK": (35.492207, -97.503342),
    "OR": (44.938461, -123.030403), "PA": (40.264378, -76.883598), "RI": (41.830914, -71.414963),
    "SC": (34.000343, -81.033211), "SD": (44.367031, -100.346405), "TN": (36.16581, -86.784241),
    "TX": (30.27467, -97.740349), "UT": (40.777477, -111.888237), "VT": (44.262436, -72.580536),
    "VA": (37.538857, -77.43364), "WA": (47.035805, -122.905014), "WV": (38.336246, -81.612328),
    "WI": (43.074684, -89.384445), "WY": (41.140259, -104.820236)
}

# --------------------------------------------------------------
# Utilities
# --------------------------------------------------------------
def now_iso() -> str:
    return datetime.utcnow().isoformat()

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    R = 6371.0
    x = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(x), math.sqrt(1 - x))
    return R * c

def km_to_miles(km: float) -> float:
    return km / 1.60934

# --------------------------------------------------------------
# aiohttp helper
# --------------------------------------------------------------
class HTTPClient:
    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        self._session = session

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def post_json(
        self,
        url: str,
        headers: Dict[str, str],
        json_body: Dict[str, Any],
        timeout: int = 12,
    ) -> Optional[Dict[str, Any]]:
        session = await self._get_session()
        try:
            async with session.post(url, json=json_body, headers=headers, timeout=timeout) as resp:
                if resp.status != 200:
                    logger.debug("HTTP %s %s", resp.status, await resp.text())
                    return None
                return await resp.json()
        except Exception as e:
            logger.debug("HTTP post failed: %s", e)
            return None

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

http_client = HTTPClient()

# --------------------------------------------------------------
# MeTTa Knowledge Graph (simple, functional persistence)
# --------------------------------------------------------------
class MeTTaKG:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.metta = MeTTa()
        self._load_initial_rules()
        self._load_from_storage()

    # ----- core rules -------------------------------------------------
    def _load_initial_rules(self):
        rules = """
        ; alias edge → distance
        (= (edge $A $B $D) (distance $A $B $D))

        ; direct path
        (= (path $A $B $D $P)
           (edge $A $B $D)
           (cons $A (cons $B ())))

        ; one-hop recursion (up to 4 hops in practice)
        (= (path $A $C $Tot $P)
           (edge $A $B $D1)
           (path $B $C $D2 $SubP)
           (= $Tot (+ $D1 $D2))
           (cons $A $SubP))
        """
        self.metta.run(rules)

    # ----- persistence ------------------------------------------------
    def _load_from_storage(self):
        raw = self.ctx.storage.get("metta_atoms") or "[]"
        atoms = json.loads(raw) if isinstance(raw, (str, bytes)) else []
        for a in atoms:
            try:
                self.metta.parse_all(a)
            except Exception as e:
                logger.debug("load atom error: %s", e)

    def _save_to_storage(self):
        atoms = [str(a) for a in self.metta.run("(get-atoms)")]
        self.ctx.storage.set("metta_atoms", json.dumps(atoms))

    # ----- public API -------------------------------------------------
    async def declare_route(self, a: str, b: str, distance_km: float, mode: str = "road") -> None:
        a_u, b_u = a.upper(), b.upper()
        fact = f"(= (distance {a_u} {b_u} {distance_km}) (meta {a_u} {b_u} {mode} \"{now_iso()}\"))"
        self.metta.run(fact)
        self._save_to_storage()

    def get_route(self, a: str, b: str) -> Optional[Dict[str, Any]]:
        a_u, b_u = a.upper(), b.upper()
        q = f"(distance {a_u} {b_u} $D)"
        res = self.metta.run(f"(query " + q + ")")
        if res and res[0].bindings:
            d = float(list(res[0].bindings.values())[0])
            mres = self.metta.run(f"(query (meta {a_u} {b_u} $M $T))")
            meta = {"mode": "unknown", "ts": now_iso()}
            if mres and mres[0].bindings:
                meta["mode"] = list(mres[0].bindings.values())[0]
                meta["ts"] = list(mres[0].bindings.values())[1]
            return {"origin": a_u, "dest": b_u, "distance_km": d, "meta": meta}
        return None

    def find_multi_hop(self, origin: str, dest: str, max_hops: int = 6) -> Optional[Dict[str, Any]]:
        o_u, d_u = origin.upper(), dest.upper()
        q = f"(path {o_u} {d_u} $Tot $P)"
        res = self.metta.run(f"(query " + q + ")")
        if res and res[0].bindings:
            total = float(res[0].bindings["$Tot"])
            path = str(res[0].bindings["$P"]).strip("()").split()
            hops = len(path) - 1
            if hops > max_hops:
                return None
            return {
                "distance_km": total,
                "distance_miles": km_to_miles(total),
                "base_time_hours": total / 80.0,
                "hops": hops,
                "path": path,
                "risk": "medium" if hops > 0 else "low"
            }
        return None

# --------------------------------------------------------------
# Baseline graph (state-to-state distances)
# --------------------------------------------------------------
async def build_baseline(ctx: Context, max_neighbors: int = 6):
    kg = MeTTaKG(ctx)
    coords = {st: STATE_CAPITALS[st] for st in STATE_CAPITALS}
    pairs = []
    for a, ca in coords.items():
        for b, cb in coords.items():
            if a == b:
                continue
            d = haversine_km(ca, cb)
            pairs.append((a, b, d))

    by_origin = defaultdict(list)
    for a, b, d in pairs:
        by_origin[a].append((b, d))
      
    for a, neigh in by_origin.items():
        neigh.sort(key=lambda x: x[1])
        for b, d in neigh[:max_neighbors]:
            if kg.get_route(a, b) is None:
                await kg.declare_route(a, b, d, mode="baseline")

# --------------------------------------------------------------
# ORS enrichment (cached)
# --------------------------------------------------------------
async def ors_driving_distance_km(ctx: Context, origin_abbr: str, dest_abbr: str, kg: MeTTaKG) -> Optional[float]:
    if not settings.ORS_API_KEY:
        return None

    cache_key = f"ors:{origin_abbr}-{dest_abbr}"
    cached_raw = ctx.storage.get(cache_key)
    if cached_raw:
        try:
            payload = json.loads(cached_raw)
            gen = datetime.fromisoformat(payload["ts"])
            if datetime.utcnow() - gen < timedelta(days=7):
                return payload["distance_km"]
        except Exception:
            pass

    if origin_abbr not in STATE_CAPITALS or dest_abbr not in STATE_CAPITALS:
        return None

    ac = STATE_CAPITALS[origin_abbr]
    bc = STATE_CAPITALS[dest_abbr]
    coords = [[ac[1], ac[0]], [bc[1], bc[0]]]
    headers = {"Authorization": settings.ORS_API_KEY, "Content-Type": "application/json"}
    body = {"coordinates": coords}

    j = await http_client.post_json(settings.ORS_BASE, headers, body, timeout=12)
    if not j:
        return None

    try:
        distance_m = j["features"][0]["properties"]["segments"][0]["distance"]
        distance_km = distance_m / 1000.0
        ctx.storage.set(
            cache_key,
            json.dumps({"distance_km": distance_km, "ts": datetime.utcnow().isoformat()})
        )
        await kg.declare_route(origin_abbr, dest_abbr, distance_km, mode="ors")
        return distance_km
    except Exception:
        return None

# --------------------------------------------------------------
# Incident ledger & forecasting
# --------------------------------------------------------------
async def report_incident_store(
    ctx: Context, origin: str, dest: str, severity: int = 1, note: str = ""
) -> Dict[str, Any]:
    ledger_key = "risk_ledger"
    raw = ctx.storage.get(ledger_key) or "{}"
    ledger = json.loads(raw) if isinstance(raw, (str, bytes)) else {}

    key = f"{origin.upper()}-{dest.upper()}"
    entry = ledger.get(key, {"incidents": 0, "severity_score": 0, "notes": []})
    entry["incidents"] = entry.get("incidents", 0) + 1
    entry["severity_score"] = entry.get("severity_score", 0) + severity
    entry["last_incident"] = now_iso()
    entry["notes"].append({"ts": now_iso(), "note": note})
    ledger[key] = entry
    ctx.storage.set(ledger_key, json.dumps(ledger))
    return entry

async def forecast_route(ctx: Context, origin: str, dest: str) -> Dict[str, Any]:
    key = f"{origin.upper()}-{dest.upper()}"
    cache_key = f"forecast:{key}"
    cached_raw = ctx.storage.get(cache_key)
    if cached_raw:
        try:
            cache = json.loads(cached_raw)
            gen = datetime.fromisoformat(cache["generated"])
            if datetime.utcnow() - gen < timedelta(hours=12):
                return cache["forecast"]
        except Exception:
            pass

    raw = ctx.storage.get("risk_ledger") or "{}"
    ledger = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
    entry = ledger.get(key, {"incidents": 0, "severity_score": 0})

    incidents = entry.get("incidents", 0)
    severity = entry.get("severity_score", 0)
    surge = 1.0 + min(0.5, incidents * 0.05 + severity * 0.02)
    delay = min(168, incidents * 4 + severity * 3 + (time.time() % 8))

    forecast = {"surge_multiplier": round(surge, 2), "expected_delay_hours": int(delay)}
    ctx.storage.set(
        cache_key, json.dumps({"generated": now_iso(), "forecast": forecast})
    )
    return forecast

# --------------------------------------------------------------
# Pricing & CO₂
# --------------------------------------------------------------
def calculate_pricing_and_co2(
    distance_miles: float, weight_kg: float, mode: str = "truck", surge: float = 1.0
) -> Tuple[float, float]:
    weight_kg = max(0.0, float(weight_kg))
    tonnes = max(0.001, weight_kg / 1000.0)
    mode_rates = MODE_RATES.get(mode, MODE_RATES["truck"])

    if mode == "air":
        km = distance_miles * 1.60934
        cost = weight_kg * km * mode_rates["per_kg_per_km"] * surge
        co2 = weight_kg * km * mode_rates["co2_kg_per_kg_km"]
    else:
        cost = distance_miles * tonnes * mode_rates["per_mile_per_ton"] * surge
        co2 = tonnes * distance_miles * mode_rates["co2_kg_per_ton_mile"]
    return cost, co2

# --------------------------------------------------------------
# LLM helper (ASI)
# --------------------------------------------------------------
async def llm_call(
    prompt: str, temperature: float = 0.7, max_tokens: int = 300, timeout: Optional[int] = None
) -> Optional[str]:
    if not settings.ASI_API_KEY:
        return None

    url = f"{settings.ASI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.ASI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a logistics domain expert. Be concise and actionable."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    j = await http_client.post_json(
        url, headers, body, timeout=timeout or settings.LLM_TIMEOUT_SECONDS
    )
    if not j:
        return None
    try:
        return j["choices"][0]["message"]["content"]
    except Exception:
        return None

# --------------------------------------------------------------
# Pydantic models
# --------------------------------------------------------------
class PlanDeliveryRequest(Model):
    origin: str
    destination: str
    weight_kg: Optional[float] = 100.0
    cargo: Optional[str] = "general"
    prefer_mode: Optional[str] = None

class PlanDeliveryResponse(Model):
    distance_miles: Optional[int] = None
    est_time_hours: Optional[float] = None
    risk: Optional[str] = None
    carrier: Optional[str] = None
    cost_usd: Optional[float] = None
    notes: Optional[str] = None
    co2_kg: Optional[float] = None
    llm_text: Optional[str] = None
    error: Optional[str] = None

class AddRouteFactRequest(Model):
    origin: str
    dest: str
    distance_km: float

class AddRouteFactResponse(Model):
    status: str
    origin: str
    dest: str
    distance_km: float

class QueryRouteRequest(Model):
    origin: str
    dest: str

class QueryRouteResponse(Model):
    found: Optional[str] = None
    distance_miles: Optional[float] = None
    path: Optional[List[str]] = None
    message: Optional[str] = None

class IncidentEntry(Model):
    incidents: int
    severity_score: int
    last_incident: str
    notes: List[Dict[str, str]]

class Forecast(Model):
    surge_multiplier: float
    expected_delay_hours: int

class ReportIncidentRequest(Model):
    origin: str
    dest: str
    severity: Optional[int] = 1
    note: Optional[str] = ""

class ReportIncidentResponse(Model):
    status: str
    entry: IncidentEntry
    forecast: Forecast

class VoteCarrierRequest(Model):
    carrier: str
    sentiment: str

class VoteCarrierResponse(Model):
    carrier: str
    current_score: float
    votes: int

class Shipment(Model):
    id: str
    ts: str
    origin: str
    dest: str
    weight_kg: float
    cost_usd: float

class ListShipmentsRequest(Model):
    pass

class ListShipmentsResponse(Model):
    shipments: List[Shipment]

class GetBadgesRequest(Model):
    user: Optional[str] = None

class GetBadgesResponse(Model):
    badges: Dict[str, List[str]]

class EscalationMessage(Model):
    escalation_type: str = "escalation"
    origin: str
    dest: str
    weight_kg: float
    cargo: Optional[str] = None
    risk: Optional[str] = None
    delay_hours: Optional[int] = None
    notes: Optional[str] = None



# --------------------------------------------------------------
# Global KG (created in startup)
# --------------------------------------------------------------
kg: MeTTaKG

# --------------------------------------------------------------
# Handlers
# --------------------------------------------------------------
@agent.on_event("startup")
async def startup(ctx: Context):
    global kg
    ctx.logger.info(f"Starting {settings.AGENT_NAME} – address: {agent.address}")
    
    kg = MeTTaKG(ctx)
    await agent.register(chat_protocol_spec)  # Register Chat Protocol

    # baseline graph (once)
    if not ctx.storage.get("baseline_built"):
        await build_baseline(ctx, max_neighbors=6)
        ctx.storage.set("baseline_built", json.dumps({"ts": now_iso()}))

    # optional ORS background sync
    if settings.ORS_API_KEY and not ctx.storage.get("ors_sync_started"):
        async def _ors_sync():
            origins = list(STATE_CAPITALS.keys())
            for a in origins:
                for b in origins:
                    if a == b:
                        continue
                    try:
                        await ors_driving_distance_km(ctx, a, b, kg)
                    except Exception as e:
                        ctx.logger.debug(f"ORS sync {a}-{b} failed: {e}")
            ctx.storage.set("ors_sync_started", json.dumps({"ts": now_iso()}))
        asyncio.create_task(_ors_sync())

@agent.on_query(model=PlanDeliveryRequest, replies={PlanDeliveryResponse})
async def handle_plan_delivery(ctx: Context, sender: str, msg: PlanDeliveryRequest):
    ctx.logger.info(f"PlanDelivery from {sender}: {msg.origin} → {msg.destination}")

    origin_u, dest_u = msg.origin.upper(), msg.destination.upper()
    if origin_u not in STATE_CAPITALS or dest_u not in STATE_CAPITALS:
        await ctx.send(sender, PlanDeliveryResponse(error="Use US state abbreviations (NY, CA, …)."),protocol=chat_protocol_spec,)
        return

    direct = kg.get_route(origin_u, dest_u)
    if not direct and settings.ORS_API_KEY:
        km = await ors_driving_distance_km(ctx, origin_u, dest_u, kg)
        if km:
            direct = kg.get_route(origin_u, dest_u)

    if direct:
        miles = km_to_miles(direct["distance_km"])
        route_data = {
            "distance_miles": miles,
            "base_time_hours": direct["distance_km"] / 80.0,
            "mode": "truck",
            "carrier": "BaselineCarrier",
            "path": [origin_u, dest_u],
            "risk": "low",
        }
    else:
        multi = kg.find_multi_hop(origin_u, dest_u, max_hops=6)
        if not multi:
            await ctx.send(sender, PlanDeliveryResponse(error="No route found."),protocol=chat_protocol_spec,)
            return
        route_data = {
            "distance_miles": multi["distance_miles"],
            "base_time_hours": multi["base_time_hours"],
            "mode": "truck",
            "carrier": "Multi-Carrier",
            "path": multi["path"],
            "risk": multi["risk"],
        }

    forecast = await forecast_route(ctx, origin_u, dest_u)
    surge = forecast.get("surge_multiplier", 1.0)
    delay = forecast.get("expected_delay_hours", 0)
    est_time = route_data["base_time_hours"] * surge + delay

    base_risk = route_data.get("risk", "unknown")
    if delay > 48 or surge > 1.4:
        risk = "high"
    elif surge > 1.15 or delay > 24:
        risk = "medium" if base_risk == "low" else "high"
    else:
        risk = base_risk

    if msg.prefer_mode and msg.prefer_mode.lower() in MODE_RATES:
        route_data["mode"] = msg.prefer_mode.lower()

    base_cost, co2 = calculate_pricing_and_co2(
        route_data["distance_miles"], msg.weight_kg, mode=route_data["mode"], surge=surge
    )
    fuel_surcharge = base_cost * (settings.FUEL_SURCHARGE_PCT_BASE + max(0.0, (surge - 1.0) * 0.10))
    cargo_surcharge = (
        base_cost * settings.PERISHABLE_SURCHARGE_PCT
        if "perish" in msg.cargo.lower() or "hazard" in msg.cargo.lower()
        else 0.0
    )
    handling = settings.BASE_HANDLING_FEE if msg.weight_kg > 1000 else 0.0
    insurance = msg.weight_kg * settings.INSURANCE_PER_KG
    cong_mult = max(CONGESTION_MULTIPLIER.get(origin_u, 1.0), CONGESTION_MULTIPLIER.get(dest_u, 1.0))

    total_cost = (base_cost + fuel_surcharge + cargo_surcharge + handling + insurance) * cong_mult

    notes = f"Route: {'→'.join(route_data['path'])}. Delay: {delay}h. Surge x{surge}."
    if risk == "high":
        notes += " High risk — consider escalation."

    try:
        raw = ctx.storage.get("shipments") or "[]"
        shipments = json.loads(raw) if isinstance(raw, (str, bytes)) else []
        shipments.append({
            "id": str(uuid4()),
            "ts": now_iso(),
            "origin": origin_u,
            "dest": dest_u,
            "weight_kg": msg.weight_kg,
            "cost_usd": round(total_cost, 2),
        })
        ctx.storage.set("shipments", json.dumps(shipments))

        raw_b = ctx.storage.get("badges") or "{}"
        badges = json.loads(raw_b) if isinstance(raw_b, (str, bytes)) else {}
        user_badges = badges.get(sender, [])
        if "First Shipment" not in user_badges:
            user_badges.append("First Shipment")
            badges[sender] = user_badges
            ctx.storage.set("badges", json.dumps(badges))
    except Exception as e:
        ctx.logger.debug("Persist error: %s", e)

    if (risk == "high" or delay > 72) and settings.PARTNER_AGENT_ADDRESS:
        try:
            await ctx.send(
                settings.PARTNER_AGENT_ADDRESS,
                EscalationMessage(
                    origin=origin_u,
                    dest=dest_u,
                    weight_kg=msg.weight_kg,
                    cargo=msg.cargo,
                    risk=risk,
                    delay_hours=delay,
                    notes=notes,
                ),
            )
            notes += " Escalated to partner."
        except Exception as e:
            ctx.logger.debug("Escalation failed: %s", e)

    llm_text = None
    try:
        prompt = (
            f"Quote: {origin_u}→{dest_u}\n"
            f"Distance: {int(route_data['distance_miles'])} mi\n"
            f"ETA: {round(est_time,1)} h\n"
            f"Cost: ${round(total_cost,2)}\n"
            f"CO₂: {round(co2,2)} kg\n"
            f"Carrier: {route_data['carrier']}\n"
            f"Risk: {risk}\n"
            f"Notes: {notes}\n"
            "Respond concisely."
        )
        llm_text = await llm_call(prompt, temperature=0.2, max_tokens=200)
    except Exception:
        pass

    await ctx.send(
        sender,
        PlanDeliveryResponse(
            distance_miles=int(round(route_data["distance_miles"])),
            est_time_hours=round(est_time, 1),
            risk=risk,
            carrier=route_data["carrier"],
            cost_usd=round(total_cost, 2),
            notes=notes,
            co2_kg=round(co2, 2),
            llm_text=llm_text,
        ),
    )
    ctx.logger.info("PlanDelivery response sent")

@agent.on_query(model=AddRouteFactRequest, replies={AddRouteFactResponse})
async def handle_add_route_fact(ctx: Context, sender: str, msg: AddRouteFactRequest):
    await kg.declare_route(msg.origin, msg.dest, msg.distance_km, mode="user")
    await ctx.send(
        sender,
        AddRouteFactResponse(
            status="ok",
            origin=msg.origin.upper(),
            dest=msg.dest.upper(),
            distance_km=msg.distance_km,
        ),
	protocol=chat_protocol_spec,
    )

@agent.on_query(model=QueryRouteRequest, replies={QueryRouteResponse})
async def handle_query_route(ctx: Context, sender: str, msg: QueryRouteRequest):
    o, d = msg.origin.upper(), msg.dest.upper()
    direct = kg.get_route(o, d)
    if direct:
        await ctx.send(
            sender,
            QueryRouteResponse(
                found="direct",
                distance_miles=round(km_to_miles(direct["distance_km"]), 1),
                path=[o, d],
            ),
	    protocol=chat_protocol_spec,
        )
        return

    multi = kg.find_multi_hop(o, d, max_hops=6)
    if multi:
        await ctx.send(
            sender,
            QueryRouteResponse(
                found="multi",
                distance_miles=round(multi["distance_miles"], 1),
                path=multi["path"],
            ),
        )
        return

    await ctx.send(sender, QueryRouteResponse(found="False", message="No route found"))

@agent.on_query(model=ReportIncidentRequest, replies={ReportIncidentResponse})
async def handle_report_incident(ctx: Context, sender: str, msg: ReportIncidentRequest):
    entry = await report_incident_store(
        ctx, msg.origin, msg.dest, severity=msg.severity, note=msg.note
    )
    forecast = await forecast_route(ctx, msg.origin, msg.dest)

    await ctx.send(
        sender,
        ReportIncidentResponse(
            status="reported",
            entry=IncidentEntry(
                incidents=entry.get("incidents", 0),
                severity_score=entry.get("severity_score", 0),
                last_incident=entry.get("last_incident", now_iso()),
                notes=entry.get("notes", []),
            ),
            forecast=Forecast(**forecast),
        ),protocol=chat_protocol_spec,
    )

@agent.on_query(model=VoteCarrierRequest, replies={VoteCarrierResponse})
async def handle_vote_carrier(ctx: Context, sender: str, msg: VoteCarrierRequest):
    vote = 1 if msg.sentiment.lower() in ("good", "great", "excellent") else -1
    key = "carrier_reputation"
    raw = ctx.storage.get(key) or "{}"
    rep = json.loads(raw) if isinstance(raw, (str, bytes)) else {}

    carrier = msg.carrier.title()
    cur = rep.get(carrier, {"score_sum": 0.0, "votes": 0})
    cur["votes"] = cur.get("votes", 0) + 1
    cur["score_sum"] = cur.get("score_sum", 0.0) + vote
    cur["score"] = cur["score_sum"] / cur["votes"]
    rep[carrier] = cur
    ctx.storage.set(key, json.dumps(rep))

    await ctx.send(
        sender,
        VoteCarrierResponse(carrier=carrier, current_score=cur["score"], votes=cur["votes"]),
	protocol=chat_protocol_spec,
    )

@agent.on_query(model=ListShipmentsRequest, replies={ListShipmentsResponse})
async def handle_list_shipments(ctx: Context, sender: str, _: ListShipmentsRequest):
    raw = ctx.storage.get("shipments") or "[]"
    shipments = [Shipment(**s) for s in (json.loads(raw) if isinstance(raw, (str, bytes)) else [])]
    await ctx.send(sender, ListShipmentsResponse(shipments=shipments))

@agent.on_query(model=GetBadgesRequest, replies={GetBadgesResponse})
async def handle_get_badges(ctx: Context, sender: str, msg: GetBadgesRequest):
    raw = ctx.storage.get("badges") or "{}"
    badges = json.loads(raw) if isinstance(raw, (str, bytes)) else {}
    if msg.user:
        await ctx.send(sender, GetBadgesResponse(badges={msg.user: badges.get(msg.user, [])}),protocol=chat_protocol_spec)
    else:
        await ctx.send(sender, GetBadgesResponse(badges=badges),protocol=chat_protocol_spec)

@agent.on_message(model=ChatMessage, replies={ChatMessage, ChatAcknowledgement})
async def handle_chat_message(ctx: Context, sender: str, msg: ChatMessage):
    ctx.logger.info(f"Received ChatMessage from {sender}: {msg}")
    response_text = "Please use a valid command, e.g., 'plan delivery from NY to CA', 'report incident from NY to CA', or 'list shipments'."

    # Validate message content
    if not hasattr(msg, 'content') or not msg.content:
        ctx.logger.error("Invalid message: No content provided")
        await ctx.send(
            sender,
            ChatMessage(content=[TextContent(text=response_text)])
        )
        await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))
        return

    # Extract text content
    text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
    user_query = " ".join(b.text for b in text_blocks).strip().lower()
    ctx.logger.debug(f"Processed user query: {user_query}")

    if not user_query:
        ctx.logger.warning("Empty query received")
        await ctx.send(
            sender,
            ChatMessage(content=[TextContent(text=response_text)])
        )
        await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))
        return

    try:
        # Handle "plan delivery" command
        if "plan delivery" in user_query:
            match = re.search(r'plan\s+delivery\s+from\s+(\w+)\s+to\s+(\w+)(?:\s+weight\s+(\d+))?(?:\s+cargo\s+(\w+))?', user_query, re.IGNORECASE)
            if not match:
                response_text = "Invalid plan delivery command. Use format: 'plan delivery from NY to CA [weight 1000] [cargo general]'"
                await ctx.send(
                    sender,
                    ChatMessage(content=[TextContent(text=response_text)])
                )
                await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))
                return

            origin, dest, weight, cargo = match.groups()
            weight = float(weight) if weight else 100.0
            cargo = cargo if cargo else "general"

            delivery_req = PlanDeliveryRequest(
                origin=origin,
                destination=dest,
                weight_kg=weight,
                cargo=cargo,
                prefer_mode=None
            )
            ctx.logger.debug(f"Plan delivery request: {delivery_req}")

            temp_response = PlanDeliveryResponse()
            try:
                origin_u, dest_u = delivery_req.origin.upper(), delivery_req.destination.upper()
                if origin_u not in STATE_CAPITALS or dest_u not in STATE_CAPITALS:
                    temp_response.error = "Use US state abbreviations (NY, CA, …)."
                else:
                    direct = kg.get_route(origin_u, dest_u)
                    if not direct and settings.ORS_API_KEY:
                        km = await ors_driving_distance_km(ctx, origin_u, dest_u, kg)
                        if km:
                            direct = kg.get_route(origin_u, dest_u)

                    if direct:
                        miles = km_to_miles(direct["distance_km"])
                        route_data = {
                            "distance_miles": miles,
                            "base_time_hours": direct["distance_km"] / 80.0,
                            "mode": "truck",
                            "carrier": "BaselineCarrier",
                            "path": [origin_u, dest_u],
                            "risk": "low",
                        }
                    else:
                        multi = kg.find_multi_hop(origin_u, dest_u, max_hops=6)
                        if not multi:
                            temp_response.error = "No route found."
                        else:
                            route_data = {
                                "distance_miles": multi["distance_miles"],
                                "base_time_hours": multi["base_time_hours"],
                                "mode": "truck",
                                "carrier": "Multi-Carrier",
                                "path": multi["path"],
                                "risk": multi["risk"],
                            }

                    if not temp_response.error:
                        forecast = await forecast_route(ctx, origin_u, dest_u)
                        surge = forecast.get("surge_multiplier", 1.0)
                        delay = forecast.get("expected_delay_hours", 0)
                        est_time = route_data["base_time_hours"] * surge + delay

                        base_risk = route_data.get("risk", "unknown")
                        if delay > 48 or surge > 1.4:
                            risk = "high"
                        elif surge > 1.15 or delay > 24:
                            risk = "medium" if base_risk == "low" else "high"
                        else:
                            risk = base_risk

                        base_cost, co2 = calculate_pricing_and_co2(
                            route_data["distance_miles"], delivery_req.weight_kg, mode=route_data["mode"], surge=surge
                        )
                        fuel_surcharge = base_cost * (settings.FUEL_SURCHARGE_PCT_BASE + max(0.0, (surge - 1.0) * 0.10))
                        cargo_surcharge = (
                            base_cost * settings.PERISHABLE_SURCHARGE_PCT
                            if "perish" in delivery_req.cargo.lower() or "hazard" in delivery_req.cargo.lower()
                            else 0.0
                        )
                        handling = settings.BASE_HANDLING_FEE if delivery_req.weight_kg > 1000 else 0.0
                        insurance = delivery_req.weight_kg * settings.INSURANCE_PER_KG
                        cong_mult = max(CONGESTION_MULTIPLIER.get(origin_u, 1.0), CONGESTION_MULTIPLIER.get(dest_u, 1.0))
                        total_cost = (base_cost + fuel_surcharge + cargo_surcharge + handling + insurance) * cong_mult

                        notes = f"Route: {'→'.join(route_data['path'])}. Delay: {delay}h. Surge x{surge}."
                        if risk == "high":
                            notes += " High risk — consider escalation."

                        raw = ctx.storage.get("shipments") or "[]"
                        shipments = json.loads(raw) if isinstance(raw, (str, bytes)) else []
                        shipments.append({
                            "id": str(uuid4()),
                            "ts": now_iso(),
                            "origin": origin_u,
                            "dest": dest_u,
                            "weight_kg": delivery_req.weight_kg,
                            "cost_usd": round(total_cost, 2),
                        })
                        ctx.storage.set("shipments", json.dumps(shipments))

                        llm_text = None
                        prompt = (
                            f"Quote: {origin_u}→{dest_u}\n"
                            f"Distance: {int(route_data['distance_miles'])} mi\n"
                            f"ETA: {round(est_time,1)} h\n"
                            f"Cost: ${round(total_cost,2)}\n"
                            f"CO₂: {round(co2,2)} kg\n"
                            f"Carrier: {route_data['carrier']}\n"
                            f"Risk: {risk}\n"
                            f"Notes: {notes}\n"
                            "Respond concisely."
                        )
                        llm_text = await llm_call(prompt, temperature=0.2, max_tokens=200)

                        temp_response = PlanDeliveryResponse(
                            distance_miles=int(round(route_data["distance_miles"])),
                            est_time_hours=round(est_time, 1),
                            risk=risk,
                            carrier=route_data["carrier"],
                            cost_usd=round(total_cost, 2),
                            notes=notes,
                            co2_kg=round(co2, 2),
                            llm_text=llm_text,
                        )

                if temp_response.error:
                    response_text = f"Error: {temp_response.error}"
                else:
                    response_text = temp_response.llm_text or (
                        f"Delivery plan: {origin_u}→{dest_u}\n"
                        f"Distance: {temp_response.distance_miles} miles\n"
                        f"ETA: {temp_response.est_time_hours} hours\n"
                        f"Cost: ${temp_response.cost_usd}\n"
                        f"CO₂: {temp_response.co2_kg} kg\n"
                        f"Carrier: {temp_response.carrier}\n"
                        f"Risk: {temp_response.risk}\n"
                        f"Notes: {temp_response.notes}"
                    )

            except Exception as e:
                ctx.logger.error(f"Plan delivery error: {e}")
                response_text = f"Error processing plan delivery: {str(e)}"

        # Handle "report incident" command
        elif "report incident" in user_query:
            match = re.search(r'report\s+incident\s+from\s+(\w+)\s+to\s+(\w+)(?:\s+severity\s+(\d+))?(?:\s+note\s+(.+))?', user_query, re.IGNORECASE)
            if not match:
                response_text = "Invalid report incident command. Use format: 'report incident from NY to CA [severity 3] [note Road closure]'"
                await ctx.send(
                    sender,
                    ChatMessage(content=[TextContent(text=response_text)])
                )
                await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))
                return

            origin, dest, severity, note = match.groups()
            severity = int(severity) if severity else 1
            note = note if note else ""

            incident_req = ReportIncidentRequest(origin=origin, dest=dest, severity=severity, note=note)
            ctx.logger.debug(f"Report incident request: {incident_req}")
            entry = await report_incident_store(ctx, origin, dest, severity, note)
            forecast = await forecast_route(ctx, origin, dest)

            response_text = (
                f"Incident reported: {origin}→{dest}\n"
                f"Incidents: {entry['incidents']}\n"
                f"Severity Score: {entry['severity_score']}\n"
                f"Last Incident: {entry['last_incident']}\n"
                f"Forecast - Surge: x{forecast['surge_multiplier']}, Delay: {forecast['expected_delay_hours']}h"
            )

        # Handle "list shipments" command
        elif "list shipments" in user_query:
            ctx.logger.debug("Listing shipments")
            raw = ctx.storage.get("shipments") or "[]"
            shipments = [Shipment(**s) for s in (json.loads(raw) if isinstance(raw, (str, bytes)) else [])]
            if not shipments:
                response_text = "No shipments found."
            else:
                response_text = "Shipments:\n" + "\n".join(
                    f"ID: {s.id}, {s.origin}→{s.dest}, {s.weight_kg}kg, ${s.cost_usd}, {s.ts}"
                    for s in shipments
                )

        # Fallback for unrecognized commands
        else:
            llm_prompt = f"Interpret logistics command: '{user_query}'. Suggest a response or correction."
            llm_response = await llm_call(llm_prompt, temperature=0.5, max_tokens=100)
            response_text = llm_response or response_text
            ctx.logger.debug(f"LLM response for unrecognized command: {llm_response}")

        # Send response
        await ctx.send(
            sender,
            ChatMessage(content=[TextContent(text=response_text)])
        )

    except Exception as e:
        ctx.logger.error(f"Chat handler error: {e}")
        response_text = f"Error processing your request: {str(e)}"
        await ctx.send(
            sender,
            ChatMessage(content=[TextContent(text=response_text)])
        )

    # Acknowledge message
    await ctx.send(sender, ChatAcknowledgement(acknowledged_msg_id=msg.msg_id))# --------------------------------------------------------------
# Graceful shutdown
# --------------------------------------------------------------
@agent.on_event("shutdown")
async def shutdown(ctx: Context):
    kg._save_to_storage()
    await http_client.close()
    ctx.logger.info("Agent shut down – MeTTa KG saved, aiohttp closed.")

# --------------------------------------------------------------
# Local runner – starts BOTH the uAgent AND the FastAPI server
# --------------------------------------------------------------
if __name__ == "__main__":
    

    print(f"\nLogistics Agent Running")
    print(f"Address : {agent.address}")
    print(f"Mailbox : http://127.0.0.1:8001/submit\n")
    agent.run()
