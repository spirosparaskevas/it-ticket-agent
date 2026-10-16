"""
Mock IT ticketing API for the Agent Platform assignment.

- Auth: every /tickets and /admin route requires header  X-API-Key: <API_KEY>  (default dev-key-123)
- Pagination: GET /tickets returns pages of `page_size` (default 10, max 50)
- FLAKY=1: a share of requests (FLAKY_RATE, default 0.3) fail with 429 (Retry-After: 1) or 500
- LATENCY_MS: artificial delay added to every request
- POST /admin/reset restores the seed data (handy for evals)

Run:  uvicorn app:app --host 0.0.0.0 --port 8000      (or: docker compose up)
Docs: http://localhost:8000/docs   OpenAPI: http://localhost:8000/openapi.json
"""
import copy
import os
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from seed import SEED

API_KEY = os.getenv("API_KEY", "dev-key-123")
FLAKY = os.getenv("FLAKY", "0") == "1"
FLAKY_RATE = float(os.getenv("FLAKY_RATE", "0.3"))
LATENCY_MS = int(os.getenv("LATENCY_MS", "0"))
_seed_env = os.getenv("FLAKY_SEED")
_rng = random.Random(int(_seed_env)) if _seed_env else random.Random()

Status = Literal["open", "in_progress", "resolved", "closed"]
Priority = Literal["low", "medium", "high", "urgent"]

app = FastAPI(
    title="Mock IT Ticketing API",
    version="1.0.0",
    description="A small, deliberately imperfect internal ticketing system for the agent assignment. "
                "Requires X-API-Key. Paginates. Can be made flaky with FLAKY=1.",
)

DB: dict[str, dict] = {}
COUNTER = {"ticket": 0, "comment": 0}


def _reset():
    DB.clear()
    for t in copy.deepcopy(SEED):
        DB[t["id"]] = t
    COUNTER["ticket"] = 1000 + len(SEED)
    COUNTER["comment"] = 0


_reset()


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@app.middleware("http")
async def chaos_and_auth(request: Request, call_next):
    request_id = str(uuid.uuid4())
    path = request.url.path
    protected = path.startswith("/tickets") or path.startswith("/admin")
    response = None
    if protected:
        key = request.headers.get("x-api-key")
        if key != API_KEY:
            response = JSONResponse({"error": "unauthorized", "detail": "missing or invalid X-API-Key"}, status_code=401)
        else:
            if LATENCY_MS:
                time.sleep(LATENCY_MS / 1000)
            if FLAKY and _rng.random() < FLAKY_RATE:
                if _rng.random() < 0.5:
                    response = JSONResponse({"error": "rate_limited", "detail": "slow down"}, status_code=429, headers={"Retry-After": "1"})
                else:
                    response = JSONResponse({"error": "internal_error", "detail": "the ticketing system hiccuped"}, status_code=500)
    if response is None:
        response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


class TicketCreate(BaseModel):
    title: str = Field(..., min_length=3, max_length=120)
    description: str = Field("", max_length=2000)
    priority: Priority = "medium"
    category: str = Field("general", max_length=40)
    requester: str = Field(..., min_length=2, max_length=80)


class TicketUpdate(BaseModel):
    status: Optional[Status] = None
    priority: Optional[Priority] = None
    assignee: Optional[str] = Field(None, max_length=80)


class CommentCreate(BaseModel):
    author: str = Field(..., min_length=2, max_length=80)
    body: str = Field(..., min_length=1, max_length=2000)


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "flaky": FLAKY, "tickets": len(DB)}


@app.get("/tickets", tags=["tickets"], summary="Search tickets (paginated)")
def search_tickets(
    q: Optional[str] = Query(None, description="Free-text match on title and description (case-insensitive)"),
    status: Optional[Status] = None,
    priority: Optional[Priority] = None,
    requester: Optional[str] = Query(None, description="Exact requester name"),
    category: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=50),
):
    items = list(DB.values())
    if q:
        ql = q.lower()
        items = [t for t in items if ql in t["title"].lower() or ql in t["description"].lower()]
    if status:
        items = [t for t in items if t["status"] == status]
    if priority:
        items = [t for t in items if t["priority"] == priority]
    if requester:
        items = [t for t in items if t["requester"].lower() == requester.lower()]
    if category:
        items = [t for t in items if t["category"] == category]
    items.sort(key=lambda t: t["updated_at"], reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    page_items = [{k: v for k, v in t.items() if k != "comments"} for t in items[start:start + page_size]]
    return {"items": page_items, "page": page, "page_size": page_size, "total": total, "has_more": start + page_size < total}


@app.get("/tickets/{ticket_id}", tags=["tickets"], summary="Get one ticket with comments")
def get_ticket(ticket_id: str):
    t = DB.get(ticket_id.upper())
    if not t:
        raise HTTPException(404, f"ticket {ticket_id} not found")
    return t


@app.post("/tickets", status_code=201, tags=["tickets"], summary="Create a ticket")
def create_ticket(body: TicketCreate):
    COUNTER["ticket"] += 1
    tid = f"TCK-{COUNTER['ticket']}"
    now = _now()
    t = {"id": tid, **body.model_dump(), "status": "open", "assignee": None, "created_at": now, "updated_at": now, "comments": []}
    DB[tid] = t
    return t


@app.patch("/tickets/{ticket_id}", tags=["tickets"], summary="Update status, priority or assignee")
def update_ticket(ticket_id: str, body: TicketUpdate):
    t = DB.get(ticket_id.upper())
    if not t:
        raise HTTPException(404, f"ticket {ticket_id} not found")
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    if not changes:
        raise HTTPException(400, "nothing to update")
    t.update(changes)
    t["updated_at"] = _now()
    return t


@app.post("/tickets/{ticket_id}/comments", status_code=201, tags=["tickets"], summary="Add a comment")
def add_comment(ticket_id: str, body: CommentCreate):
    t = DB.get(ticket_id.upper())
    if not t:
        raise HTTPException(404, f"ticket {ticket_id} not found")
    COUNTER["comment"] += 1
    c = {"id": f"CMT-{COUNTER['comment']:04d}", **body.model_dump(), "created_at": _now()}
    t["comments"].append(c)
    t["updated_at"] = c["created_at"]
    return c


@app.post("/admin/reset", tags=["meta"], summary="Restore seed data")
def reset():
    _reset()
    return {"status": "reset", "tickets": len(DB)}
