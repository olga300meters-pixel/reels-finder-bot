"""
Reels Finder — Instagram DM bot (official Instagram Messaging API).

A user forwards a Reel / video / photo to the business account in Direct,
taps a Quick Reply, and within ~3-5 s the bot identifies:
  * the song            (AudD API, 8-second mono MP3 clip)
  * the movie / series  (Gemini Vision, 3 frames)
  * clothes / items     (Gemini Vision, 3 frames)

Stack: FastAPI + uvicorn, httpx (pooled AsyncClient), asyncpg, google-genai,
ffmpeg via asyncio.create_subprocess_exec.

IMPORTANT: run with exactly ONE uvicorn worker / ONE replica. Webhook
de-duplication and per-user locks live in process memory.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import hmac
import json
import logging
import os
import re
import shutil
import string
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote_plus, urlencode, urlparse

import asyncpg
import httpx
from cachetools import TTLCache
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

# =============================================================================
# Logging
# =============================================================================

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("reels-finder")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)


# =============================================================================
# Configuration
# =============================================================================


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(name, default)
    if value is not None:
        value = value.strip()
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value or ""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw and raw.strip() else default


def _env_list(name: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, "").split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    # Instagram / Meta
    ig_access_token: str
    ig_verify_token: str
    ig_app_secrets: list[str]
    ig_account_id: str
    graph_api_base: str
    bot_name: str
    contact_email: str
    instagram_handle: str
    operator_name: str
    # Database
    database_url: str
    # Gemini
    gemini_api_key: str
    gemini_primary_model: str
    gemini_fallback_model: str
    gemini_timeout_sec: float
    gemini_thinking_level: str
    # AudD
    audd_api_token: str
    audd_timeout_sec: float
    # Lemon Squeezy
    lemon_checkout_url: str
    lemon_subscription_checkout_url: str
    lemon_signing_secret: str
    lemon_subscription_variant_ids: list[str]
    lemon_credit_pack_variant_ids: list[str]
    paywall_image_url: str
    # Business rules
    free_credits: int
    credits_per_pack: int
    subscription_days: int
    # Runtime limits
    max_download_mb: int
    max_parallel_jobs: int
    tmp_root: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            ig_access_token=_env("IG_ACCESS_TOKEN", required=True),
            ig_verify_token=_env("IG_VERIFY_TOKEN", required=True),
            ig_app_secrets=_env_list("IG_APP_SECRET") or [_env("IG_APP_SECRET", required=True)],
            ig_account_id=_env("IG_ACCOUNT_ID", "me"),
            # Unified login tokens (EAA...) work on graph.facebook.com; Instagram-login tokens (IG...)
            # on graph.instagram.com. Picked automatically unless GRAPH_API_BASE is set.
            graph_api_base=(_env("GRAPH_API_BASE") or (
                "https://graph.facebook.com/v26.0" if _env("IG_ACCESS_TOKEN").startswith("EAA")
                else "https://graph.instagram.com/v24.0")).rstrip("/"),
            bot_name=_env("BOT_NAME", "Reels Finder"),
            contact_email=_env("CONTACT_EMAIL", ""),
            instagram_handle=_env("INSTAGRAM_HANDLE", "").lstrip("@"),
            operator_name=_env("OPERATOR_NAME", ""),
            database_url=_env("DATABASE_URL", required=True),
            gemini_api_key=_env("GEMINI_API_KEY", required=True),
            gemini_primary_model=_env("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash"),
            gemini_fallback_model=_env("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"),
            gemini_timeout_sec=_env_float("GEMINI_TIMEOUT_SEC", 8.0),
            gemini_thinking_level=_env("GEMINI_THINKING_LEVEL", "low").upper(),
            audd_api_token=_env("AUDD_API_TOKEN", required=True),
            audd_timeout_sec=_env_float("AUDD_TIMEOUT_SEC", 8.0),
            lemon_checkout_url=_env("LEMON_CHECKOUT_URL", ""),  # empty = payments off
            lemon_subscription_checkout_url=_env("LEMON_SUBSCRIPTION_CHECKOUT_URL", ""),
            lemon_signing_secret=_env("LEMON_SIGNING_SECRET", ""),
            lemon_subscription_variant_ids=_env_list("LEMON_SUBSCRIPTION_VARIANT_IDS"),
            lemon_credit_pack_variant_ids=_env_list("LEMON_CREDIT_PACK_VARIANT_IDS"),
            paywall_image_url=_env("PAYWALL_IMAGE_URL", ""),
            free_credits=_env_int("FREE_CREDITS", 3),
            credits_per_pack=_env_int("CREDITS_PER_PACK", 50),
            subscription_days=_env_int("SUBSCRIPTION_DAYS", 30),
            max_download_mb=_env_int("MAX_DOWNLOAD_MB", 80),
            max_parallel_jobs=_env_int("MAX_PARALLEL_JOBS", 4),
            tmp_root=_env("TMP_ROOT", "/tmp"),
        )


# =============================================================================
# Constants & user-facing copy (English UI)
# =============================================================================

FIND_MUSIC = "FIND_MUSIC"
FIND_MOVIE = "FIND_MOVIE"
FIND_ITEM = "FIND_ITEM"
FIND_ALL = "FIND_ALL"
ACTIONS = {FIND_MUSIC, FIND_MOVIE, FIND_ITEM, FIND_ALL}
ACTION_COST = {FIND_MUSIC: 1, FIND_MOVIE: 1, FIND_ITEM: 1, FIND_ALL: 2}

QUICK_REPLIES = [
    {"content_type": "text", "title": "🎵 Find Song", "payload": FIND_MUSIC},
    {"content_type": "text", "title": "🎬 Find Movie", "payload": FIND_MOVIE},
    {"content_type": "text", "title": "👕 Find Outfit/Item", "payload": FIND_ITEM},
    {"content_type": "text", "title": "🔥 Find Everything", "payload": FIND_ALL},
]

# Typed fallbacks (Quick Replies are not rendered on Instagram desktop).
TEXT_COMMANDS = {
    "song": FIND_MUSIC, "music": FIND_MUSIC, "find song": FIND_MUSIC, "track": FIND_MUSIC,
    "movie": FIND_MOVIE, "film": FIND_MOVIE, "show": FIND_MOVIE, "series": FIND_MOVIE,
    "find movie": FIND_MOVIE,
    "outfit": FIND_ITEM, "item": FIND_ITEM, "items": FIND_ITEM, "clothes": FIND_ITEM,
    "find outfit": FIND_ITEM, "find item": FIND_ITEM,
    "all": FIND_ALL, "everything": FIND_ALL, "find everything": FIND_ALL,
}
BALANCE_COMMANDS = {"balance", "credits", "status", "my credits"}
BUY_COMMANDS = {"buy", "upgrade", "pricing", "premium", "subscribe", "pay"}
DELETE_COMMANDS = {"delete my data", "delete data", "delete my account", "remove my data"}
MSG_DATA_DELETED = (
    "Done ✅ Your data has been deleted: your ID, balance, subscription and last media link.\n"
    "If you message me again, a new empty profile will be created (free searches are not renewed)."
)

MSG_OVERLOAD = "Temporary overload, your credit was not deducted. Please try again in a moment."
MSG_MENU = "Got it! 🔥 What should I find in this one?\n(On desktop? Just reply: song, movie, outfit or all)"
MSG_NO_MEDIA = "Send me a Reel, video or photo first 🙂\nTap ✈️ under any Reel → send it to me."
MSG_LINK_UNSUPPORTED = (
    "I can't open Instagram / TikTok / YouTube links directly 🙈\n"
    "Please share the Reel itself: tap ✈️ under the Reel and send it to me — it works instantly."
)
MSG_MEDIA_RESTRICTED = (
    "I couldn't access this Reel 😕 It may be from a private account or restricted by the author.\n"
    "Try another Reel, or send the video file / a screenshot."
)
MSG_EXPIRED = (
    "This media is no longer available (Instagram links expire after a while). "
    "Please share the Reel again 🙏 No credit was used."
)
MSG_DOWNLOAD_FAILED = "I couldn't download this media 😕 Please share it again. No credit was used."
MSG_TOO_BIG = "This file is too large for me 😅 Please send a shorter clip. No credit was used."
MSG_NOT_MEDIA = "That link doesn't point to a video or image I can read 🤔 Please share the Reel itself."
MSG_UNSUPPORTED = "I couldn't read this file format 😕 Please send a Reel, an MP4 video or a JPG/PNG photo."
MSG_NO_AUDIO = "This one has no sound, so there's no song to find 🔇 No credit was used."
MSG_PHOTO_NO_AUDIO = "That's a photo — there's no sound to recognise 🙂 Send a Reel or video for song search. No credit was used."
MSG_NO_VIDEO = "That's an audio clip — I can only find songs in it 🎵 Tap “Find Song”."
MSG_NOT_FOUND = {
    FIND_MUSIC: "😕 I couldn't recognise the song in this clip — it may be an original sound, a voice-over or a heavy remix.\nNo credit was used.",
    FIND_MOVIE: "😕 I couldn't confidently identify a movie or TV show here — it may be original content.\nNo credit was used.",
    FIND_ITEM: "😕 I couldn't spot any identifiable clothes or items here.\nNo credit was used.",
    FIND_ALL: "😕 I couldn't find a song, a movie/show or identifiable items in this one.\nNo credits were used.",
}
MSG_PAYMENTS_SOON = (
    "You've used all your free searches 🔋\n"
    "Paid plans are coming very soon — stay tuned, I'll be ready for your next Reel!"
)
MSG_PAYMENT_CREDITS = "✅ Payment received — thank you! +{n} credits added.\nYour balance: {balance} credits.\nSend me any Reel to continue 🎬"
MSG_PAYMENT_SUB = "✅ Unlimited is active until {date}! 🔥\nSend me any Reel — every search is on us now."


def welcome_text(bot_name: str, balance_line: str) -> str:
    return (
        f"Hey there! 👋 I'm {bot_name}.\n\n"
        "Send me any Reel (tap ✈️ under the Reel → send it to me), a video or a photo, "
        "and in a few seconds I'll find:\n"
        "🎵 the song\n🎬 the movie or TV show\n👕 the outfit & items\n\n"
        f"{balance_line}"
    )


def private_reply_text(free_credits: int) -> str:
    return (
        "Hey! 👋 Thanks for the comment!\n"
        "I find the song 🎵, the movie 🎬 or the outfit 👕 from any Reel in seconds.\n\n"
        "👉 Forward me any Reel right here in DM (tap ✈️ under it → send to me) and pick what to find. "
        f"Your first {free_credits} searches are free!"
    )


# Domains whose *page* links cannot be fetched through the official API.
UNSUPPORTED_LINK_DOMAINS = (
    "instagram.com", "instagr.am", "tiktok.com", "youtube.com", "youtu.be",
    "facebook.com", "fb.watch", "x.com", "twitter.com", "pinterest.",
)
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
FIND_KEYWORD_RE = re.compile(r"\bfind\b", re.IGNORECASE)

IG_TEXT_LIMIT = 950  # Instagram text messages must be < 1000 characters


# =============================================================================
# Errors
# =============================================================================


class ExternalServiceError(Exception):
    """An upstream AI/recognition API is overloaded or down. Never charge the user."""


class MediaUnavailable(Exception):
    """The media cannot be processed; carries a user-facing explanation."""

    def __init__(self, user_message: str, log_reason: str = ""):
        super().__init__(log_reason or user_message)
        self.user_message = user_message


class GraphAPIError(Exception):
    def __init__(self, status: int, code: int | None, subcode: int | None, message: str):
        super().__init__(f"Graph API error status={status} code={code} subcode={subcode}: {message}")
        self.status = status
        self.code = code
        self.subcode = subcode
        self.message = message

    @property
    def outside_messaging_window(self) -> bool:
        msg = (self.message or "").lower()
        return (
            self.subcode in (2018278, 2534022)
            or self.code == 10
            or "outside of allowed window" in msg
            or "outside the allowed window" in msg
        )


# =============================================================================
# Idempotency & per-user locking
# =============================================================================


class Deduplicator:
    """Remembers the last N processed webhook ids (message mid / comment id)."""

    def __init__(self, maxsize: int = 1000, ttl_sec: int = 6 * 3600):
        self._cache: TTLCache[str, bool] = TTLCache(maxsize=maxsize, ttl=ttl_sec)

    def seen(self, key: str) -> bool:
        """Returns True if the key was already processed, otherwise records it."""
        if key in self._cache:
            return True
        self._cache[key] = True
        return False


class UserLocks:
    """Serialises jobs of one user so double taps can't double-spend credits."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._waiters: dict[str, int] = {}

    @asynccontextmanager
    async def hold(self, key: str):
        lock = self._locks.setdefault(key, asyncio.Lock())
        self._waiters[key] = self._waiters.get(key, 0) + 1
        try:
            async with lock:
                yield
        finally:
            self._waiters[key] -= 1
            if self._waiters[key] <= 0:
                self._waiters.pop(key, None)
                self._locks.pop(key, None)


# =============================================================================
# Database (asyncpg)
# =============================================================================

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    igsid           TEXT PRIMARY KEY,
    credits         INT NOT NULL DEFAULT 3,
    sub_until       TIMESTAMPTZ NULL,
    last_media_url  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_media_type TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS pending_notice  TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at      TIMESTAMPTZ NOT NULL DEFAULT now();

CREATE TABLE IF NOT EXISTS payment_events (
    event_key   TEXT PRIMARY KEY,
    igsid       TEXT,
    event_name  TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS deleted_users (
    id_hash     TEXT PRIMARY KEY,
    deleted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS app_kv (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class Database:
    def __init__(self, dsn: str, free_credits: int):
        self._dsn = self._normalize_dsn(dsn)
        self._free_credits = free_credits
        self.pool: asyncpg.Pool | None = None

    @staticmethod
    def _normalize_dsn(dsn: str) -> str:
        """asyncpg doesn't understand libpq-only params that Neon/Supabase add to URLs."""
        parts = urlparse(dsn)
        if not parts.query:
            return dsn
        unsupported = {"channel_binding", "pgbouncer", "connection_limit", "pool_timeout"}
        query = "&".join(
            kv for kv in parts.query.split("&") if kv and kv.split("=", 1)[0].lower() not in unsupported
        )
        return parts._replace(query=query).geturl()

    async def connect(self) -> None:
        last_exc: Exception | None = None
        for attempt in range(1, 6):
            try:
                self.pool = await asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=1,
                    max_size=5,
                    command_timeout=10,
                    max_inactive_connection_lifetime=120,
                    # Required for Supabase/Neon PgBouncer poolers (transaction mode).
                    statement_cache_size=0,
                )
                async with self.pool.acquire() as conn:
                    await conn.execute(SCHEMA_SQL)
                log.info("Database connected, schema ensured")
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.warning("DB connect attempt %s failed: %s", attempt, exc)
                await asyncio.sleep(min(2 * attempt, 8))
        raise RuntimeError(f"Could not connect to database: {last_exc}")

    async def close(self) -> None:
        if self.pool:
            await self.pool.close()

    async def ping(self) -> bool:
        assert self.pool
        try:
            async with self.pool.acquire() as conn:
                return await conn.fetchval("SELECT 1") == 1
        except Exception:  # noqa: BLE001
            return False

    # --- users ---------------------------------------------------------------

    @staticmethod
    def _id_hash(igsid: str) -> str:
        return hashlib.sha256(f"reels-finder:{igsid}".encode()).hexdigest()

    async def _starting_credits(self, igsid: str) -> int:
        """Users who deleted their data don't get the free searches again."""
        assert self.pool
        deleted = await self.pool.fetchval(
            "SELECT 1 FROM deleted_users WHERE id_hash = $1", self._id_hash(igsid))
        return 0 if deleted else self._free_credits

    async def get_or_create_user(self, igsid: str) -> asyncpg.Record:
        assert self.pool
        row = await self.pool.fetchrow("SELECT * FROM users WHERE igsid = $1", igsid)
        if row:
            return row
        return await self.pool.fetchrow(
            """
            INSERT INTO users (igsid, credits) VALUES ($1, $2)
            ON CONFLICT (igsid) DO UPDATE SET igsid = EXCLUDED.igsid
            RETURNING *
            """,
            igsid,
            await self._starting_credits(igsid),
        )

    async def delete_user(self, igsid: str) -> None:
        """Deletes the user's profile; keeps only a one-way hash to block free-credit farming."""
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("DELETE FROM users WHERE igsid = $1", igsid)
            await conn.execute(
                "INSERT INTO deleted_users (id_hash) VALUES ($1) ON CONFLICT (id_hash) DO NOTHING",
                self._id_hash(igsid))

    async def set_last_media(self, igsid: str, url: str, media_type: str) -> None:
        assert self.pool
        await self.pool.execute(
            """
            INSERT INTO users (igsid, credits, last_media_url, last_media_type)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (igsid) DO UPDATE
               SET last_media_url = EXCLUDED.last_media_url,
                   last_media_type = EXCLUDED.last_media_type,
                   updated_at = now()
            """,
            igsid,
            await self._starting_credits(igsid),
            url,
            media_type,
        )

    async def charge(self, igsid: str, cost: int) -> int | None:
        """Atomically deducts credits. Returns the new balance or None if insufficient."""
        assert self.pool
        return await self.pool.fetchval(
            """
            UPDATE users SET credits = credits - $2, updated_at = now()
             WHERE igsid = $1 AND credits >= $2
         RETURNING credits
            """,
            igsid,
            cost,
        )

    async def set_pending_notice(self, igsid: str, text: str) -> None:
        assert self.pool
        await self.pool.execute(
            """
            UPDATE users
               SET pending_notice = COALESCE(pending_notice || E'\\n\\n', '') || $2,
                   updated_at = now()
             WHERE igsid = $1
            """,
            igsid,
            text,
        )

    async def pop_pending_notice(self, igsid: str) -> str | None:
        assert self.pool
        return await self.pool.fetchval(
            """
            UPDATE users u SET pending_notice = NULL
              FROM (SELECT igsid, pending_notice FROM users WHERE igsid = $1 FOR UPDATE) old
             WHERE u.igsid = old.igsid AND old.pending_notice IS NOT NULL
         RETURNING old.pending_notice
            """,
            igsid,
        )

    # --- payments --------------------------------------------------------------

    async def apply_credit_pack(self, event_key: str, igsid: str, event_name: str, credits: int) -> int | None:
        """Returns new balance, or None if this payment event was already processed."""
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            claimed = await conn.fetchval(
                "INSERT INTO payment_events (event_key, igsid, event_name) VALUES ($1, $2, $3) "
                "ON CONFLICT (event_key) DO NOTHING RETURNING event_key",
                event_key, igsid, event_name,
            )
            if not claimed:
                return None
            return await conn.fetchval(
                """
                INSERT INTO users (igsid, credits) VALUES ($1, $2::int + $3::int)
                ON CONFLICT (igsid) DO UPDATE
                   SET credits = users.credits + $3::int, updated_at = now()
                RETURNING credits
                """,
                igsid, self._free_credits, credits,
            )

    async def apply_subscription(self, event_key: str, igsid: str, event_name: str, days: int) -> datetime | None:
        """Extends sub_until by N days from max(now, current sub_until). None if duplicate."""
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            claimed = await conn.fetchval(
                "INSERT INTO payment_events (event_key, igsid, event_name) VALUES ($1, $2, $3) "
                "ON CONFLICT (event_key) DO NOTHING RETURNING event_key",
                event_key, igsid, event_name,
            )
            if not claimed:
                return None
            return await conn.fetchval(
                """
                INSERT INTO users (igsid, credits, sub_until)
                VALUES ($1, $2::int, now() + make_interval(days => $3::int))
                ON CONFLICT (igsid) DO UPDATE
                   SET sub_until = GREATEST(COALESCE(users.sub_until, now()), now())
                                   + make_interval(days => $3::int),
                       updated_at = now()
                RETURNING sub_until
                """,
                igsid, self._free_credits, days,
            )

    # --- key/value (access-token storage) -------------------------------------

    async def kv_get(self, key: str) -> asyncpg.Record | None:
        assert self.pool
        return await self.pool.fetchrow("SELECT value, updated_at FROM app_kv WHERE key = $1", key)

    async def kv_set(self, key: str, value: str) -> None:
        assert self.pool
        await self.pool.execute(
            "INSERT INTO app_kv (key, value) VALUES ($1, $2) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
            key, value,
        )


# =============================================================================
# Instagram Messaging API client
# =============================================================================


def split_text(text: str, limit: int = IG_TEXT_LIMIT) -> list[str]:
    """Splits long text on paragraph/line boundaries to respect the IG 1000-char limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current = ""
    for para in text.split("\n"):
        candidate = f"{current}\n{para}" if current else para
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(para) > limit:
            chunks.append(para[:limit])
            para = para[limit:]
        current = para
    if current:
        chunks.append(current)
    return chunks


class InstagramClient:
    def __init__(self, http: httpx.AsyncClient, base: str, account_id: str, token: str):
        self._http = http
        self._base = base
        self._account_id = account_id
        self.token = token

    @property
    def _messages_url(self) -> str:
        return f"{self._base}/{self._account_id}/messages"

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in (1, 2):
            try:
                resp = await self._http.post(
                    self._messages_url,
                    json=payload,
                    headers={"Authorization": f"Bearer {self.token}"},
                    timeout=httpx.Timeout(10.0, connect=3.0),
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt == 1:
                    await asyncio.sleep(0.3)
                    continue
                raise GraphAPIError(0, None, None, f"transport error: {exc}") from exc

            try:
                data = resp.json()
            except ValueError:
                data = {}
            if resp.status_code < 400 and "error" not in data:
                return data

            err = data.get("error") or {}
            gerr = GraphAPIError(
                resp.status_code,
                err.get("code"),
                err.get("error_subcode"),
                err.get("message") or resp.text[:300],
            )
            # Retry only transient server-side failures.
            if resp.status_code >= 500 and attempt == 1:
                last_exc = gerr
                await asyncio.sleep(0.3)
                continue
            raise gerr
        raise GraphAPIError(0, None, None, f"unreachable: {last_exc}")

    async def send_text(self, igsid: str, text: str, quick_replies: list[dict] | None = None) -> None:
        chunks = split_text(text)
        for i, chunk in enumerate(chunks):
            message: dict[str, Any] = {"text": chunk}
            if quick_replies and i == len(chunks) - 1:
                message["quick_replies"] = quick_replies
            await self._post({"recipient": {"id": igsid}, "messaging_type": "RESPONSE", "message": message})

    async def send_action(self, igsid: str, action: str) -> None:
        await self._post({"recipient": {"id": igsid}, "sender_action": action})

    async def send_generic_template(self, igsid: str, elements: list[dict]) -> None:
        await self._post(
            {
                "recipient": {"id": igsid},
                "messaging_type": "RESPONSE",
                "message": {
                    "attachment": {
                        "type": "template",
                        "payload": {"template_type": "generic", "elements": elements},
                    }
                },
            }
        )

    async def send_private_reply(self, comment_id: str, text: str) -> None:
        await self._post({"recipient": {"comment_id": comment_id}, "message": {"text": text}})

    async def subscribe_app(self, fields: str) -> dict[str, Any]:
        """Subscribes the Instagram account to this app's webhooks (needed for Unified login)."""
        resp = await self._http.post(
            f"{self._base}/{self._account_id}/subscribed_apps",
            params={"subscribed_fields": fields},
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(10.0, connect=3.0),
        )
        try:
            data = resp.json()
        except ValueError:
            data = {}
        if resp.status_code >= 400 or "error" in data:
            err = data.get("error") or {}
            raise GraphAPIError(resp.status_code, err.get("code"), err.get("error_subcode"),
                                err.get("message") or resp.text[:300])
        return data

    async def refresh_long_lived_token(self) -> str:
        """Instagram-Login long-lived tokens live 60 days and can be refreshed."""
        resp = await self._http.get(
            "https://graph.instagram.com/refresh_access_token",
            params={"grant_type": "ig_refresh_token", "access_token": self.token},
            timeout=httpx.Timeout(10.0, connect=3.0),
        )
        data = resp.json()
        if resp.status_code >= 400 or "access_token" not in data:
            raise GraphAPIError(resp.status_code, (data.get("error") or {}).get("code"), None, str(data)[:300])
        return data["access_token"]


# =============================================================================
# Media processing: download + ffmpeg/ffprobe
# =============================================================================


@dataclass
class MediaInfo:
    path: Path
    kind: str  # "video" | "image" | "audio"
    duration: float
    has_audio: bool
    has_video: bool


async def run_cmd(args: list[str], timeout: float) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, out, err
    except asyncio.TimeoutError as exc:
        raise MediaUnavailable(MSG_UNSUPPORTED, f"{args[0]} timed out") from exc
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()


IMAGE_FORMATS = {"image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe", "tiff_pipe", "heif"}
IMAGE_CODECS = {"mjpeg", "png", "webp", "bmp", "tiff", "hevc_image"}


async def download_media(http: httpx.AsyncClient, url: str, dest: Path, max_bytes: int) -> str:
    """Streams media to disk with a hard size cap. Returns the response content-type."""

    async def _do() -> str:
        try:
            async with http.stream(
                "GET",
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ReelsFinderBot/1.0)"},
                timeout=httpx.Timeout(12.0, connect=4.0),
                follow_redirects=True,
            ) as resp:
                if resp.status_code in (401, 403, 404, 410):
                    raise MediaUnavailable(MSG_EXPIRED, f"media HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    raise MediaUnavailable(MSG_DOWNLOAD_FAILED, f"media HTTP {resp.status_code}")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype.startswith("text/") or ctype in ("application/json",):
                    raise MediaUnavailable(MSG_NOT_MEDIA, f"content-type {ctype}")
                declared = int(resp.headers.get("content-length") or 0)
                if declared and declared > max_bytes:
                    raise MediaUnavailable(MSG_TOO_BIG, f"content-length {declared}")
                size = 0
                with dest.open("wb") as fh:
                    async for chunk in resp.aiter_bytes(256 * 1024):
                        size += len(chunk)
                        if size > max_bytes:
                            raise MediaUnavailable(MSG_TOO_BIG, f"stream exceeded {max_bytes}")
                        fh.write(chunk)
                if size == 0:
                    raise MediaUnavailable(MSG_DOWNLOAD_FAILED, "empty body")
                return ctype
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise MediaUnavailable(MSG_DOWNLOAD_FAILED, f"download error: {exc}") from exc

    try:
        return await asyncio.wait_for(_do(), timeout=25)
    except asyncio.TimeoutError as exc:
        raise MediaUnavailable(MSG_DOWNLOAD_FAILED, "download timeout") from exc


async def probe_media(path: Path, content_type: str) -> MediaInfo:
    rc, out, err = await run_cmd(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration,format_name:stream=codec_type,codec_name",
            "-of", "json", str(path),
        ],
        timeout=10,
    )
    if rc != 0:
        raise MediaUnavailable(MSG_UNSUPPORTED, f"ffprobe failed: {err[:200]!r}")
    data = json.loads(out or b"{}")
    streams = data.get("streams") or []
    fmt = data.get("format") or {}
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    has_video = bool(video_streams)
    try:
        duration = float(fmt.get("duration"))
    except (TypeError, ValueError):
        duration = 0.0
    format_names = set((fmt.get("format_name") or "").split(","))

    is_image = content_type.startswith("image/") or bool(format_names & IMAGE_FORMATS) or (
        has_video and not has_audio and duration <= 0.1
        and any(s.get("codec_name") in IMAGE_CODECS for s in video_streams)
    )
    if is_image and has_video:
        kind = "image"
    elif has_video:
        kind = "video"
    elif has_audio:
        kind = "audio"
    else:
        raise MediaUnavailable(MSG_UNSUPPORTED, "no audio/video streams")
    return MediaInfo(path=path, kind=kind, duration=duration, has_audio=has_audio, has_video=has_video)


async def extract_audio_clip(info: MediaInfo, workdir: Path) -> bytes:
    """Exactly 8 s of mono 22.05 kHz 64 kbps MP3: from 0:03 (or 0:00 for short clips)."""
    if info.kind == "image":
        raise MediaUnavailable(MSG_PHOTO_NO_AUDIO, "image has no audio")
    if not info.has_audio:
        raise MediaUnavailable(MSG_NO_AUDIO, "no audio stream")
    start = 3.0 if info.duration >= 11.0 else 0.0
    out = workdir / "clip.mp3"
    rc, _, err = await run_cmd(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.2f}", "-i", str(info.path),
            "-t", "8", "-vn", "-ac", "1", "-ar", "22050", "-b:a", "64k",
            "-f", "mp3", str(out),
        ],
        timeout=15,
    )
    if rc != 0 or not out.exists() or out.stat().st_size == 0:
        raise MediaUnavailable(MSG_UNSUPPORTED, f"audio extract failed: {err[:200]!r}")
    return out.read_bytes()


async def _grab_frame(src: Path, at: float, out: Path) -> bytes | None:
    """Grabs one frame. Fast path decodes only the nearest keyframe (~3x less CPU,
    matters on 0.1-0.5 vCPU free instances); falls back to an exact seek."""
    base = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    tail = ["-frames:v", "1", "-vf", "scale=640:-1", "-q:v", "4", "-update", "1", str(out)]
    ss = ["-ss", f"{at:.2f}"] if at > 0 else []
    attempts = [
        base + ["-skip_frame", "nokey"] + ss + ["-noaccurate_seek", "-i", str(src)] + tail,
        base + ss + ["-i", str(src)] + tail,
    ]
    for args in attempts:
        out.unlink(missing_ok=True)
        rc, _, err = await run_cmd(args, timeout=15)
        if rc == 0 and out.exists() and out.stat().st_size > 0:
            return out.read_bytes()
        log.debug("frame grab at %.2fs attempt failed: %r", at, err[:200])
    log.warning("frame grab at %.2fs failed", at)
    return None


async def extract_frames(info: MediaInfo, workdir: Path) -> list[bytes]:
    """3 JPEG frames at 20/50/80% of the duration, 640px wide (1 frame for photos)."""
    if info.kind == "audio":
        raise MediaUnavailable(MSG_NO_VIDEO, "audio-only media")
    if info.kind == "image" or info.duration < 0.5:
        timestamps = [0.0]
    else:
        timestamps = [info.duration * p for p in (0.2, 0.5, 0.8)]
    results = await asyncio.gather(
        *(_grab_frame(info.path, t, workdir / f"frame_{i}.jpg") for i, t in enumerate(timestamps))
    )
    frames = [f for f in results if f]
    if not frames:
        raise MediaUnavailable(MSG_UNSUPPORTED, "no frames extracted")
    return frames


# =============================================================================
# AudD (music recognition)
# =============================================================================


class AudDClient:
    URL = "https://api.audd.io/"

    def __init__(self, http: httpx.AsyncClient, token: str, timeout: float):
        self._http = http
        self._token = token
        self._timeout = timeout

    async def recognize(self, mp3: bytes) -> dict | None:
        """Returns AudD `result` (None = not recognised). Raises ExternalServiceError."""
        for attempt in (1, 2):
            try:
                resp = await self._http.post(
                    self.URL,
                    data={"api_token": self._token, "return": "apple_music,spotify"},
                    files={"file": ("clip.mp3", mp3, "audio/mpeg")},
                    timeout=httpx.Timeout(self._timeout, connect=3.0),
                )
            except httpx.TimeoutException as exc:
                raise ExternalServiceError(f"AudD timeout: {exc}") from exc
            except httpx.TransportError as exc:
                if attempt == 1:
                    await asyncio.sleep(0.25)
                    continue
                raise ExternalServiceError(f"AudD transport error: {exc}") from exc

            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt == 1 and resp.status_code >= 500:
                    await asyncio.sleep(0.25)
                    continue
                raise ExternalServiceError(f"AudD HTTP {resp.status_code}")
            try:
                data = resp.json()
            except ValueError as exc:
                raise ExternalServiceError("AudD returned non-JSON") from exc
            if data.get("status") == "success":
                return data.get("result") or None
            err = data.get("error") or {}
            raise ExternalServiceError(f"AudD error {err.get('error_code')}: {err.get('error_message')}")
        raise ExternalServiceError("AudD unreachable")


# =============================================================================
# Gemini Vision (google-genai) with model fallback
# =============================================================================

MOVIE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "title": {"type": "string"},
        "original_title": {"type": "string"},
        "year": {"type": "string"},
        "kind": {"type": "string", "enum": ["movie", "tv_series", "anime", "documentary", "music_video", "other"]},
        "season": {"type": "string"},
        "episode": {"type": "string"},
        "actors": {"type": "array", "items": {"type": "string"}},
        "scene": {"type": "string"},
        "approx_timestamp": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["found", "title", "year", "kind", "scene", "confidence"],
}

ITEMS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "brand": {"type": "string"},
                    "model": {"type": "string"},
                    "color": {"type": "string"},
                    "description": {"type": "string"},
                    "search_query": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["category", "brand", "description", "search_query", "confidence"],
            },
        },
    },
    "required": ["found", "items"],
}

ALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"movie": MOVIE_SCHEMA, "items": ITEMS_SCHEMA},
    "required": ["movie", "items"],
}

_MOVIE_RULES = (
    "MOVIE / TV TASK: Identify the movie, TV series or anime this footage comes from. Use actors' faces, "
    "sets, costumes, cinematography, subtitles, logos and on-screen text. Social-media edits often add "
    "captions or watermarks — ignore those. If it's original creator content (vlog, skit, ad, UGC), "
    "set found=false. If you are not reasonably sure (confidence < 0.5), set found=false — never invent a title. "
    "Fill `scene` with a one-sentence description of what happens and where it sits in the story. "
    "Fill `approx_timestamp` with an approximate moment (e.g. '~01:12:00' or 'S02E05, ~00:34:00') only if you "
    "genuinely know it, otherwise an empty string. Use empty strings for unknown fields."
)
_ITEMS_RULES = (
    "FASHION / ITEMS TASK: List up to 5 of the most prominent clothing items, shoes, accessories, bags, "
    "eyewear, jewellery or notable products. For each: category, brand (only if a logo/design makes it "
    "identifiable, otherwise 'Unknown'), model name if recognisable (else empty), main color, a short "
    "description, and a precise shopping search query (brand + model + color + item type, or a descriptive "
    "query if brand is unknown). confidence is 0..1. found=false if nothing identifiable is visible."
)
PROMPTS = {
    FIND_MOVIE: ("You receive up to 3 frames (20%, 50%, 80% of the clip) from an Instagram Reel. " + _MOVIE_RULES, MOVIE_SCHEMA),
    FIND_ITEM: ("You receive up to 3 frames (20%, 50%, 80% of the clip) from an Instagram Reel or a photo. " + _ITEMS_RULES, ITEMS_SCHEMA),
    FIND_ALL: (
        "You receive up to 3 frames (20%, 50%, 80% of the clip) from an Instagram Reel. Solve BOTH tasks and "
        "return them under the keys `movie` and `items`.\n" + _MOVIE_RULES + "\n" + _ITEMS_RULES,
        ALL_SCHEMA,
    ),
}

RETRYABLE_GEMINI_CODES = {404, 408, 429, 499, 500, 502, 503, 504}


class VisionClient:
    def __init__(self, api_key: str, models: list[str], timeout_sec: float, thinking_level: str):
        self._timeout = timeout_sec
        self._models = [m for m in models if m]
        self._thinking_level = thinking_level
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                timeout=int(timeout_sec * 1000),
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )

    def _thinking_config(self, model: str) -> genai_types.ThinkingConfig | None:
        m = model.lower()
        if m.startswith("gemini-2.5-flash"):
            return genai_types.ThinkingConfig(thinking_budget=0)  # no thinking => lowest latency
        if m.startswith("gemini-3"):
            level = getattr(genai_types.ThinkingLevel, self._thinking_level, genai_types.ThinkingLevel.LOW)
            return genai_types.ThinkingConfig(thinking_level=level)
        return None

    async def analyze(self, task: str, frames: list[bytes]) -> dict:
        prompt, schema = PROMPTS[task]
        contents: list[Any] = [genai_types.Part.from_bytes(data=f, mime_type="image/jpeg") for f in frames]
        contents.append(prompt)
        last_error = "no models configured"

        for model in self._models:
            config = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema,
                temperature=0.2,
                max_output_tokens=2048,
                thinking_config=self._thinking_config(model),
            )
            try:
                resp = await asyncio.wait_for(
                    self._client.aio.models.generate_content(model=model, contents=contents, config=config),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                last_error = f"{model}: timeout {self._timeout}s"
                log.warning("Gemini %s timed out → fallback", model)
                continue
            except genai_errors.APIError as exc:
                last_error = f"{model}: HTTP {exc.code} {exc.message}"
                if exc.code in RETRYABLE_GEMINI_CODES:
                    log.warning("Gemini %s overloaded/unavailable (%s) → fallback", model, exc.code)
                    continue
                log.error("Gemini %s non-retryable error: %s", model, exc)
                raise ExternalServiceError(last_error) from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{model}: transport {exc}"
                log.warning("Gemini %s transport error → fallback: %s", model, exc)
                continue

            text = (resp.text or "").strip()
            if not text:
                last_error = f"{model}: empty/blocked response"
                log.warning("Gemini %s returned empty response → fallback", model)
                continue
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                last_error = f"{model}: invalid JSON"
                log.warning("Gemini %s returned invalid JSON → fallback", model)
                continue

        raise ExternalServiceError(f"All Gemini models failed: {last_error}")

    async def aclose(self) -> None:
        closer = getattr(self._client.aio, "aclose", None)
        if closer:
            try:
                await closer()
            except Exception:  # noqa: BLE001
                pass


# =============================================================================
# Result formatting
# =============================================================================


def _s(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def format_song(result: dict) -> str:
    artist, title = _s(result.get("artist")), _s(result.get("title"))
    album, release = _s(result.get("album")), _s(result.get("release_date"))
    query = quote_plus(f"{artist} {title}")
    spotify = ((result.get("spotify") or {}).get("external_urls") or {}).get("spotify") or (
        f"https://open.spotify.com/search/{quote_plus(f'{artist} {title}')}"
    )
    apple = (result.get("apple_music") or {}).get("url") or f"https://music.apple.com/us/search?term={query}"
    lines = ["🎵 Song found!", f"{artist} — {title}"]
    if album:
        lines.append(f"💿 {album}" + (f" ({release[:4]})" if release else ""))
    lines += [f"▶️ Spotify: {spotify}", f"🍎 Apple Music: {apple}"]
    if result.get("song_link"):
        lines.append(f"🔗 All platforms: {result['song_link']}")
    return "\n".join(lines)


def movie_is_found(data: dict) -> bool:
    try:
        conf = float(data.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    return bool(data.get("found")) and bool(_s(data.get("title"))) and conf >= 0.45


def format_movie(data: dict) -> str:
    title, year = _s(data.get("title")), _s(data.get("year"))
    original = _s(data.get("original_title"))
    kind = {"movie": "Movie", "tv_series": "TV series", "anime": "Anime", "documentary": "Documentary",
            "music_video": "Music video"}.get(_s(data.get("kind")), "")
    season, episode = _s(data.get("season")), _s(data.get("episode"))
    head = f"{title} ({year})" if year else title
    if original and original.lower() != title.lower():
        head += f" / {original}"
    lines = ["🎬 It's from:", head]
    meta = kind
    if season or episode:
        se = f"S{season}" if season else ""
        se += f"E{episode}" if episode else ""
        meta = f"{meta}, {se}" if meta else se
    if meta:
        lines.append(f"📺 {meta}")
    actors = [a for a in (data.get("actors") or []) if _s(a)][:4]
    if actors:
        lines.append(f"🎭 Cast: {', '.join(actors)}")
    if _s(data.get("scene")):
        lines.append(f"📍 Scene: {_s(data.get('scene'))}")
    if _s(data.get("approx_timestamp")):
        lines.append(f"⏱ Moment: {_s(data.get('approx_timestamp'))}")
    try:
        conf = float(data.get("confidence") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    lines.append(f"🎯 Confidence: {'high' if conf >= 0.8 else 'medium'}")
    lines.append(f"🔎 IMDb: https://www.imdb.com/find/?q={quote_plus(f'{title} {year}'.strip())}")
    return "\n".join(lines)


def filter_items(data: dict) -> list[dict]:
    items = []
    for it in data.get("items") or []:
        try:
            conf = float(it.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf >= 0.3 and _s(it.get("search_query")):
            items.append(it)
    return items[:5]


def format_items(items: list[dict]) -> str:
    lines = ["👕 Items spotted:"]
    for i, it in enumerate(items, 1):
        brand = _s(it.get("brand"))
        brand = "" if brand.lower() in ("unknown", "n/a", "none") else brand
        name = " ".join(x for x in (brand, _s(it.get("model"))) if x) or _s(it.get("category")).capitalize()
        details = ", ".join(x for x in (_s(it.get("color")), _s(it.get("category"))) if x)
        lines.append(f"{i}. {name}" + (f" — {details}" if details else ""))
        if _s(it.get("description")):
            lines.append(f"   {_s(it.get('description'))}")
        q = _s(it.get("search_query"))
        lines.append(f"   🔎 “{q}”")
        lines.append(f"   🛍 https://www.google.com/search?tbm=shop&q={quote_plus(q)}")
    return "\n".join(lines)


# =============================================================================
# Search orchestration
# =============================================================================


@dataclass
class BranchResult:
    found: bool = False
    text: str = ""
    failed: bool = False   # upstream API failure (do not charge for it)
    skipped: bool = False  # branch could not run (e.g. photo has no audio)


@dataclass
class SearchOutcome:
    branches: dict[str, BranchResult] = field(default_factory=dict)

    @property
    def any_found(self) -> bool:
        return any(b.found for b in self.branches.values())

    @property
    def any_failed(self) -> bool:
        return any(b.failed for b in self.branches.values())

    @property
    def partial(self) -> bool:
        return any(b.failed or b.skipped for b in self.branches.values())


# =============================================================================
# The bot
# =============================================================================


class Bot:
    def __init__(self, settings: Settings, http: httpx.AsyncClient, db: Database, ig: InstagramClient,
                 audd: AudDClient, vision: VisionClient):
        self.s = settings
        self.http = http
        self.db = db
        self.ig = ig
        self.audd = audd
        self.vision = vision
        self.dedup = Deduplicator(maxsize=1000)
        self.locks = UserLocks()
        self.jobs = asyncio.Semaphore(settings.max_parallel_jobs)
        self.seen_accounts: set[str] = set()

    # ---------------------------------------------------------------- helpers

    async def safe_send(self, igsid: str, text: str, quick_replies: list[dict] | None = None) -> bool:
        try:
            await self.ig.send_text(igsid, text, quick_replies)
            return True
        except GraphAPIError as exc:
            log.error("send_text to %s failed: %s", igsid, exc)
            return False

    async def typing(self, igsid: str) -> None:
        try:
            await self.ig.send_action(igsid, "typing_on")
        except GraphAPIError as exc:
            log.debug("typing_on failed: %s", exc)

    @staticmethod
    def is_subscribed(user: asyncpg.Record) -> bool:
        sub = user["sub_until"]
        return bool(sub and sub > datetime.now(timezone.utc))

    def balance_line(self, user: asyncpg.Record, credits: int | None = None) -> str:
        if self.is_subscribed(user):
            return f"⭐ Unlimited until {user['sub_until']:%b %d, %Y}"
        n = user["credits"] if credits is None else credits
        return f"💳 Credits left: {n}"

    def checkout_link(self, base: str, igsid: str) -> str:
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}{urlencode({'checkout[custom][igsid]': igsid})}"

    async def send_paywall(self, igsid: str, user: asyncpg.Record, cost: int) -> None:
        if not self.s.lemon_checkout_url:
            await self.safe_send(igsid, MSG_PAYMENTS_SOON)
            return
        credits_url = self.checkout_link(self.s.lemon_checkout_url, igsid)
        buttons = [{"type": "web_url", "url": credits_url, "title": f"Get {self.s.credits_per_pack} credits"}]
        if self.s.lemon_subscription_checkout_url:
            buttons.append({
                "type": "web_url",
                "url": self.checkout_link(self.s.lemon_subscription_checkout_url, igsid),
                "title": "Go Unlimited ⭐",
            })
        element: dict[str, Any] = {
            "title": "You're out of credits 🔋",
            # Generic Template subtitle is limited to 80 characters.
            "subtitle": (f"Needs {cost} credit{'s' if cost > 1 else ''}, you have {user['credits']}. "
                         "Top up to keep finding songs & movies.")[:80],
            "buttons": buttons,
        }
        if self.s.paywall_image_url:
            element["image_url"] = self.s.paywall_image_url
        try:
            await self.ig.send_generic_template(igsid, [element])
        except GraphAPIError as exc:
            log.warning("Generic template failed (%s); falling back to text", exc)
            await self.safe_send(
                igsid,
                f"You're out of credits 🔋 This search needs {cost}, you have {user['credits']}.\n"
                f"Get {self.s.credits_per_pack} credits here: {credits_url}",
            )

    async def deliver_pending_notice(self, igsid: str) -> None:
        try:
            notice = await self.db.pop_pending_notice(igsid)
        except Exception as exc:  # noqa: BLE001
            log.error("pop_pending_notice failed: %s", exc)
            return
        if notice and not await self.safe_send(igsid, notice):
            await self.db.set_pending_notice(igsid, notice)

    # ---------------------------------------------------------------- webhook routing

    async def handle_messaging_event(self, ev: dict, own_id: str) -> None:
        sender = _s((ev.get("sender") or {}).get("id"))
        if not sender or sender == own_id:
            return
        msg = ev.get("message")
        postback = ev.get("postback")

        if msg:
            if msg.get("is_echo") or msg.get("is_deleted") or msg.get("is_unsupported"):
                return
            await self.deliver_pending_notice(sender)

            qr_payload = (msg.get("quick_reply") or {}).get("payload")
            if qr_payload in ACTIONS:
                await self.handle_action(sender, qr_payload)
                return

            attachments = msg.get("attachments") or []
            if attachments:
                media = self.pick_media(attachments)
                if media is None:
                    await self.safe_send(sender, MSG_NO_MEDIA)
                elif media == ("", "restricted"):
                    await self.safe_send(sender, MSG_MEDIA_RESTRICTED)
                else:
                    await self.handle_media(sender, *media)
                return

            await self.handle_text(sender, _s(msg.get("text")))
            return

        if postback:
            await self.deliver_pending_notice(sender)
            payload = _s(postback.get("payload"))
            if payload in ACTIONS:
                await self.handle_action(sender, payload)
            else:
                await self.handle_text(sender, payload)

    @staticmethod
    def pick_media(attachments: list[dict]) -> tuple[str, str] | None:
        """Returns (url, type_hint), ("", "restricted") or None if nothing usable."""
        type_hint = {
            "ig_reel": "video", "reel": "video", "video": "video",
            "image": "image", "animated_image": "video", "audio": "audio",
            "ig_post": "unknown", "share": "unknown", "story_mention": "unknown",
        }
        saw_media_type = False
        for att in attachments:
            a_type = _s(att.get("type")).lower()
            if a_type not in type_hint:
                continue
            saw_media_type = True
            url = _s((att.get("payload") or {}).get("url"))
            if url.startswith("http"):
                return url, type_hint[a_type]
        return ("", "restricted") if saw_media_type else None

    async def handle_media(self, igsid: str, url: str, media_type: str) -> None:
        results = await asyncio.gather(
            self.db.set_last_media(igsid, url, media_type),
            self.ig.send_text(igsid, MSG_MENU, QUICK_REPLIES),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, Exception):
                log.error("handle_media error for %s: %s", igsid, r)
        if isinstance(results[0], Exception):
            await self.safe_send(igsid, "Oops, I couldn't save that one 😕 Please send it again in a moment.")

    async def handle_text(self, igsid: str, text: str) -> None:
        clean = text.strip()
        lowered = clean.lower().strip(" !.?")

        url_match = URL_RE.search(clean)
        if url_match:
            url = url_match.group(0).rstrip(").,!?")
            host = (urlparse(url).hostname or "").lower()
            if any(d in host for d in UNSUPPORTED_LINK_DOMAINS):
                await self.safe_send(igsid, MSG_LINK_UNSUPPORTED)
                return
            await self.handle_media(igsid, url, "unknown")
            return

        if lowered in TEXT_COMMANDS:
            await self.handle_action(igsid, TEXT_COMMANDS[lowered])
            return

        if lowered in DELETE_COMMANDS:
            await self.db.delete_user(igsid)
            log.info("User data deleted on request")
            await self.safe_send(igsid, MSG_DATA_DELETED)
            return

        user = await self.db.get_or_create_user(igsid)
        if lowered in BALANCE_COMMANDS:
            await self.safe_send(igsid, self.balance_line(user))
            return
        if lowered in BUY_COMMANDS:
            await self.send_paywall(igsid, user, 1)
            return
        await self.safe_send(igsid, welcome_text(self.s.bot_name, self.balance_line(user)))

    async def handle_comment(self, value: dict, own_id: str) -> None:
        text = _s(value.get("text"))
        from_id = _s((value.get("from") or {}).get("id"))
        comment_id = _s(value.get("id"))
        if not comment_id or not FIND_KEYWORD_RE.search(text) or from_id == own_id:
            return
        try:
            await self.ig.send_private_reply(comment_id, private_reply_text(self.s.free_credits))
            log.info("Private reply sent for comment %s", comment_id)
        except GraphAPIError as exc:
            log.warning("Private reply for comment %s failed: %s", comment_id, exc)

    # ---------------------------------------------------------------- core action

    async def handle_action(self, igsid: str, action: str) -> None:
        async with self.locks.hold(igsid):
            typing_task = asyncio.create_task(self.typing(igsid))
            try:
                user = await self.db.get_or_create_user(igsid)
                cost = ACTION_COST[action]
                subscribed = self.is_subscribed(user)

                if not user["last_media_url"]:
                    await self.safe_send(igsid, MSG_NO_MEDIA)
                    return
                if not subscribed and user["credits"] < cost:
                    await self.send_paywall(igsid, user, cost)
                    return

                try:
                    outcome = await self.run_search(action, user["last_media_url"])
                except MediaUnavailable as exc:
                    log.info("Media unavailable for %s: %s", igsid, exc)
                    await self.safe_send(igsid, exc.user_message, QUICK_REPLIES if exc.user_message in
                                         (MSG_NO_AUDIO, MSG_PHOTO_NO_AUDIO, MSG_NO_VIDEO) else None)
                    return
                except ExternalServiceError as exc:
                    log.warning("External service failure for %s: %s", igsid, exc)
                    await self.safe_send(igsid, MSG_OVERLOAD)
                    return

                if not outcome.any_found:
                    await self.safe_send(igsid, MSG_OVERLOAD if outcome.any_failed else MSG_NOT_FOUND[action])
                    return

                # Charge ONLY after a successful, non-empty answer.
                # FIND_ALL where one half failed / couldn't run is charged as a single search.
                charge = cost if not outcome.partial else 1
                footer: str
                if subscribed:
                    footer = self.balance_line(user)
                else:
                    new_balance = await self.db.charge(igsid, charge)
                    if new_balance is None:  # should not happen thanks to the per-user lock
                        log.error("Charge race for %s (cost %s)", igsid, charge)
                        new_balance = max(user["credits"] - charge, 0)
                    footer = self.balance_line(user, new_balance)

                blocks = [b.text for b in outcome.branches.values() if b.text]
                blocks.append(footer)
                blocks.append("Want something else from this Reel? 👇")
                await self.safe_send(igsid, "\n\n".join(blocks), QUICK_REPLIES)
            except Exception:  # noqa: BLE001 — never leave the user hanging
                log.exception("Unexpected error while handling %s for %s", action, igsid)
                await self.safe_send(igsid, MSG_OVERLOAD)
            finally:
                if not typing_task.done():
                    typing_task.cancel()

    async def run_search(self, action: str, media_url: str) -> SearchOutcome:
        async with self.jobs:
            workdir = Path(self.s.tmp_root) / uuid.uuid4().hex  # /tmp/{uuid}
            workdir.mkdir(parents=True, exist_ok=False)
            try:
                src = workdir / "source"
                ctype = await download_media(self.http, media_url, src, self.s.max_download_mb * 1024 * 1024)
                info = await probe_media(src, ctype)
                log.info("Media: kind=%s duration=%.1fs audio=%s ctype=%s",
                         info.kind, info.duration, info.has_audio, ctype)

                tasks: dict[str, Awaitable[BranchResult]] = {}
                if action in (FIND_MUSIC, FIND_ALL):
                    tasks["music"] = self._music_branch(info, workdir, strict=action == FIND_MUSIC)
                if action in (FIND_MOVIE, FIND_ITEM, FIND_ALL):
                    tasks["vision"] = self._vision_branch(action, info, workdir, strict=action != FIND_ALL)

                # Music (ffmpeg → AudD) and vision (ffmpeg → Gemini) run in parallel.
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                outcome = SearchOutcome()
                for name, res in zip(tasks.keys(), results, strict=True):
                    if isinstance(res, BaseException):
                        raise res  # strict-mode MediaUnavailable / ExternalServiceError
                    outcome.branches[name] = res

                if action == FIND_ALL and outcome.any_found:
                    music, vision = outcome.branches["music"], outcome.branches["vision"]
                    if not music.found and not music.text:
                        music.text = ("🎵 Song: search temporarily unavailable" if music.failed
                                      else "🎵 Song: not recognised")
                    if not vision.found and not vision.text:
                        vision.text = ("🎬 Movie & 👕 items: temporarily unavailable" if vision.failed
                                       else "🎬 Movie & 👕 items: nothing identifiable")
                return outcome
            finally:
                shutil.rmtree(workdir, ignore_errors=True)

    async def _music_branch(self, info: MediaInfo, workdir: Path, strict: bool) -> BranchResult:
        try:
            mp3 = await extract_audio_clip(info, workdir)
            result = await self.audd.recognize(mp3)
        except MediaUnavailable as exc:
            if strict:
                raise
            no_sound = exc.user_message in (MSG_NO_AUDIO, MSG_PHOTO_NO_AUDIO)
            return BranchResult(skipped=True, text="🎵 Song: no sound in this media" if no_sound else "")
        except ExternalServiceError as exc:
            if strict:
                raise
            log.warning("Music branch failed: %s", exc)
            return BranchResult(failed=True)
        if not result:
            return BranchResult(found=False)
        return BranchResult(found=True, text=format_song(result))

    async def _vision_branch(self, action: str, info: MediaInfo, workdir: Path, strict: bool) -> BranchResult:
        try:
            frames = await extract_frames(info, workdir)
            data = await self.vision.analyze(action, frames)
        except MediaUnavailable:
            if strict:
                raise
            return BranchResult(skipped=True)
        except ExternalServiceError as exc:
            if strict:
                raise
            log.warning("Vision branch failed: %s", exc)
            return BranchResult(failed=True)

        if action == FIND_MOVIE:
            return BranchResult(found=True, text=format_movie(data)) if movie_is_found(data) else BranchResult()
        if action == FIND_ITEM:
            items = filter_items(data)
            return BranchResult(found=True, text=format_items(items)) if items else BranchResult()

        # FIND_ALL
        parts: list[str] = []
        movie = data.get("movie") or {}
        if movie_is_found(movie):
            parts.append(format_movie(movie))
        items = filter_items(data.get("items") or {})
        if items:
            parts.append(format_items(items))
        return BranchResult(found=bool(parts), text="\n\n".join(parts))

    # ---------------------------------------------------------------- payments

    async def notify_payment(self, igsid: str, text: str) -> None:
        try:
            await self.ig.send_text(igsid, text)
        except GraphAPIError as exc:
            if exc.outside_messaging_window:
                log.info("User %s is outside the 24h window; notice queued", igsid)
            else:
                log.error("Payment DM to %s failed: %s; notice queued", igsid, exc)
            try:
                await self.db.set_pending_notice(igsid, text)
            except Exception as db_exc:  # noqa: BLE001
                log.error("Could not queue pending notice for %s: %s", igsid, db_exc)

    # ---------------------------------------------------------------- token refresh

    async def bootstrap_token(self) -> None:
        """Keeps a self-refreshing copy of the Instagram token in the DB."""
        seed = hashlib.sha256(self.s.ig_access_token.encode()).hexdigest()
        stored_seed = await self.db.kv_get("ig_token_seed")
        stored_token = await self.db.kv_get("ig_access_token")
        if stored_seed and stored_seed["value"] == seed and stored_token:
            self.ig.token = stored_token["value"]
        else:
            await self.db.kv_set("ig_token_seed", seed)
            await self.db.kv_set("ig_access_token", self.s.ig_access_token)
            self.ig.token = self.s.ig_access_token

    async def ensure_webhook_subscription(self) -> None:
        """Unified login (graph.facebook.com + system-user token): the dashboard toggle is not
        always enough, so the account is subscribed to the app's webhooks on every start."""
        if "graph.facebook.com" not in self.s.graph_api_base:
            return
        if not self.s.ig_account_id.isdigit():
            log.error("IG_ACCOUNT_ID must be the numeric Instagram account ID when GRAPH_API_BASE "
                      "is graph.facebook.com (Unified login); got %r", self.s.ig_account_id)
            return
        for fields in ("messages,messaging_postbacks,comments", "messages,comments"):
            try:
                await self.ig.subscribe_app(fields)
                log.info("Instagram account %s subscribed to webhooks: %s", self.s.ig_account_id, fields)
                return
            except GraphAPIError as exc:
                log.warning("Webhook subscription with fields %s failed: %s", fields, exc)
        log.error("Could not subscribe the Instagram account to webhooks — check token permissions "
                  "(instagram_basic, instagram_manage_messages, instagram_manage_comments)")

    async def token_refresh_loop(self) -> None:
        if "graph.instagram.com" not in self.s.graph_api_base:
            return  # Page tokens (Facebook Login) don't use this refresh flow
        while True:
            try:
                rec = await self.db.kv_get("ig_access_token")
                age = datetime.now(timezone.utc) - rec["updated_at"] if rec else timedelta(days=999)
                if age > timedelta(days=5):
                    new_token = await self.ig.refresh_long_lived_token()
                    await self.db.kv_set("ig_access_token", new_token)
                    self.ig.token = new_token
                    log.info("Instagram access token refreshed")
            except Exception as exc:  # noqa: BLE001
                log.warning("Token refresh skipped: %s", exc)
            await asyncio.sleep(12 * 3600)


# =============================================================================
# Privacy policy page (served at /privacy — use this URL in the Meta dashboard)
# =============================================================================

PRIVACY_EFFECTIVE_DATE = "November 21, 2026"

PRIVACY_TEMPLATE = string.Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$bot_name — Privacy Policy</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
         max-width: 760px; margin: 40px auto; padding: 0 20px; line-height: 1.6; color: #1a1a1a; }
  h1 { font-size: 28px; margin-bottom: 4px; }
  h2 { font-size: 20px; margin-top: 32px; }
  .muted { color: #666; font-size: 14px; }
  li { margin-bottom: 6px; }
  code { background: #f2f2f2; padding: 1px 5px; border-radius: 4px; }
</style>
</head>
<body>
<h1>$bot_name — Privacy Policy</h1>
<p class="muted">Effective date: $effective_date</p>

<p>$bot_name ("we", "us") is an Instagram Direct Message service operated by $operator via the
Instagram account $handle_text. You forward us a Reel, video, photo or link, choose what to find,
and we identify the song, the movie or TV show, or the clothes and items in it. This policy explains
what data we process, why, and how you can have it deleted.</p>

<h2>1. Data we collect</h2>
<ul>
  <li><b>Your Instagram-scoped ID</b> — a number Instagram assigns to you for our account only.
      We do not receive or store your name, email address or phone number.</li>
  <li><b>The link to the last media you sent us</b> and its type (video, photo or audio).</li>
  <li><b>Your search balance</b>, subscription end date (if any) and the date you first messaged us.</li>
  <li><b>Payment records</b> — which package you bought and when, if paid plans are enabled.
      We never receive or store card details; payments are handled by our payment provider.</li>
  <li><b>Comments containing the word "FIND"</b> under our posts — used only to send you a private reply.
      We do not store the comment text.</li>
  <li><b>Technical server logs</b> (for example, request times and errors), kept for a limited time
      for security and troubleshooting.</li>
</ul>

<h2>2. How we process your media</h2>
<p>When you start a search, the media is downloaded to our server temporarily. We cut an 8-second audio
clip and/or 3 still frames from it, send them for recognition, and <b>delete all files right after
processing</b>, usually within seconds. We do not keep your videos or photos.</p>

<h2>3. Who we share data with</h2>
<p>We share data only to provide the service. We do <b>not</b> sell your data or use it for advertising.</p>
<ul>
  <li><b>Meta Platforms (Instagram Messaging API)</b> — to receive and send messages.</li>
  <li><b>Google (Gemini API)</b> — still frames, to recognize movies, shows and items.</li>
  <li><b>AudD</b> — an 8-second audio clip, to recognize the song.</li>
  <li><b>Railway</b> — server hosting (United States).</li>
  <li><b>Supabase</b> — database hosting (United States).</li>
  <li><b>Our payment provider</b> — to process purchases, under its own privacy policy, if paid plans are enabled.</li>
</ul>

<h2>4. Why we process data</h2>
<p>To perform the searches you request, keep track of free and paid searches, confirm payments,
reply to your messages and comments, and keep the service secure.</p>

<h2>5. How long we keep data</h2>
<ul>
  <li>Your ID, balance and dates are kept while you use the service or until you ask us to delete them.</li>
  <li>Payment records may be kept longer where accounting or tax law requires it.</li>
  <li>Media files are not stored (see section 2).</li>
</ul>

<h2 id="delete">6. How to delete your data</h2>
<p>You can delete your data at any time, in either of these ways:</p>
<ul>
  <li>Send <code>delete my data</code> to $handle_text in Instagram Direct. Deletion happens immediately
      and you get a confirmation.</li>
  <li>Or email <a href="mailto:$email">$email</a> from any address, including your Instagram username.
      We delete the data within 30 days and confirm by reply.</li>
</ul>
<p>Deletion removes your ID, balance, subscription and media link. Remaining searches and subscription
time are lost. To prevent abuse of free searches, we keep only a one-way (irreversible) hash of your ID,
which cannot be used to identify you.</p>

<h2>7. Your rights</h2>
<p>You can ask what data we hold about you, ask us to correct or delete it, and stop using the service at
any time. If you are in the European Union or the UK, you have rights under the GDPR, including access,
rectification, erasure, restriction and objection, and the right to complain to your data protection
authority. If you are a California resident, you have rights under the CCPA, including the right to know
and to delete; we do not sell or share personal information for advertising. Contact us at
<a href="mailto:$email">$email</a> to exercise any of these rights.</p>

<h2>8. Children</h2>
<p>The service is not intended for anyone under 13 (under 16 in the EU). We do not knowingly collect
data from children. If you believe a child has used the service, contact us and we will delete the data.</p>

<h2>9. Security</h2>
<p>All connections are encrypted (HTTPS/TLS), database access is restricted, and secret keys are not
stored in our code.</p>

<h2>10. Cookies</h2>
<p>The Instagram service does not use cookies.</p>

<h2>11. Changes to this policy</h2>
<p>We will post any changes on this page and update the effective date above.</p>

<h2>12. Contact</h2>
<p>$operator — <a href="mailto:$email">$email</a></p>
</body>
</html>
""")


def render_privacy_page(s: Settings) -> str:
    handle = f"@{s.instagram_handle}" if s.instagram_handle else "our Instagram account"
    return PRIVACY_TEMPLATE.substitute(
        bot_name=html.escape(s.bot_name),
        effective_date=PRIVACY_EFFECTIVE_DATE,
        operator=html.escape(s.operator_name or s.bot_name),
        handle_text=html.escape(handle),
        email=html.escape(s.contact_email or "the email listed on our Instagram profile"),
    )


# =============================================================================
# FastAPI application
# =============================================================================

bot: Bot | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bot
    settings = Settings.from_env()
    for binary in ("ffmpeg", "ffprobe"):
        if not shutil.which(binary):
            raise RuntimeError(f"{binary} is not installed")

    http = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=3.0, pool=3.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=30),
    )
    db = Database(settings.database_url, settings.free_credits)
    await db.connect()
    ig = InstagramClient(http, settings.graph_api_base, settings.ig_account_id, settings.ig_access_token)
    audd = AudDClient(http, settings.audd_api_token, settings.audd_timeout_sec)
    vision = VisionClient(
        settings.gemini_api_key,
        [settings.gemini_primary_model, settings.gemini_fallback_model],
        settings.gemini_timeout_sec,
        settings.gemini_thinking_level,
    )
    bot = Bot(settings, http, db, ig, audd, vision)
    await bot.bootstrap_token()
    await bot.ensure_webhook_subscription()
    refresh_task = asyncio.create_task(bot.token_refresh_loop())
    log.info("Bot started (models: %s → %s)", settings.gemini_primary_model, settings.gemini_fallback_model)
    try:
        yield
    finally:
        refresh_task.cancel()
        await vision.aclose()
        await http.aclose()
        await db.close()
        bot = None


app = FastAPI(title="Reels Finder Bot", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


async def _run_safely(coro_fn: Callable[..., Awaitable[None]], *args: Any) -> None:
    try:
        await coro_fn(*args)
    except Exception:  # noqa: BLE001
        log.exception("Background task %s failed", getattr(coro_fn, "__name__", coro_fn))


def _verify_meta_signature(raw: bytes, header: str | None, secrets: list[str]) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    received = header.split("=", 1)[1]
    return any(
        hmac.compare_digest(hmac.new(s.encode(), raw, hashlib.sha256).hexdigest(), received) for s in secrets
    )


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "service": "reels-finder-bot"}


@app.get("/privacy", response_class=HTMLResponse)
async def privacy() -> HTMLResponse:
    settings = bot.s if bot else Settings.from_env()
    return HTMLResponse(render_privacy_page(settings))


@app.get("/health")
async def health() -> JSONResponse:
    ok = bool(bot and await bot.db.ping())
    return JSONResponse({"status": "ok" if ok else "degraded", "db": ok}, status_code=200 if ok else 503)


@app.get("/webhook")
async def verify_webhook(request: Request) -> Response:
    params = request.query_params
    if (
        bot
        and params.get("hub.mode") == "subscribe"
        and hmac.compare_digest(params.get("hub.verify_token", ""), bot.s.ig_verify_token)
    ):
        return PlainTextResponse(params.get("hub.challenge", ""))
    return PlainTextResponse("Verification failed", status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request, background: BackgroundTasks) -> Response:
    raw = await request.body()
    assert bot is not None
    if not _verify_meta_signature(raw, request.headers.get("x-hub-signature-256"), bot.s.ig_app_secrets):
        log.warning("Rejected webhook with invalid signature")
        return PlainTextResponse("invalid signature", status_code=401)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return PlainTextResponse("EVENT_RECEIVED")

    if payload.get("object") not in ("instagram", "page"):
        return PlainTextResponse("EVENT_RECEIVED")

    for entry in payload.get("entry") or []:
        own_id = _s(entry.get("id"))
        if own_id and own_id != bot.s.ig_account_id and own_id not in bot.seen_accounts:
            bot.seen_accounts.add(own_id)
            log.warning("Webhook is for Instagram account id=%s (IG_ACCOUNT_ID=%s). "
                        "With Unified login set IG_ACCOUNT_ID to this id.", own_id, bot.s.ig_account_id)

        for ev in entry.get("messaging") or []:
            mid = _s((ev.get("message") or {}).get("mid") or (ev.get("postback") or {}).get("mid"))
            if not mid:
                continue  # read receipts, reactions, typing, etc.
            if bot.dedup.seen(f"m:{mid}"):
                log.info("Duplicate webhook ignored: %s", mid)
                continue
            background.add_task(_run_safely, bot.handle_messaging_event, ev, own_id)

        for change in entry.get("changes") or []:
            if change.get("field") != "comments":
                continue
            value = change.get("value") or {}
            cid = _s(value.get("id"))
            if not cid or bot.dedup.seen(f"c:{cid}"):
                continue
            background.add_task(_run_safely, bot.handle_comment, value, own_id)

    return PlainTextResponse("EVENT_RECEIVED")


@app.post("/payment-webhook")
async def payment_webhook(request: Request, background: BackgroundTasks) -> Response:
    raw = await request.body()
    assert bot is not None
    if not bot.s.lemon_signing_secret:
        return JSONResponse({"error": "payments are not configured"}, status_code=503)
    signature = request.headers.get("x-signature", "")
    expected = hmac.new(bot.s.lemon_signing_secret.encode(), raw, hashlib.sha256).hexdigest()
    if not signature or not hmac.compare_digest(expected, signature):
        log.warning("Rejected Lemon Squeezy webhook: bad signature")
        return JSONResponse({"error": "invalid signature"}, status_code=401)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid json"}, status_code=400)

    meta = payload.get("meta") or {}
    event_name = _s(meta.get("event_name"))
    igsid = _s((meta.get("custom_data") or {}).get("igsid"))
    data = payload.get("data") or {}
    attrs = data.get("attributes") or {}
    event_key = f"{event_name}:{data.get('type')}:{data.get('id')}"

    if not igsid:
        log.warning("Lemon event %s without igsid custom data — ignored", event_key)
        return JSONResponse({"status": "ignored", "reason": "no igsid"})

    try:
        if event_name == "order_created":
            if _s(attrs.get("status")) != "paid":
                return JSONResponse({"status": "ignored", "reason": "order not paid"})
            variant_id = _s((attrs.get("first_order_item") or {}).get("variant_id"))
            if variant_id and variant_id in bot.s.lemon_subscription_variant_ids:
                # First subscription payment is handled by subscription_payment_success.
                return JSONResponse({"status": "ignored", "reason": "subscription order"})
            if bot.s.lemon_credit_pack_variant_ids and variant_id not in bot.s.lemon_credit_pack_variant_ids:
                log.warning("Order %s for unknown variant %s ignored", data.get("id"), variant_id)
                return JSONResponse({"status": "ignored", "reason": "unknown variant"})

            balance = await bot.db.apply_credit_pack(event_key, igsid, event_name, bot.s.credits_per_pack)
            if balance is None:
                return JSONResponse({"status": "duplicate"})
            log.info("Credit pack applied: igsid=%s balance=%s", igsid, balance)
            background.add_task(
                _run_safely, bot.notify_payment, igsid,
                MSG_PAYMENT_CREDITS.format(n=bot.s.credits_per_pack, balance=balance),
            )
            return JSONResponse({"status": "ok", "credits": balance})

        if event_name == "subscription_payment_success":
            if _s(attrs.get("status")) not in ("paid", ""):
                return JSONResponse({"status": "ignored", "reason": "invoice not paid"})
            until = await bot.db.apply_subscription(event_key, igsid, event_name, bot.s.subscription_days)
            if until is None:
                return JSONResponse({"status": "duplicate"})
            log.info("Subscription extended: igsid=%s until=%s", igsid, until.isoformat())
            if _s(attrs.get("billing_reason")) in ("initial", ""):
                background.add_task(
                    _run_safely, bot.notify_payment, igsid, MSG_PAYMENT_SUB.format(date=f"{until:%b %d, %Y}")
                )
            return JSONResponse({"status": "ok", "sub_until": until.isoformat()})

    except Exception:  # noqa: BLE001 — 500 makes Lemon Squeezy retry
        log.exception("Payment webhook processing failed for %s", event_key)
        return JSONResponse({"error": "processing failed"}, status_code=500)

    return JSONResponse({"status": "ignored", "event": event_name})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), workers=1)
