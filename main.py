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
import ipaddress
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import string
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import quote_plus, urlencode, urljoin, urlparse

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
    gemini_extra_models: str
    backup_ai_url: str
    backup_ai_key: str
    backup_ai_model: str
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
    inbox_poll_sec: int
    tmp_root: str
    allow_private_urls: bool
    # Reel downloader (yt-dlp)
    admin_code: str
    ig_cookies: str
    download_proxy: str
    # Shazam (second recogniser; better for phonk / TikTok-style remixes)
    shazam_enabled: bool
    ai_song_enabled: bool

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
            # Per-attempt timeout; vision calls always get at least 12 s (item lists are long answers).
            gemini_timeout_sec=_env_float("GEMINI_TIMEOUT_SEC", 12.0),
            # Extra Gemini models tried after primary/fallback (comma list). Unknown names are skipped.
            gemini_extra_models=_env("GEMINI_EXTRA_MODELS", ""),
            # Optional second AI provider for photo/video tasks when ALL Gemini models fail.
            # Any OpenAI-compatible API, e.g. OpenRouter: https://openrouter.ai/api/v1 + a vision model.
            backup_ai_url=_env("BACKUP_AI_URL", "https://openrouter.ai/api/v1").rstrip("/"),
            backup_ai_key=_env("BACKUP_AI_KEY", ""),
            backup_ai_model=_env("BACKUP_AI_MODEL", ""),
            gemini_thinking_level=_env("GEMINI_THINKING_LEVEL", "minimal").upper(),
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
            # Picks up DMs stuck in "Message requests" (Meta may not send webhooks for them). 0 = off.
            inbox_poll_sec=_env_int("INBOX_POLL_SEC", 10),
            tmp_root=_env("TMP_ROOT", "/tmp"),
            # Only for local tests: lets the bot fetch media from localhost / private networks.
            allow_private_urls=_env("ALLOW_PRIVATE_URLS", "").lower() in ("1", "true", "yes"),
            # Optional: cookies.txt of a spare Instagram account (raw or base64) and a proxy
            # (http://user:pass@host:port) — only needed if Instagram starts blocking the server.
            # Secret for the DM command "admin <code> <credits>" (adds test credits to yourself).
            # Empty = admin command disabled. Use 16+ random characters.
            admin_code=_env("ADMIN_CODE", ""),
            ig_cookies=_env("IG_COOKIES", ""),
            download_proxy=_env("DOWNLOAD_PROXY", ""),
            shazam_enabled=_env("SHAZAM_ENABLED", "1").lower() not in ("0", "false", "no"),
            # Gemini listens to the audio when fingerprinting fails (remixes, covers, edits).
            ai_song_enabled=_env("AI_SONG_ENABLED", "1").lower() not in ("0", "false", "no"),
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
PROMO_RE = re.compile(r"[A-Z0-9][A-Z0-9_-]{2,23}")
DELETE_COMMANDS = {"delete my data", "delete data", "delete my account", "remove my data"}
MSG_DATA_DELETED = (
    "Done ✅ Your data has been deleted: your ID, balance, subscription and last media link.\n"
    "If you message me again, a new empty profile will be created (free searches are not renewed)."
)

MSG_OVERLOAD = "Temporary overload, your credit was not deducted. Please try again in a moment."
MSG_MENU = "Got it! 🔥 What should I find in this one?\n(On desktop? Just reply: song, movie, outfit or all)"
MSG_NO_MEDIA = "Send me a Reel, video or photo first 🙂\nTap ✈️ under any Reel → send it to me."
MSG_LINK_UNSUPPORTED = (
    "I can't open TikTok / YouTube links yet 🙈\n"
    "Send me an Instagram Reel (tap ✈️ under it → send to me) or the video file itself."
)
MSG_REEL_BLOCKED = (
    "Instagram didn't let me open this Reel right now 😕 (it may be private or restricted).\n"
    "Quick fix: screen-record it with sound (or save the video) and send me the file. "
    "For movie/outfit search a screenshot works too.\nNo credit was used."
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


# Instagram post / Reel page links (Meta sends these instead of a video file for shared Reels).
IG_POST_RE = re.compile(
    r"^https?://(?:www\.|m\.)?instagram\.com/(?:[A-Za-z0-9_.]+/)?(reels?|p|tv)/([A-Za-z0-9_-]{5,})",
    re.IGNORECASE,
)
# Domains whose *page* links we cannot fetch.
UNSUPPORTED_LINK_DOMAINS = (
    "instagram.com", "instagr.am", "tiktok.com", "youtube.com", "youtu.be",
    "facebook.com", "fb.watch", "x.com", "twitter.com", "pinterest.",
)
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
FIND_KEYWORD_RE = re.compile(r"\bfind\b", re.IGNORECASE)

BUILD = "2026-11-22 promo"  # shown in the startup log to confirm which code is deployed
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
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_media_mid  TEXT;
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

CREATE TABLE IF NOT EXISTS promo_codes (
    code        TEXT PRIMARY KEY,
    credits     INT  NOT NULL,
    max_uses    INT  NOT NULL,
    used        INT  NOT NULL DEFAULT 0,
    expires_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS promo_redemptions (
    code        TEXT NOT NULL,
    id_hash     TEXT NOT NULL,          -- hashed user id: survives "delete my data", no plain id kept
    redeemed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (code, id_hash)
);

-- Supabase exposes the public schema through its REST "Data API". Row Level Security with no
-- policies closes that door; the bot connects as the table owner and is not affected.
ALTER TABLE users          ENABLE ROW LEVEL SECURITY;
ALTER TABLE payment_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE deleted_users  ENABLE ROW LEVEL SECURITY;
ALTER TABLE app_kv         ENABLE ROW LEVEL SECURITY;
ALTER TABLE promo_codes    ENABLE ROW LEVEL SECURITY;
ALTER TABLE promo_redemptions ENABLE ROW LEVEL SECURITY;
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

    async def set_last_media(self, igsid: str, url: str, media_type: str, mid: str = "") -> None:
        assert self.pool
        await self.pool.execute(
            """
            INSERT INTO users (igsid, credits, last_media_url, last_media_type, last_media_mid)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (igsid) DO UPDATE
               SET last_media_url = EXCLUDED.last_media_url,
                   last_media_type = EXCLUDED.last_media_type,
                   last_media_mid = EXCLUDED.last_media_mid,
                   updated_at = now()
            """,
            igsid,
            await self._starting_credits(igsid),
            url,
            media_type,
            mid or None,
        )

    async def create_promo(self, code: str, credits: int, max_uses: int, days: int | None) -> bool:
        assert self.pool
        res = await self.pool.execute(
            """INSERT INTO promo_codes (code, credits, max_uses, expires_at)
               VALUES ($1, $2::int, $3::int, CASE WHEN $4::int IS NULL THEN NULL
                                             ELSE now() + make_interval(days => $4::int) END)
               ON CONFLICT (code) DO NOTHING""", code, credits, max_uses, days)
        return res.endswith("1")

    async def list_promos(self) -> list[asyncpg.Record]:
        assert self.pool
        return await self.pool.fetch(
            """SELECT * FROM promo_codes WHERE used < max_uses AND (expires_at IS NULL OR expires_at > now())
               ORDER BY created_at DESC LIMIT 20""")

    async def disable_promo(self, code: str) -> bool:
        assert self.pool
        res = await self.pool.execute("UPDATE promo_codes SET max_uses = used WHERE code = $1", code)
        return res.endswith("1")

    async def redeem_promo(self, igsid: str, code: str) -> tuple[str, int, int]:
        """→ (status, credits_added, new_balance). status: ok | unknown | used | expired."""
        assert self.pool
        async with self.pool.acquire() as conn, conn.transaction():
            promo = await conn.fetchrow("SELECT * FROM promo_codes WHERE code = $1 FOR UPDATE", code)
            if not promo:
                return "unknown", 0, 0
            if promo["used"] >= promo["max_uses"] or (
                    promo["expires_at"] and promo["expires_at"] <= datetime.now(timezone.utc)):
                return "expired", 0, 0
            inserted = await conn.execute(
                "INSERT INTO promo_redemptions (code, id_hash) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                code, self._id_hash(igsid))
            if not inserted.endswith("1"):
                return "used", 0, 0
            await conn.execute("UPDATE promo_codes SET used = used + 1 WHERE code = $1", code)
            balance = await conn.fetchval(
                "UPDATE users SET credits = credits + $2::int, updated_at = now() WHERE igsid = $1 RETURNING credits",
                igsid, promo["credits"])
            return "ok", promo["credits"], balance or 0

    async def add_credits(self, igsid: str, amount: int) -> int:
        assert self.pool
        return await self.pool.fetchval(
            "UPDATE users SET credits = credits + $2::int, updated_at = now() WHERE igsid = $1 RETURNING credits",
            igsid, amount)

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


def extract_graph_media(message: dict[str, Any]) -> list[str]:
    """Media URLs from a Graph API message object (attachments + shares)."""
    urls: list[str] = []
    for a in (message.get("attachments") or {}).get("data") or []:
        for key in ("video_data", "image_data"):
            u = _s((a.get(key) or {}).get("url"))
            if u:
                urls.append(u)
        if _s(a.get("file_url")):
            urls.append(_s(a.get("file_url")))
    for sh in (message.get("shares") or {}).get("data") or []:
        for key in ("url", "link"):
            if _s(sh.get(key)):
                urls.append(_s(sh.get(key)))
    seen: set[str] = set()
    return [u for u in urls if u.startswith("http") and not (u in seen or seen.add(u))]


def graph_message_to_event(message: dict[str, Any], own_id: str) -> dict[str, Any] | None:
    """Turns a Graph API message into the same shape as a messaging webhook event."""
    sender = _s((message.get("from") or {}).get("id"))
    mid = _s(message.get("id"))
    if not sender or not mid:
        return None
    attachments: list[dict[str, Any]] = []
    for a in (message.get("attachments") or {}).get("data") or []:
        if _s((a.get("video_data") or {}).get("url")):
            attachments.append({"type": "video", "payload": {"url": a["video_data"]["url"]}})
        elif _s((a.get("image_data") or {}).get("url")):
            attachments.append({"type": "image", "payload": {"url": a["image_data"]["url"]}})
        elif _s(a.get("file_url")):
            attachments.append({"type": "share", "payload": {"url": a["file_url"]}})
    for sh in (message.get("shares") or {}).get("data") or []:
        url = _s(sh.get("url")) or _s(sh.get("link"))
        kind = _s(sh.get("type")).lower()
        if url:
            attachments.append({"type": kind if kind in ("ig_reel", "reel", "ig_post") else "share",
                                "payload": {"url": url}})
    msg: dict[str, Any] = {"mid": mid}
    if _s(message.get("message")):
        msg["text"] = _s(message.get("message"))
    if attachments:
        msg["attachments"] = attachments
    if "text" not in msg and "attachments" not in msg:
        return None
    return {"sender": {"id": sender}, "recipient": {"id": own_id}, "message": msg}


def parse_graph_time(value: str) -> datetime | None:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
    except (TypeError, ValueError):
        return None


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

    async def _get(self, path: str, params: dict[str, str]) -> dict[str, Any]:
        resp = await self._http.get(
            f"{self._base}/{path}",
            params=params,
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

    async def get_message_media_urls(self, mid: str) -> list[str]:
        """Alternative media URLs for a message, read through the Graph API."""
        for fields in ("attachments{video_data,image_data,file_url},shares{link,url,type,name}",
                       "attachments,shares"):
            try:
                data = await self._get(mid, {"fields": fields})
                break
            except GraphAPIError as exc:
                log.warning("Message lookup with fields %s failed: %s", fields, exc)
        else:
            return []
        log.info("Message lookup keys: %s", sorted(data.keys()))
        return extract_graph_media(data)

    async def list_recent_messages(self) -> list[dict[str, Any]]:
        """Recent messages across conversations, including the Requests folder."""
        field_sets = (
            "updated_time,messages.limit(5){id,created_time,from,message,"
            "attachments{video_data,image_data,file_url},shares{link,url,type,name}}",
            "updated_time,messages.limit(5){id,created_time,from,message,attachments,shares}",
        )
        last_exc: GraphAPIError | None = None
        for fields in field_sets:
            try:
                data = await self._get(f"{self._account_id}/conversations",
                                       {"platform": "instagram", "fields": fields, "limit": "10"})
                break
            except GraphAPIError as exc:
                last_exc = exc
                if exc.code != 100:  # 100 = invalid parameter/field: try the simpler field set
                    raise
        else:
            raise last_exc or GraphAPIError(0, None, None, "conversations unavailable")
        out: list[dict[str, Any]] = []
        for conv in data.get("data") or []:
            out.extend((conv.get("messages") or {}).get("data") or [])
        return out

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


async def assert_public_url(url: str) -> None:
    """Blocks SSRF: the bot must never fetch localhost, private networks or cloud metadata IPs."""
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise MediaUnavailable(MSG_NOT_MEDIA, f"bad url scheme/host: {parts.scheme}")
    if parts.port not in (None, 80, 443):
        raise MediaUnavailable(MSG_NOT_MEDIA, f"non-standard port {parts.port}")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(parts.hostname, None)
    except OSError as exc:
        raise MediaUnavailable(MSG_DOWNLOAD_FAILED, f"dns error: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if not ip.is_global:
            raise MediaUnavailable(MSG_NOT_MEDIA, f"blocked non-public address {ip}")


REDIRECT_CODES = {301, 302, 303, 307, 308}

# Tried in order when a media URL answers with an HTML page instead of a file.
DOWNLOAD_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36",
    "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
)


def instagram_post_url(url: str) -> str | None:
    """Canonical https://www.instagram.com/<reel|p|tv>/<code>/ for an Instagram post link, else None."""
    m = IG_POST_RE.match(url.strip())
    if not m:
        return None
    kind = "reel" if m.group(1).lower().startswith("reel") else m.group(1).lower()
    return f"https://www.instagram.com/{kind}/{m.group(2)}/"


@dataclass
class ResolvedPost:
    video_url: str = ""   # progressive mp4 (video + audio)
    dash_video: str = ""  # used only if no progressive format exists
    dash_audio: str = ""
    thumbnail: str = ""
    caption: str = ""
    music_title: str = ""    # Instagram's own audio label
    music_artist: str = ""
    music_original: bool = False  # True = "original audio" (user-uploaded), False = licensed track


def pick_post_formats(info: dict) -> ResolvedPost:
    """Chooses the lightest usable formats from yt-dlp's info dict (no network)."""
    if info.get("_type") == "playlist":  # carousel: first entry that has a video, else the first one
        entries = [e for e in (info.get("entries") or []) if e]
        with_video = [e for e in entries if e.get("formats") or e.get("url")]
        parent = info
        info = dict((with_video or entries or [{}])[0])
        for key in ("description", "ig_music"):
            info.setdefault(key, parent.get(key))
    music = info.get("ig_music") or {}
    out = ResolvedPost(thumbnail=_s(info.get("thumbnail")), caption=_s(info.get("description"))[:600],
                       music_title=_s(music.get("title")), music_artist=_s(music.get("artist")),
                       music_original=bool(music.get("original")))
    formats = [f for f in (info.get("formats") or []) if _s(f.get("url")).startswith("http")]
    if not formats and _s(info.get("url")).startswith("http"):
        formats = [info]

    def has(f: dict, kind: str) -> bool:
        return _s(f.get(kind)) not in ("none",)  # missing codec info = unknown = assume present

    def height(f: dict) -> int:
        return int(f.get("height") or 0)

    def by_size(f: dict) -> tuple:  # prefer ~480-720p: enough for vision, small for speed
        h = height(f)
        return (0 if 400 <= h <= 1080 else 1, abs(h - 720) if h else 9999)

    progressive = [f for f in formats if has(f, "vcodec") and has(f, "acodec")
                   and "dash" not in _s(f.get("format_id")).lower()]
    if progressive:
        out.video_url = sorted(progressive, key=by_size)[0]["url"]
        return out
    videos = [f for f in formats if has(f, "vcodec") and _s(f.get("acodec")) == "none"]
    audios = [f for f in formats if has(f, "acodec") and _s(f.get("vcodec")) == "none"]
    if videos:
        out.dash_video = sorted(videos, key=by_size)[0]["url"]
    if audios:
        out.dash_audio = sorted(audios, key=lambda f: float(f.get("abr") or f.get("tbr") or 0))[-1]["url"]
    if not videos and not audios and formats:
        out.video_url = formats[-1]["url"]
    return out


def ig_music_from_product(product: dict) -> dict:
    """Instagram's audio label from the raw media JSON: licensed track or 'original audio'."""
    clips = (product or {}).get("clips_metadata") or {}
    asset = (((clips.get("music_info") or {}).get("music_asset_info"))
             or (((product or {}).get("music_metadata") or {}).get("music_info") or {}).get("music_asset_info")
             or {})
    if _s(asset.get("title")):
        return {"title": _s(asset.get("title")), "artist": _s(asset.get("display_artist")), "original": False}
    sound = clips.get("original_sound_info") or {}
    if _s(sound.get("original_audio_title")):
        return {"title": _s(sound.get("original_audio_title")),
                "artist": _s((sound.get("ig_artist") or {}).get("username")), "original": True}
    return {}


_YTDL_HOOKED = False


def _hook_ytdl_instagram() -> None:
    """yt-dlp drops Instagram's music label; this keeps it as info['ig_music']."""
    global _YTDL_HOOKED
    if _YTDL_HOOKED:
        return
    _YTDL_HOOKED = True
    try:
        from yt_dlp.extractor.instagram import InstagramBaseIE

        original = InstagramBaseIE._extract_product

        def _extract_product(self, product_info, *args, **kwargs):  # noqa: ANN001
            info = original(self, product_info, *args, **kwargs)
            try:
                raw = product_info[0] if isinstance(product_info, list) else product_info
                if isinstance(info, dict) and isinstance(raw, dict):
                    info["ig_music"] = ig_music_from_product(raw)
            except Exception:  # noqa: BLE001
                pass
            return info

        InstagramBaseIE._extract_product = _extract_product
    except Exception as exc:  # noqa: BLE001 — never break downloads over metadata
        log.warning("Could not hook yt-dlp Instagram music metadata: %s", exc)


class _QuietYtdlLogger:
    def debug(self, msg: str) -> None: ...
    def info(self, msg: str) -> None: ...
    def warning(self, msg: str) -> None: ...
    def error(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)


class ReelExtractor:
    """Resolves an Instagram Reel/post page into direct CDN media URLs with yt-dlp.
    Results are cached per post for 10 minutes, so a second button tap is instant."""

    def __init__(self, cookies: str = "", proxy: str = "", tmp_root: str = "/tmp"):
        self.proxy = proxy
        self.cookiefile = self._write_cookies(cookies, tmp_root) if cookies else ""
        self.cache: TTLCache = TTLCache(maxsize=500, ttl=600)
        self.inflight: dict[str, asyncio.Task] = {}

    @staticmethod
    def _write_cookies(raw: str, tmp_root: str) -> str:
        text = raw.strip()
        if "\t" not in text:
            try:
                import base64
                text = base64.b64decode(text).decode("utf-8")
            except Exception:  # noqa: BLE001
                log.warning("IG_COOKIES is neither a cookies.txt nor base64 of one — ignored")
                return ""
        if not text.startswith("# Netscape"):
            text = "# Netscape HTTP Cookie File\n" + text
        path = Path(tmp_root) / f"igcookies-{uuid.uuid4().hex}.txt"
        path.write_text(text + "\n")
        path.chmod(0o600)
        log.info("Instagram cookies loaded for the Reel downloader")
        return str(path)

    def _extract_sync(self, url: str) -> dict:
        import yt_dlp  # imported lazily: heavy module

        _hook_ytdl_instagram()

        opts: dict[str, Any] = {
            "quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": False,
            "socket_timeout": 10, "extractor_retries": 1, "logger": _QuietYtdlLogger(),
            "cachedir": False,
        }
        if self.cookiefile:
            opts["cookiefile"] = self.cookiefile
        if self.proxy:
            opts["proxy"] = self.proxy
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False) or {}

    async def _resolve_uncached(self, url: str) -> ResolvedPost:
        started = asyncio.get_running_loop().time()
        try:
            info = await asyncio.wait_for(asyncio.to_thread(self._extract_sync, url), timeout=20)
        except asyncio.TimeoutError as exc:
            raise MediaUnavailable(MSG_REEL_BLOCKED, "reel extractor timeout") from exc
        except Exception as exc:  # noqa: BLE001 — yt-dlp raises many types
            reason = re.sub(r"\s+", " ", str(exc))[:300]
            raise MediaUnavailable(MSG_REEL_BLOCKED, f"reel extractor failed: {reason}") from exc
        post = pick_post_formats(info)
        log.info("Reel resolved in %.1fs: progressive=%s dash_video=%s dash_audio=%s thumb=%s "
                 "audio_label=%r (%s)",
                 asyncio.get_running_loop().time() - started, bool(post.video_url),
                 bool(post.dash_video), bool(post.dash_audio), bool(post.thumbnail),
                 f"{post.music_artist} — {post.music_title}"[:80] if post.music_title else "",
                 "original" if post.music_original else "licensed")
        return post

    async def resolve(self, url: str) -> ResolvedPost:
        if url in self.cache:
            return self.cache[url]
        task = self.inflight.get(url)
        if task is None:  # several taps / prefetch share one extraction
            task = asyncio.create_task(self._resolve_uncached(url))
            self.inflight[url] = task
            task.add_done_callback(lambda _t, u=url: self.inflight.pop(u, None))
        post = await asyncio.shield(task)
        self.cache[url] = post
        return post

    def cached(self, url: str) -> ResolvedPost | None:
        ig_url = instagram_post_url(url)
        return self.cache.get(ig_url) if ig_url else None

    def prefetch(self, url: str) -> None:
        """Starts resolving while the user is still choosing a button."""
        async def _run() -> None:
            try:
                await self.resolve(url)
            except Exception as exc:  # noqa: BLE001
                log.info("Reel prefetch failed: %s", exc)
        if url not in self.cache and url not in self.inflight:
            asyncio.create_task(_run())


OG_IMAGE_RE = re.compile(
    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']'
    r'|<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
    re.IGNORECASE,
)


async def fetch_og_image(http: httpx.AsyncClient, page_url: str) -> str:
    """Cover image from the post page's Open Graph tags (what link previews use). '' if none."""
    try:
        async with http.stream("GET", page_url, headers={"User-Agent": DOWNLOAD_USER_AGENTS[-1]},
                               timeout=httpx.Timeout(8.0, connect=4.0), follow_redirects=False) as resp:
            if resp.status_code != 200:
                return ""
            body = b""
            async for chunk in resp.aiter_bytes(64 * 1024):
                body += chunk
                if len(body) > 512 * 1024 or b"og:image" in body and b"</head>" in body:
                    break
    except httpx.HTTPError:
        return ""
    m = OG_IMAGE_RE.search(body.decode("utf-8", "replace"))
    return html.unescape(m.group(1) or m.group(2)) if m else ""


def describe_url(url: str) -> str:
    """host + shortened path, without the query string (which carries CDN signatures)."""
    parts = urlparse(url)
    path = parts.path if len(parts.path) <= 60 else parts.path[:60] + "…"
    return f"{parts.hostname}{path}"


async def download_media(http: httpx.AsyncClient, url: str, dest: Path, max_bytes: int,
                         allow_private: bool = False) -> str:
    """Streams media to disk with a hard size cap. Returns the response content-type.
    Redirects are followed manually so every hop is checked against SSRF."""

    async def _attempt(user_agent: str) -> str | None:
        """Returns content-type on success, None if the server answered with an HTML page."""
        current = url
        for _hop in range(5):
            if not allow_private:
                await assert_public_url(current)
            async with http.stream(
                "GET",
                current,
                headers={"User-Agent": user_agent, "Accept": "video/*,image/*,audio/*,*/*;q=0.8"},
                timeout=httpx.Timeout(12.0, connect=4.0),
                follow_redirects=False,
            ) as resp:
                if resp.status_code in REDIRECT_CODES and resp.headers.get("location"):
                    current = urljoin(current, resp.headers["location"])
                    continue
                if resp.status_code in (401, 403, 404, 410):
                    raise MediaUnavailable(MSG_EXPIRED, f"media HTTP {resp.status_code} at {describe_url(current)}")
                if resp.status_code >= 400:
                    raise MediaUnavailable(MSG_DOWNLOAD_FAILED, f"media HTTP {resp.status_code} at {describe_url(current)}")
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype.startswith("text/") or ctype in ("application/json",):
                    head = b""
                    async for chunk in resp.aiter_bytes(2048):
                        head = chunk[:300]
                        break
                    snippet = re.sub(r"\s+", " ", head.decode("utf-8", "replace"))[:200]
                    log.warning("Media URL returned %s (UA=%s…) at %s; first bytes: %r",
                                ctype, user_agent[:20], describe_url(current), snippet)
                    return None
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
        raise MediaUnavailable(MSG_DOWNLOAD_FAILED, "too many redirects")

    async def _do() -> str:
        try:
            for user_agent in DOWNLOAD_USER_AGENTS:
                ctype = await _attempt(user_agent)
                if ctype is not None:
                    return ctype
            raise MediaUnavailable(MSG_NOT_MEDIA, f"html page instead of media at {describe_url(url)}")
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


async def extract_wav_clip(info: MediaInfo, workdir: Path, rate: float = 1.0) -> bytes:
    """12 s mono 16 kHz WAV for Shazam. rate != 1 plays it faster/slower WITH pitch change:
    that undoes "sped up" / "slowed" edits, so the original track can be matched."""
    start = 1.0 if info.duration >= 14.0 else 0.0
    out = workdir / f"shz_{rate:.2f}_{uuid.uuid4().hex[:6]}.wav"
    af = ["-af", f"aresample=44100,asetrate={44100 * rate:.0f},aresample=16000"] if rate != 1.0 else []
    rc, _, err = await run_cmd(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start:.2f}", "-i", str(info.path), "-vn", *af,
         "-ac", "1", "-ar", "16000", "-t", "12", "-f", "wav", str(out)],
        timeout=15,
    )
    if rc != 0 or not out.exists() or out.stat().st_size < 1000:
        raise MediaUnavailable(MSG_UNSUPPORTED, f"wav extract failed: {err[:200]!r}")
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


def _norm(text: str) -> str:
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text.lower())  # drop "(feat. X)", "[Remix]"
    text = re.sub(r"\b(feat|ft|featuring|prod|remix|edit|version|sped up|slowed|reverb)\b.*", " ", text)
    return re.sub(r"[^\w]+", " ", text).strip()


def _similar(a: str, b: str) -> float:
    from difflib import SequenceMatcher

    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


async def verify_song_itunes(http: httpx.AsyncClient, artist: str, title: str) -> dict | None:
    """Checks that the AI's answer is a real released track (free iTunes Search API, no key).
    Returns an AudD-shaped result with canonical names + Apple Music link, or None."""
    try:
        resp = await http.get("https://itunes.apple.com/search",
                              params={"term": f"{artist} {title}"[:200], "entity": "song", "limit": 15},
                              timeout=httpx.Timeout(5.0, connect=3.0))
        items = resp.json().get("results") or [] if resp.status_code == 200 else []
    except (httpx.HTTPError, ValueError) as exc:
        log.info("iTunes verification unavailable: %s", exc)
        return None
    best, best_score = None, 0.0
    for it in items:
        t_score = _similar(title, _s(it.get("trackName")))
        a_score = _similar(artist, _s(it.get("artistName")))
        score = t_score * 0.6 + a_score * 0.4
        if t_score >= 0.75 and a_score >= 0.6 and score > best_score:
            best, best_score = it, score
    if not best:
        return None
    return {
        "artist": _s(best.get("artistName")), "title": _s(best.get("trackName")),
        "album": _s(best.get("collectionName")), "release_date": _s(best.get("releaseDate"))[:10],
        "apple_music": {"url": _s(best.get("trackViewUrl")).split("?")[0]},
        "_cover": _s(best.get("artworkUrl100")).replace("100x100", "600x600"),
    }


def shazam_to_result(track: dict) -> dict:
    """Shazam `track` → the same shape format_song() expects from AudD."""
    result: dict[str, Any] = {"title": _s(track.get("title")), "artist": _s(track.get("subtitle"))}
    for section in track.get("sections") or []:
        for meta in (section or {}).get("metadata") or []:
            label = _s((meta or {}).get("title")).lower()
            if label == "album":
                result["album"] = _s(meta.get("text"))
            elif label == "released":
                result["release_date"] = _s(meta.get("text"))
    if _s(track.get("url")).startswith("https://"):
        result["song_link"] = track["url"]
    cover = _s((track.get("images") or {}).get("coverart"))
    if cover.startswith("https://"):
        result["_cover"] = cover
    return result


class ShazamClient:
    """Shazam via the open-source shazamio library (unofficial API, free). Never raises:
    any problem simply counts as "not recognised", AudD stays the primary engine."""

    def __init__(self, timeout: float = 8.0):
        from shazamio import Shazam  # lazy: optional dependency

        self._shazam = Shazam(segment_duration_seconds=12)
        self._timeout = timeout

    async def recognize(self, wav: bytes) -> dict | None:
        try:
            out = await asyncio.wait_for(self._shazam.recognize(wav), timeout=self._timeout)
        except Exception as exc:  # noqa: BLE001
            log.warning("Shazam failed: %s", re.sub(r"\s+", " ", str(exc))[:200] or type(exc).__name__)
            return None
        track = (out or {}).get("track")
        if not track or not _s(track.get("title")):
            return None
        return shazam_to_result(track)


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

SONG_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "artist": {"type": "string", "description": "Artist of the ORIGINAL released track"},
        "title": {"type": "string", "description": "Title of the ORIGINAL released track"},
        "version": {"type": "string", "description": "e.g. 'original', 'sped up', 'slowed', 'phonk remix', "
                                                     "'cover', 'mashup'; empty if unsure"},
        "confidence": {"type": "number", "description": "0..1 — how sure you are about artist+title"},
        "evidence": {"type": "string", "description": "short: melody / lyrics heard / caption / audio label"},
    },
    "required": ["found", "artist", "title", "confidence"],
}


def song_prompt(caption: str, audio_label: str) -> str:
    ctx = []
    if audio_label:
        ctx.append(f"Instagram audio label: {audio_label}")
    if caption:
        ctx.append(f"Reel caption: {caption}")
    context = "\n".join(ctx) or "(no caption or audio label available)"
    return (
        "You are a music identification expert. Identify the song playing in this audio clip from an "
        "Instagram Reel. Audio fingerprinting (Shazam) already failed, so it is probably a remix, "
        "sped-up/slowed edit, phonk/funk edit, cover or mashup — identify the ORIGINAL track it is based on "
        "(recognise the melody, hook, vocals, lyrics or sample).\n"
        "Context (often names the track or credits it with 🎵, 'song:', '#songname'; may also be unrelated):\n"
        f"{context}\n\n"
        "Rules: never invent a song. If you can't name a real released track, return found=false. "
        "confidence must reflect real certainty (0.9 = you clearly recognise it)."
    )


class _ModelFailed(Exception):
    pass


class VisionClient:
    TOTAL_BUDGET_SEC = 25.0  # whole fallback chain must fit in this
    HEDGE_SEC = 5.0          # no answer after this → start the next model in parallel

    def __init__(self, api_key: str, models: list[str], timeout_sec: float, thinking_level: str,
                 backup: tuple[str, str, str] | None = None):
        self._timeout = max(timeout_sec, 12.0)
        self._models = list(dict.fromkeys(m.strip() for m in models if m and m.strip()))  # dedupe, keep order
        self._dead_models: set[str] = set()  # 404 "model not found" → don't try again
        self._thinking_level = thinking_level
        self._backup = backup if backup and all(backup) else None
        self._backup_http: httpx.AsyncClient | None = None
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                timeout=int(self._timeout * 1000),
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )

    @property
    def describe(self) -> str:
        chain = " → ".join(self._models)
        return chain + (f" → backup {self._backup[2]}" if self._backup else "")

    def _thinking_config(self, model: str) -> genai_types.ThinkingConfig | None:
        m = model.lower()
        if m.startswith("gemini-2.5-flash"):
            return genai_types.ThinkingConfig(thinking_budget=0)  # no thinking => lowest latency
        if m.startswith("gemini-3"):
            level = getattr(genai_types.ThinkingLevel, self._thinking_level, genai_types.ThinkingLevel.MINIMAL)
            return genai_types.ThinkingConfig(thinking_level=level)
        return None

    async def analyze(self, task: str, frames: list[bytes]) -> dict:
        prompt, schema = PROMPTS[task]
        contents: list[Any] = [genai_types.Part.from_bytes(data=f, mime_type="image/jpeg") for f in frames]
        contents.append(prompt)
        try:
            return await self._generate(contents, schema)
        except ExternalServiceError as exc:
            if not self._backup:
                raise
            log.warning("%s → trying backup AI %s", exc, self._backup[2])
            return await self._backup_generate(frames, prompt, schema, str(exc))

    async def identify_song(self, wav: bytes, caption: str, audio_label: str) -> dict:
        contents: list[Any] = [genai_types.Part.from_bytes(data=wav, mime_type="audio/wav"),
                               song_prompt(caption, audio_label)]
        return await self._generate(contents, SONG_SCHEMA)

    async def _call_model(self, model: str, contents: list[Any], schema: dict, timeout: float) -> dict:
        """One Gemini attempt. Returns parsed JSON or raises _ModelFailed(reason)."""
        started = asyncio.get_running_loop().time()
        tuned = True  # minimal thinking + medium image detail = much faster; plain call if rejected
        for attempt in (1, 2):
            extra: dict[str, Any] = {}
            if tuned:
                extra = {"thinking_config": self._thinking_config(model),
                         "media_resolution": genai_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM}
            config = genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=schema,
                temperature=0.2,
                max_output_tokens=2048,
                **extra,
            )
            try:
                resp = await asyncio.wait_for(
                    self._client.aio.models.generate_content(model=model, contents=contents, config=config),
                    timeout=timeout,
                )
            except asyncio.TimeoutError as exc:
                raise _ModelFailed(f"{model}: timeout {timeout:.0f}s") from exc
            except genai_errors.APIError as exc:
                msg = _s(exc.message)
                if exc.code == 400 and tuned and attempt == 1:
                    log.info("Gemini %s rejected speed settings (%s) → plain call", model, msg[:120])
                    tuned = False
                    continue
                if exc.code == 404:
                    self._dead_models.add(model)
                raise _ModelFailed(f"{model}: HTTP {exc.code} {msg[:160]}") from exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise _ModelFailed(f"{model}: transport {exc}") from exc
            text = (resp.text or "").strip()
            if not text:
                raise _ModelFailed(f"{model}: empty/blocked response")
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise _ModelFailed(f"{model}: invalid JSON") from exc
            log.info("Gemini %s answered in %.1fs", model, asyncio.get_running_loop().time() - started)
            return data
        raise _ModelFailed(f"{model}: request rejected")

    async def _generate(self, contents: list[Any], schema: dict) -> dict:
        """Hedged fallback chain: start the 1st model; if it is slow (> HEDGE_SEC) or fails, start the
        next one in parallel; the first good answer wins, the rest are cancelled."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.TOTAL_BUDGET_SEC
        queue = [m for m in self._models if m not in self._dead_models]
        errors: list[str] = []
        pending: dict[asyncio.Task, str] = {}

        def start_next() -> bool:
            left = deadline - loop.time()
            if not queue or left < 3:
                return False
            model = queue.pop(0)
            task = asyncio.create_task(self._call_model(model, contents, schema, min(self._timeout, left)))
            pending[task] = model
            return True

        start_next()
        try:
            while pending:
                wait_for = self.HEDGE_SEC if queue else max(deadline - loop.time(), 0.1)
                done, _ = await asyncio.wait(set(pending), timeout=wait_for,
                                             return_when=asyncio.FIRST_COMPLETED)
                if not done:  # current model(s) slow → hedge with the next one
                    if queue:
                        log.info("Gemini %s is slow → also asking %s", ", ".join(pending.values()), queue[0])
                        start_next()
                    elif loop.time() >= deadline:
                        break
                    continue
                for task in done:
                    model = pending.pop(task)
                    exc = task.exception()
                    if exc is None:
                        return task.result()
                    errors.append(str(exc))
                    log.warning("Gemini %s failed (%s)", model, exc)
                    if not pending:
                        start_next()
        finally:
            for task in pending:
                task.cancel()
        raise ExternalServiceError("All Gemini models failed: " + ("; ".join(errors[-4:]) or "no models"))

    async def discover_models(self) -> None:
        """Adds Flash models that really exist for this API key to the end of the chain
        (names change between Gemini generations; a wrong name in settings is skipped)."""
        try:
            pager = await asyncio.wait_for(self._client.aio.models.list(), timeout=8)
            names = []
            async for m in pager:
                name = _s(m.name).removeprefix("models/")
                actions = m.supported_actions or []
                if "generateContent" in actions and "flash" in name:
                    names.append(name)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not list Gemini models: %s", exc)
            return
        bad = ("tts", "image", "live", "audio", "embed", "robotics", "computer", "exp", "8b", "native")
        usable = [n for n in names if not any(b in n for b in bad)]
        if not usable:
            return
        self._dead_models |= {m for m in self._models if m not in names}

        def rank(n: str) -> tuple:
            ver = re.search(r"(\d+(?:\.\d+)?)", n)
            return ("preview" in n, -(float(ver.group(1)) if ver else 0), "lite" not in n, n)

        configured = [m for m in self._models if m in names]
        extra = [n for n in sorted(usable, key=rank) if n not in configured]
        self._models = (configured + extra)[: max(4, len(configured))]
        log.info("Gemini models available: %s", ", ".join(sorted(usable)))
        if self._dead_models:
            log.warning("Gemini models from settings that don't exist (skipped): %s",
                        ", ".join(sorted(self._dead_models)))

    async def _backup_generate(self, frames: list[bytes], prompt: str, schema: dict, why: str) -> dict:
        """Same task on an OpenAI-compatible vision model (OpenRouter, OpenAI, Groq, …)."""
        import base64

        url, key, model = self._backup  # type: ignore[misc]
        if self._backup_http is None:
            self._backup_http = httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=5.0))
        content: list[dict] = [{"type": "text", "text": prompt + "\n\nReply with ONLY a JSON object "
                                "that matches this JSON schema:\n" + json.dumps(schema)}]
        content += [{"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(f).decode()}}
                    for f in frames]
        body = {"model": model, "messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_object"}, "temperature": 0.2, "max_tokens": 2048}
        try:
            resp = await self._backup_http.post(f"{url}/chat/completions", json=body,
                                                headers={"Authorization": f"Bearer {key}"})
            data = resp.json()
            if resp.status_code >= 400:
                raise ExternalServiceError(f"backup AI HTTP {resp.status_code}: {str(data)[:200]}")
            text = _s(((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
            text = re.sub(r"^```(?:json)?|```$", "", text.strip()).strip()
            result = json.loads(text)
        except (httpx.HTTPError, ValueError, AttributeError, IndexError) as exc:
            raise ExternalServiceError(f"{why}; backup AI failed: {exc}") from exc
        log.info("Backup AI %s answered", model)
        return result

    async def aclose(self) -> None:
        if self._backup_http is not None:
            await self._backup_http.aclose()
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


def url_button(title: str, url: str) -> dict:
    return {"type": "web_url", "title": title[:20], "url": url}


def make_card(title: str, subtitle: str, buttons: list[dict], image: str = "") -> dict:
    """One element of an Instagram generic template (tappable buttons — plain links in DMs often
    aren't clickable for people who don't follow the account)."""
    card: dict[str, Any] = {"title": title[:80] or "Result", "buttons": buttons[:3]}
    if subtitle:
        card["subtitle"] = subtitle[:80]
    if image.startswith("https://"):
        card["image_url"] = image
    return card


def cards_as_text(cards: list[dict]) -> str:
    """Fallback when the template can't be sent: same links as plain text."""
    lines: list[str] = []
    for c in cards:
        lines.append(c.get("title", ""))
        lines += [f"  {b['title']}: {b['url']}" for b in c.get("buttons", [])]
    return "\n".join(lines)


def song_cover(result: dict) -> str:
    cover = _s(result.get("_cover"))
    if cover:
        return cover
    art = ((result.get("apple_music") or {}).get("artwork") or {}).get("url")
    if art:
        return _s(art).replace("{w}", "600").replace("{h}", "600")
    images = (((result.get("spotify") or {}).get("album") or {}).get("images")) or []
    return _s(images[0].get("url")) if images else ""


def song_card(result: dict) -> dict:
    artist, title = _s(result.get("artist")), _s(result.get("title"))
    query = quote_plus(f"{artist} {title}")
    spotify = ((result.get("spotify") or {}).get("external_urls") or {}).get("spotify") or (
        f"https://open.spotify.com/search/{query}")
    apple = (result.get("apple_music") or {}).get("url") or f"https://music.apple.com/us/search?term={query}"
    album, release = _s(result.get("album")), _s(result.get("release_date"))
    subtitle = album + (f" ({release[:4]})" if album and release else "")
    return make_card(f"{artist} — {title}", subtitle, [
        url_button("▶️ Spotify", spotify),
        url_button("🍎 Apple Music", apple),
        url_button("📺 YouTube", f"https://www.youtube.com/results?search_query={query}"),
    ], song_cover(result))


def format_song(result: dict, links: bool = True) -> str:
    artist, title = _s(result.get("artist")), _s(result.get("title"))
    album, release = _s(result.get("album")), _s(result.get("release_date"))
    query = quote_plus(f"{artist} {title}")
    spotify = ((result.get("spotify") or {}).get("external_urls") or {}).get("spotify") or (
        f"https://open.spotify.com/search/{quote_plus(f'{artist} {title}')}"
    )
    apple = (result.get("apple_music") or {}).get("url") or f"https://music.apple.com/us/search?term={query}"
    lines = ["🎵 Song found!"]
    if result.get("_note"):
        lines.append(_s(result["_note"]))
    lines.append(f"{artist} — {title}")
    if album:
        lines.append(f"💿 {album}" + (f" ({release[:4]})" if release else ""))
    if links:
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
    return "\n".join(lines)


def movie_card(data: dict) -> dict:
    title, year = _s(data.get("title")), _s(data.get("year"))
    q = quote_plus(f"{title} {year}".strip())
    return make_card(f"🎬 {title}" + (f" ({year})" if year else ""), "Where to watch & details", [
        url_button("📺 Where to watch", f"https://www.justwatch.com/us/search?q={quote_plus(title)}"),
        url_button("⭐ IMDb", f"https://www.imdb.com/find/?q={q}"),
        url_button("🎞 Trailer", f"https://www.youtube.com/results?search_query={q}+trailer"),
    ])


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
        lines.append(f"   🔎 “{_s(it.get('search_query'))}”")
    return "\n".join(lines)


def item_cards(items: list[dict]) -> list[dict]:
    cards = []
    for i, it in enumerate(items, 1):
        q = _s(it.get("search_query"))
        brand = _s(it.get("brand"))
        brand = "" if brand.lower() in ("unknown", "n/a", "none") else brand
        name = " ".join(x for x in (brand, _s(it.get("model"))) if x) or _s(it.get("category")).capitalize()
        cards.append(make_card(f"{i}. {name}", f"🔎 {q}", [
            url_button("🛍 Google Shopping", f"https://www.google.com/search?tbm=shop&q={quote_plus(q)}"),
            url_button("📦 Amazon", f"https://www.amazon.com/s?k={quote_plus(q)}"),
        ]))
    return cards


# =============================================================================
# Search orchestration
# =============================================================================


@dataclass
class BranchResult:
    found: bool = False
    text: str = ""
    failed: bool = False   # upstream API failure (do not charge for it)
    skipped: bool = False  # branch could not run (e.g. photo has no audio)
    cards: list[dict] = field(default_factory=list)  # tappable link buttons


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
                 audd: AudDClient, vision: VisionClient, shazam: ShazamClient | None = None):
        self.s = settings
        self.shazam = shazam
        self.http = http
        self.db = db
        self.ig = ig
        self.audd = audd
        self.vision = vision
        self.dedup = Deduplicator(maxsize=1000)
        self.locks = UserLocks()
        self.jobs = asyncio.Semaphore(settings.max_parallel_jobs)
        self.seen_accounts: set[str] = set()
        self.admin_fails: dict[str, int] = {}
        self.redeem_fails: dict[str, int] = {}
        self.reels = ReelExtractor(settings.ig_cookies, settings.download_proxy, settings.tmp_root)

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
                    await self.handle_media(sender, *media, mid=_s(msg.get("mid")))
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
                log.info("Attachment type=%s keys=%s", a_type, sorted((att.get("payload") or {}).keys()))
                return url, type_hint[a_type]
        return ("", "restricted") if saw_media_type else None

    async def handle_media(self, igsid: str, url: str, media_type: str, mid: str = "") -> None:
        log.info("Media received: type=%s url=%s", media_type, describe_url(url))
        ig_url = instagram_post_url(url)
        if ig_url:
            self.reels.prefetch(ig_url)
        results = await asyncio.gather(
            self.db.set_last_media(igsid, url, media_type, mid),
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
            if instagram_post_url(url):
                await self.handle_media(igsid, url, "video")
                return
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
        if lowered.startswith("admin ") and await self.handle_admin(igsid, clean):
            return
        if lowered in BALANCE_COMMANDS:
            await self.safe_send(igsid, self.balance_line(user))
            return
        if lowered in BUY_COMMANDS:
            await self.send_paywall(igsid, user, 1)
            return
        if await self.try_redeem(igsid, clean):
            return
        await self.safe_send(igsid, welcome_text(self.s.bot_name, self.balance_line(user)))

    async def handle_admin(self, igsid: str, text: str) -> bool:
        """Secret DM commands (wrong code = ordinary text; 5 wrong tries lock the sender out):
          admin <code> 500                     → +500 credits to yourself (max 10000)
          admin <code> gift 50                 → one-time code worth 50 credits, to send to a friend
          admin <code> promo SUMMER 10 100 7   → code SUMMER: 10 credits, 100 people, valid 7 days
          admin <code> codes                   → list active codes
          admin <code> off SUMMER              → disable a code"""
        code = self.s.admin_code
        if len(code) < 12 or self.admin_fails.get(igsid, 0) >= 5:
            return False
        parts = text.split()
        if len(parts) < 2 or not hmac.compare_digest(parts[1].encode(), code.encode()):
            self.admin_fails[igsid] = self.admin_fails.get(igsid, 0) + 1
            log.warning("Wrong admin code from %s (%d/5)", igsid, self.admin_fails[igsid])
            return False
        args = parts[2:]
        sub = args[0].lower() if args else ""

        def num(i: int, default: int, hi: int) -> int:
            return max(1, min(int(args[i]), hi)) if len(args) > i and args[i].isdigit() else default

        if sub == "gift":
            credits = num(1, 10, 10_000)
            gift = "GIFT-" + secrets.token_hex(4).upper()
            await self.db.create_promo(gift, credits, 1, 30)
            log.info("Admin: gift code %s for %d credits", gift, credits)
            await self.safe_send(igsid, f"🎁 Gift code for {credits} credits (1 person, 30 days):")
            await self.safe_send(igsid, gift)  # separate message = easy to copy & forward
            await self.safe_send(igsid, "Your friend just sends this code to me in DM.")
        elif sub == "promo":
            name = args[1].upper() if len(args) > 1 else ""
            if not PROMO_RE.fullmatch(name):
                await self.safe_send(igsid, "Promo name: 3–24 letters/digits, e.g. SUMMER")
                return True
            credits, uses = num(2, 5, 10_000), num(3, 100, 1_000_000)
            days = num(4, 0, 3650) if len(args) > 4 else None
            ok = await self.db.create_promo(name, credits, uses, days)
            await self.safe_send(igsid, (
                f"✅ Promo {name}: {credits} credits, up to {uses} people"
                + (f", {days} days" if days else ", no expiry")
                + "\nPeople just send the code to me in DM.") if ok else f"⚠️ Code {name} already exists")
        elif sub == "codes":
            rows = await self.db.list_promos()
            lines = [f"{r['code']}: {r['credits']} cr, used {r['used']}/{r['max_uses']}"
                     + (f", until {r['expires_at']:%b %d}" if r["expires_at"] else "") for r in rows]
            await self.safe_send(igsid, "🎟 Active codes:\n" + ("\n".join(lines) or "none"))
        elif sub == "off" and len(args) > 1:
            ok = await self.db.disable_promo(args[1].upper())
            await self.safe_send(igsid, f"⛔ {args[1].upper()} disabled" if ok else "Code not found")
        else:
            amount = num(0, 100, 10_000)
            balance = await self.db.add_credits(igsid, amount)
            log.info("Admin: +%d credits to %s (now %d)", amount, igsid, balance)
            await self.safe_send(igsid, f"🛠 Admin: +{amount} credits added.\n💳 Credits left: {balance}")
        return True

    async def try_redeem(self, igsid: str, text: str) -> bool:
        """A message that is just a promo/gift code → credits. Unknown code → False (normal reply)."""
        token = text.strip().upper()
        if token.startswith("REDEEM "):
            token = token[7:].strip()
        if not PROMO_RE.fullmatch(token) or self.redeem_fails.get(igsid, 0) >= 20:
            return False
        status, credits, balance = await self.db.redeem_promo(igsid, token)
        if status == "unknown":
            if token.startswith("GIFT-"):  # only guessing of private gift codes is rate-limited
                self.redeem_fails[igsid] = self.redeem_fails.get(igsid, 0) + 1
            return False
        log.info("Promo %s → %s for %s", token, status, igsid)
        msg = {
            "ok": f"🎁 Code applied: +{credits} credits!\n💳 Credits left: {balance}\n\n"
                  "Send me a Reel to use them 👇",
            "used": "You've already used this code 🙂",
            "expired": "This code has expired or run out 😕",
        }[status]
        await self.safe_send(igsid, msg)
        return True

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
                    outcome = await self.run_search(action, user["last_media_url"], user["last_media_mid"] or "")
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
                cards = [c for b in outcome.branches.values() for c in b.cards][:10]
                if not cards:
                    blocks.append("Want something else from this Reel? 👇")
                    await self.safe_send(igsid, "\n\n".join(blocks), QUICK_REPLIES)
                    return
                await self.safe_send(igsid, "\n\n".join(blocks))
                try:
                    await self.ig.send_generic_template(igsid, cards)
                except GraphAPIError as exc:
                    log.warning("Link buttons failed (%s) → sending links as text", exc)
                    await self.safe_send(igsid, cards_as_text(cards))
                await self.safe_send(igsid, "Want something else from this Reel? 👇", QUICK_REPLIES)
            except Exception:  # noqa: BLE001 — never leave the user hanging
                log.exception("Unexpected error while handling %s for %s", action, igsid)
                await self.safe_send(igsid, MSG_OVERLOAD)
            finally:
                if not typing_task.done():
                    typing_task.cancel()

    async def run_search(self, action: str, media_url: str, mid: str = "") -> SearchOutcome:
        async with self.jobs:
            workdir = Path(self.s.tmp_root) / uuid.uuid4().hex  # /tmp/{uuid}
            workdir.mkdir(parents=True, exist_ok=False)
            try:
                src = workdir / "source"
                ctype = await self.download_with_fallback(
                    media_url, mid, src, cover_ok=action in (FIND_MOVIE, FIND_ITEM))
                info = await probe_media(src, ctype)
                log.info("Media: kind=%s duration=%.1fs audio=%s ctype=%s",
                         info.kind, info.duration, info.has_audio, ctype)

                tasks: dict[str, Awaitable[BranchResult]] = {}
                if action in (FIND_MUSIC, FIND_ALL):
                    tasks["music"] = self._music_branch(info, workdir, strict=action == FIND_MUSIC,
                                                        hint=self.reels.cached(media_url))
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

    async def download_with_fallback(self, media_url: str, mid: str, dest: Path,
                                     cover_ok: bool = False) -> str:
        """Instagram post links go through the Reel downloader; anything else is fetched directly
        (and if that returns an HTML page, the Graph API is asked for other URLs)."""
        max_bytes = self.s.max_download_mb * 1024 * 1024
        ig_url = instagram_post_url(media_url)
        if ig_url:
            return await self.download_instagram_post(ig_url, dest, max_bytes, cover_ok)
        try:
            return await download_media(self.http, media_url, dest, max_bytes,
                                        allow_private=self.s.allow_private_urls)
        except MediaUnavailable as first_exc:
            if first_exc.user_message not in (MSG_NOT_MEDIA, MSG_EXPIRED) or not mid:
                raise
            try:
                alternatives = [u for u in await self.ig.get_message_media_urls(mid) if u != media_url]
            except Exception as exc:  # noqa: BLE001
                log.warning("Graph media lookup failed: %s", exc)
                raise first_exc from exc
            log.info("Trying %d alternative media URL(s): %s", len(alternatives),
                     [describe_url(u) for u in alternatives])
            for alt in alternatives:
                try:
                    return await download_media(self.http, alt, dest, max_bytes,
                                                allow_private=self.s.allow_private_urls)
                except MediaUnavailable as exc:
                    log.info("Alternative %s failed: %s", describe_url(alt), exc)
            raise first_exc

    async def download_instagram_post(self, ig_url: str, dest: Path, max_bytes: int, cover_ok: bool) -> str:
        allow = self.s.allow_private_urls
        post: ResolvedPost | None = None
        try:
            post = await self.reels.resolve(ig_url)
        except MediaUnavailable as exc:
            log.warning("Reel downloader: %s (%s)", exc, describe_url(ig_url))

        if post and post.video_url:
            try:
                return await download_media(self.http, post.video_url, dest, max_bytes, allow_private=allow)
            except MediaUnavailable as exc:
                if exc.user_message == MSG_TOO_BIG:
                    raise
                log.warning("Reel video download failed: %s", exc)
                self.reels.cache.pop(ig_url, None)
        elif post and (post.dash_video or post.dash_audio):
            try:
                return await self._download_dash(post, dest, max_bytes)
            except MediaUnavailable as exc:
                if exc.user_message == MSG_TOO_BIG:
                    raise
                log.warning("Reel DASH download failed: %s", exc)
                self.reels.cache.pop(ig_url, None)

        if cover_ok:  # movie / outfit search can work from the cover frame alone
            cover = (post.thumbnail if post else "") or await fetch_og_image(self.http, ig_url)
            if cover:
                try:
                    ctype = await download_media(self.http, cover, dest, max_bytes, allow_private=allow)
                    log.info("Using the Reel cover image instead of the video")
                    return ctype
                except MediaUnavailable as exc:
                    log.warning("Reel cover download failed: %s", exc)
        raise MediaUnavailable(MSG_REEL_BLOCKED, f"could not fetch {describe_url(ig_url)}")

    async def _download_dash(self, post: ResolvedPost, dest: Path, max_bytes: int) -> str:
        """Separate video + audio tracks → one mp4 (stream copy, no re-encode)."""
        allow = self.s.allow_private_urls
        parts: list[tuple[str, Path]] = []
        if post.dash_video:
            parts.append((post.dash_video, dest.with_name("dash_v")))
        if post.dash_audio:
            parts.append((post.dash_audio, dest.with_name("dash_a")))
        await asyncio.gather(*(download_media(self.http, u, p, max_bytes, allow_private=allow)
                               for u, p in parts))
        if len(parts) == 1:
            parts[0][1].replace(dest)
            return "video/mp4" if post.dash_video else "audio/mp4"
        out = dest.with_name("merged.mp4")
        rc, _out, err = await run_cmd(
            ["ffmpeg", "-v", "error", "-y", "-i", str(parts[0][1]), "-i", str(parts[1][1]),
             "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", "-f", "mp4", str(out)], timeout=15)
        if rc != 0:
            raise MediaUnavailable(MSG_DOWNLOAD_FAILED, f"dash merge failed: {err[:200]!r}")
        out.replace(dest)
        return "video/mp4"

    # ---------------------------------------------------------------- inbox polling

    def mark_seen_event(self, sender: str, timestamp_ms: int) -> None:
        """Second dedup key (sender + second) in case webhook mids differ from Graph message ids."""
        if sender and timestamp_ms:
            self.dedup.seen(f"t:{sender}:{timestamp_ms // 1000}")

    async def poll_inbox_once(self, not_before: datetime) -> None:
        own_id = self.s.ig_account_id
        now = datetime.now(timezone.utc)
        for message in await self.ig.list_recent_messages():
            created = parse_graph_time(_s(message.get("created_time")))
            if not created or created < not_before or now - created > timedelta(minutes=15):
                continue
            if now - created < timedelta(seconds=4):
                continue  # give the webhook a moment to arrive first
            sender = _s((message.get("from") or {}).get("id"))
            if not sender or sender == own_id:
                continue
            mid = _s(message.get("id"))
            second = int(created.timestamp())
            keys = [f"t:{sender}:{second + d}" for d in (-1, 0, 1)]
            if f"m:{mid}" in self.dedup._cache or any(k in self.dedup._cache for k in keys):
                continue
            self.dedup.seen(f"m:{mid}")
            self.dedup.seen(keys[1])
            ev = graph_message_to_event(message, own_id)
            if ev is None:
                continue
            log.info("Inbox poll picked up a message the webhook did not deliver (mid=%s…)", mid[:12])
            asyncio.create_task(_run_safely(self.handle_messaging_event, ev, own_id))

    async def inbox_poll_loop(self) -> None:
        if self.s.inbox_poll_sec <= 0 or not self.s.ig_account_id.isdigit():
            return
        not_before = datetime.now(timezone.utc) - timedelta(minutes=2)
        backoff = 0
        log.info("Inbox polling every %ss (catches DMs stuck in Message requests)", self.s.inbox_poll_sec)
        while True:
            try:
                await self.poll_inbox_once(not_before)
                backoff = 0
            except (GraphAPIError, httpx.HTTPError) as exc:
                backoff = min(backoff * 2 or 60, 900)
                log.warning("Inbox poll failed (%s); next try in %ss", exc, self.s.inbox_poll_sec + backoff)
            except Exception:  # noqa: BLE001
                log.exception("Inbox poll crashed")
            await asyncio.sleep(self.s.inbox_poll_sec + backoff)

    async def _shazam_once(self, info: MediaInfo, workdir: Path, rate: float = 1.0) -> dict | None:
        if not self.shazam:
            return None
        try:
            wav = await extract_wav_clip(info, workdir, rate)
        except MediaUnavailable as exc:
            log.info("Shazam clip skipped: %s", exc)
            return None
        return await self.shazam.recognize(wav)

    async def _ai_song(self, info: MediaInfo, workdir: Path, caption: str, label: str) -> dict | None:
        """Gemini listens to the clip (+ caption / audio label) → guess → verified in iTunes catalog."""
        try:
            wav = await extract_wav_clip(info, workdir)
            data = await self.vision.identify_song(wav, caption, label)
        except (MediaUnavailable, ExternalServiceError) as exc:
            log.warning("AI song search failed: %s", exc)
            return None
        artist, title = _s(data.get("artist")), _s(data.get("title"))
        try:
            conf = float(data.get("confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        log.info("AI song guess: %r — %r version=%r conf=%.2f evidence=%r", artist, title,
                 _s(data.get("version")), conf, _s(data.get("evidence"))[:120])
        if not data.get("found") or not title or conf < 0.5:
            return None
        verified = await verify_song_itunes(self.http, artist, title)
        if not verified:
            log.info("AI guess not found in the Apple Music catalog → ignored")
            return None
        version = _s(data.get("version")).lower()
        if version and version not in ("original", "unknown", "n/a", "none"):
            verified["_note"] = f"🎛 This Reel uses a {version[:40]} of:"
        elif conf < 0.8:
            verified["_note"] = "🤔 Most likely:"
        return verified

    async def recognize_song(self, info: MediaInfo, workdir: Path,
                             hint: ResolvedPost | None = None) -> dict | None:
        """0) Instagram's licensed-track label → 1) AudD + Shazam race → 2) Shazam on sped-up /
        slowed versions + Gemini listening (verified in the iTunes catalog).
        Raises ExternalServiceError only if AudD failed AND nothing was found."""
        mp3 = await extract_audio_clip(info, workdir)  # MediaUnavailable for photos / silent clips
        loop = asyncio.get_running_loop()
        started = loop.time()

        if hint and hint.music_title and not hint.music_original:
            log.info("Song from the Instagram audio label: %s — %s", hint.music_artist, hint.music_title)
            return {"artist": hint.music_artist, "title": hint.music_title}

        label = ""
        if hint and hint.music_title:
            label = f"{hint.music_artist}: {hint.music_title} (user-uploaded original audio)"
        ai_task = (asyncio.create_task(self._ai_song(info, workdir, hint.caption if hint else "", label))
                   if self.s.ai_song_enabled else None)
        try:
            audd_task = asyncio.create_task(self.audd.recognize(mp3))
            names = {audd_task: "AudD", asyncio.create_task(self._shazam_once(info, workdir)): "Shazam"}
            for task in names:  # mark errors as retrieved even if the other engine wins first
                task.add_done_callback(lambda t: t.cancelled() or t.exception())
            pending: set[asyncio.Task] = set(names)
            audd_error: BaseException | None = None
            try:  # whichever fingerprint engine finds the song first wins
                while pending:
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        if task.exception() is not None:
                            if task is audd_task:
                                audd_error = task.exception()
                            else:
                                log.warning("Shazam branch error: %s", task.exception())
                            continue
                        if task.result():
                            log.info("Song found by %s in %.1fs", names[task], loop.time() - started)
                            return task.result()
            finally:
                for task in pending:
                    task.cancel()

            if self.shazam:  # 0.8 / 0.88 undo "sped up" edits, 1.2 undoes "slowed + reverb"
                rates = (0.8, 0.88, 1.2)
                variants = await asyncio.gather(*(self._shazam_once(info, workdir, r) for r in rates),
                                                return_exceptions=True)
                for rate, res in zip(rates, variants, strict=True):
                    if res and not isinstance(res, BaseException):
                        log.info("Song found by Shazam at speed x%.2f in %.1fs", 1 / rate, loop.time() - started)
                        res["_note"] = ("⚡ This Reel uses a sped-up version of:" if rate < 1
                                        else "🐢 This Reel uses a slowed-down version of:")
                        return res
            if ai_task:
                res = await ai_task
                if res:
                    log.info("Song found by AI (verified) in %.1fs", loop.time() - started)
                    return res
            log.info("Song not recognised (AudD %s, Shazam %s, AI %s) in %.1fs",
                     "error" if audd_error else "miss", "miss" if self.shazam else "off",
                     "miss" if ai_task else "off", loop.time() - started)
            if audd_error:
                raise audd_error
            return None
        finally:
            if ai_task and not ai_task.done():
                ai_task.cancel()

    async def _music_branch(self, info: MediaInfo, workdir: Path, strict: bool,
                            hint: ResolvedPost | None = None) -> BranchResult:
        try:
            result = await self.recognize_song(info, workdir, hint)
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
        return BranchResult(found=True, text=format_song(result, links=False), cards=[song_card(result)])

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
            return (BranchResult(found=True, text=format_movie(data), cards=[movie_card(data)])
                    if movie_is_found(data) else BranchResult())
        if action == FIND_ITEM:
            items = filter_items(data)
            return (BranchResult(found=True, text=format_items(items), cards=item_cards(items))
                    if items else BranchResult())

        # FIND_ALL
        parts: list[str] = []
        cards: list[dict] = []
        movie = data.get("movie") or {}
        if movie_is_found(movie):
            parts.append(format_movie(movie))
            cards.append(movie_card(movie))
        items = filter_items(data.get("items") or {})
        if items:
            parts.append(format_items(items))
            cards += item_cards(items)
        return BranchResult(found=bool(parts), text="\n\n".join(parts), cards=cards)

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
            except (GraphAPIError, httpx.HTTPError) as exc:
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
        [settings.gemini_primary_model, settings.gemini_fallback_model,
         *settings.gemini_extra_models.split(",")],
        settings.gemini_timeout_sec,
        settings.gemini_thinking_level,
        backup=(settings.backup_ai_url, settings.backup_ai_key, settings.backup_ai_model),
    )
    shazam: ShazamClient | None = None
    if settings.shazam_enabled:
        try:
            shazam = ShazamClient()
        except Exception as exc:  # noqa: BLE001 — Shazam is optional
            log.warning("Shazam disabled: %s", exc)
    bot = Bot(settings, http, db, ig, audd, vision, shazam)
    await bot.bootstrap_token()
    await bot.ensure_webhook_subscription()
    refresh_task = asyncio.create_task(bot.token_refresh_loop())
    poll_task = asyncio.create_task(bot.inbox_poll_loop())
    await vision.discover_models()
    log.info("Bot started [build %s] (AI chain: %s)", BUILD, vision.describe)
    try:
        yield
    finally:
        refresh_task.cancel()
        poll_task.cancel()
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
            sender_id = _s((ev.get("sender") or {}).get("id"))
            ts = int(ev.get("timestamp") or 0)
            if bot.dedup.seen(f"m:{mid}"):
                log.info("Duplicate webhook ignored: %s", mid)
                continue
            bot.mark_seen_event(sender_id, ts)
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
