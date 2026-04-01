from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import spotipy
from spotipy.oauth2 import SpotifyOAuth, SpotifyClientCredentials
import psycopg2
from db import conn, DATABASE_URL
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import logging
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("gemify")

def get_cursor():
    global conn
    if conn.closed:
        from db import DATABASE_URL
        conn = psycopg2.connect(DATABASE_URL)
    conn.rollback()
    return conn.cursor(cursor_factory=RealDictCursor)
import json
import math
import os
import pickle
import re
import time
import numpy as np
import networkx as nx
import requests as http_requests
from concurrent.futures import ThreadPoolExecutor, as_completed

_artists_cache = {}  # spotify_id -> {"ts": float, "data": list}
_refresh_in_progress = set()  # spotify_ids currently being refreshed
ARTISTS_CACHE_TTL = 21600  # 6 hours

_mbid_name_cache = {}  # mbid -> canonical name (MusicBrainz)

# ── XGBoost helpers ──────────────────────────────────────────────────────────

def _get_user_history_features(spotify_id: str) -> dict:
    """Query feedback + recommendations to compute per-user history features."""
    try:
        cur = get_cursor()
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM feedback WHERE spotify_user_id = %s",
            (spotify_id,)
        )
        feedback_count = cur.fetchone()["cnt"]

        if feedback_count == 0:
            cur.close()
            return {"feedback_count": 0, "avg_liked_listeners": 0.0, "liked_b1_rate": 0.5}

        cur.execute("""
            SELECT r.monthly_listeners, r.batch_number
            FROM feedback f
            JOIN recommendations r
              ON r.spotify_user_id = f.spotify_user_id
             AND lower(r.name) = lower(f.artist_name)
            WHERE f.spotify_user_id = %s AND f.label = 1
        """, (spotify_id,))
        liked = cur.fetchall()
        cur.close()

        if not liked:
            return {"feedback_count": feedback_count, "avg_liked_listeners": 0.0, "liked_b1_rate": 0.5}

        listeners_vals = [row["monthly_listeners"] or 0 for row in liked]
        batch_vals = [row["batch_number"] or 1 for row in liked]

        avg_liked_listeners = float(np.mean(listeners_vals))
        liked_b1_rate = float(sum(1 for b in batch_vals if b == 1) / len(batch_vals))

        return {
            "feedback_count": feedback_count,
            "avg_liked_listeners": avg_liked_listeners,
            "liked_b1_rate": liked_b1_rate,
        }
    except Exception as e:
        logger.warning(f"[xgb] history features failed: {e}")
        return {"feedback_count": 0, "avg_liked_listeners": 0.0, "liked_b1_rate": 0.5}


def _build_feature_matrix(candidates: list, session: dict, history: dict) -> np.ndarray:
    """
    Build (N, 13) feature matrix for XGBoost inference or training.

    candidates: list of recommendation dicts (with ML fields present)
    session:    { avg_seed_listeners, seed_count }
    history:    { feedback_count, avg_liked_listeners, liked_b1_rate }
    """
    rows = []
    for r in candidates:
        rows.append([
            r.get("sum_weight") or 0.0,                    # 1
            r.get("batch_number") or 1,                    # 2
            r.get("listeners") or 0.0,                     # 3
            r.get("popularity_factor") or 0.85,            # 4
            r.get("final_score") or 0.0,                   # 5
            r.get("n_seeds") or 1,                         # 6
            1 if (r.get("listeners") or 0) > 0 else 0,    # 7 has_listeners_data
            1 if r.get("image") else 0,                    # 8 has_image
            session.get("avg_seed_listeners") or 0.0,      # 9
            session.get("seed_count") or 1,                # 10
            history.get("avg_liked_listeners") or 0.0,     # 11
            history.get("liked_b1_rate") or 0.5,           # 12
            history.get("feedback_count") or 0,            # 13
        ])
    return np.array(rows, dtype=float)


app = FastAPI()

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

# Ensure recommendations table exists
try:
    _init_cur = conn.cursor()
    _init_cur.execute("""
        CREATE TABLE IF NOT EXISTS recommendations (
            id SERIAL PRIMARY KEY,
            spotify_user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            image_url TEXT,
            monthly_listeners BIGINT,
            rank INT,
            added_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(spotify_user_id, name)
        )
    """)
    conn.commit()
    _init_cur.close()
except Exception as e:
    print(f"[startup] DB init: {e}", flush=True)
    conn.rollback()


SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

scope = "user-top-read user-follow-read playlist-modify-private playlist-modify-public"

sp_oauth = SpotifyOAuth(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET,
    redirect_uri=SPOTIFY_REDIRECT_URI,
    scope=scope
)

@app.get("/login")
def login():
    auth_url = sp_oauth.get_authorize_url()
    return RedirectResponse(auth_url)

@app.get("/callback")
def callback(background_tasks: BackgroundTasks, code: str = None):
    logger.info(f"[callback] hit — code={'yes' if code else 'no'}")
    token_info = sp_oauth.get_access_token(code)
    sp = spotipy.Spotify(auth=token_info['access_token'])
    user = sp.current_user()
    spotify_id = user['id']

    cur = get_cursor()
    cur.execute("""
        INSERT INTO users (spotify_id, token_info)
        VALUES (%s, %s)
        ON CONFLICT (spotify_id) DO UPDATE
        SET token_info = EXCLUDED.token_info, last_login = NOW()
    """, (spotify_id, json.dumps(token_info)))
    conn.commit()
    cur.close()

    # Always refresh on login in background — ensures new artists/images are picked up immediately
    _start_refresh(background_tasks, spotify_id, sp)

    return RedirectResponse(f"/app?spotify_id={spotify_id}")

def _get_sp(spotify_id: str):
    cur = get_cursor()
    cur.execute("SELECT token_info FROM users WHERE spotify_id = %s", (spotify_id,))
    row = cur.fetchone()
    cur.close()
    if not row:
        return None, None
    token_info = dict(row["token_info"])
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
        cur = get_cursor()
        cur.execute("UPDATE users SET token_info = %s WHERE spotify_id = %s",
                    (json.dumps(token_info), spotify_id))
        conn.commit()
        cur.close()
    return spotipy.Spotify(auth=token_info["access_token"]), token_info


def _all_followed_artists(sp):
    """All followed artists, paginated."""
    artists, after = [], None
    while True:
        resp = sp.current_user_followed_artists(limit=50, after=after)
        items = resp["artists"]["items"]
        artists.extend(items)
        after = resp["artists"]["cursors"]["after"]
        if not after:
            break
    return artists


def _all_top_artists(sp, time_range: str, limit: int):
    """Top artists up to limit, paginated past 50."""
    artists, offset = [], 0
    while offset < limit:
        batch = min(50, limit - offset)
        resp = sp.current_user_top_artists(limit=batch, offset=offset, time_range=time_range)
        artists.extend(resp["items"])
        if len(resp["items"]) < batch:
            break
        offset += batch
    return artists


def _all_top_tracks_ranked(sp, time_range: str, limit: int):
    """Top tracks → [(rank, artist)], 1-indexed."""
    result, offset, rank = [], 0, 1
    while offset < limit:
        batch = min(50, limit - offset)
        resp = sp.current_user_top_tracks(limit=batch, offset=offset, time_range=time_range)
        for track in resp["items"]:
            for artist in track["artists"]:
                result.append((rank, artist))
            rank += 1
        if len(resp["items"]) < batch:
            break
        offset += batch
    return result


def _fetch_and_cache_artists(spotify_id: str, sp):
    logger.info(f"[artists] starting fetch for {spotify_id}")
    try:
        _fetch_and_cache_artists_inner(spotify_id, sp)
    except Exception as e:
        logger.info(f"[artists] fetch failed for {spotify_id}: {e}")
    finally:
        _refresh_in_progress.discard(spotify_id)
        logger.info(f"[artists] fetch done for {spotify_id}")


def _fetch_and_cache_artists_inner(spotify_id: str, sp):
    artist_map = {}  # id -> entry

    def get_or_create(a):
        aid = a["id"]
        images = a.get("images", [])
        if aid not in artist_map:
            artist_map[aid] = {
                "id": aid,
                "name": a["name"],
                "image": images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None),
                "image_thumb": images[-1]["url"] if images else None,
                "sources": [],
                "score": 0.0,
            }
        else:
            entry = artist_map[aid]
            if images and not entry["image"]:
                entry["image"] = images[1]["url"] if len(images) > 1 else images[0]["url"]
                entry["image_thumb"] = images[-1]["url"]
        return artist_map[aid]

    # Followed artists — flat +8 each
    for a in _all_followed_artists(sp):
        entry = get_or_create(a)
        entry["sources"].append("followed")
        entry["score"] += 8

    # Top artists — score by rank
    for rank, a in enumerate(_all_top_artists(sp, "short_term", 50), start=1):
        entry = get_or_create(a)
        entry["sources"].append("top-4weeks")
        entry["score"] += (50 - rank + 1) * 3

    for rank, a in enumerate(_all_top_artists(sp, "medium_term", 50), start=1):
        entry = get_or_create(a)
        entry["sources"].append("top-6months")
        entry["score"] += (50 - rank + 1) * 1.5

    for rank, a in enumerate(_all_top_artists(sp, "long_term", 50), start=1):
        entry = get_or_create(a)
        entry["sources"].append("top-alltime")
        entry["score"] += (50 - rank + 1) * 0.5

    # Top track artists — score by track rank
    for rank, a in _all_top_tracks_ranked(sp, "short_term", 50):
        entry = get_or_create(a)
        entry["sources"].append("track-4weeks")
        entry["score"] += (50 - rank + 1) * 0.3

    for rank, a in _all_top_tracks_ranked(sp, "medium_term", 50):
        entry = get_or_create(a)
        entry["sources"].append("track-6months")
        entry["score"] += (50 - rank + 1) * 0.1

    ordered = sorted(artist_map.values(), key=lambda x: x["score"], reverse=True)

    def cache_artists(artists_to_cache):
        """Upsert images to artists table. Uses own cursor — avoids tx conflicts on shared conn."""
        to_insert = [a for a in artists_to_cache if a.get("image")]
        if not to_insert:
            return
        try:
            c = get_cursor()
            for a in to_insert:
                c.execute("""
                    INSERT INTO artists (spotify_id, name, image_url)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (spotify_id) DO UPDATE
                    SET name = EXCLUDED.name,
                        image_url = EXCLUDED.image_url,
                        cached_at = NOW()
                """, (a["id"], a["name"], a["image"]))
            conn.commit()
            c.close()
            logger.info(f"[artists] cached {len(to_insert)} artist images to DB")
        except Exception as e:
            logger.info(f"[artists] DB cache write failed: {e}")
            conn.rollback()

    # Step 1: cache artists we already have images for
    cache_artists(ordered)

    # Step 2: for artists missing images, check DB cache (only entries cached within 7 days)
    missing = [a for a in ordered if not a["image"]]
    if missing:
        try:
            c = get_cursor()
            ids = [a["id"] for a in missing]
            c.execute("""
                SELECT spotify_id, image_url FROM artists
                WHERE spotify_id = ANY(%s)
                  AND image_url IS NOT NULL
                  AND cached_at > NOW() - INTERVAL '7 days'
            """, (ids,))
            db_cached = {row["spotify_id"]: row["image_url"] for row in c.fetchall()}
            c.close()
        except Exception as e:
            logger.info(f"[artists] DB cache read failed: {e}")
            db_cached = {}

        still_missing = []
        for a in missing:
            if a["id"] in db_cached:
                a["image"] = db_cached[a["id"]]
                a["image_thumb"] = db_cached[a["id"]]
            else:
                still_missing.append(a)

        # Step 3: fetch from Spotify for artists still missing images, then cache
        logger.info(f"[artists] {len(missing)} missing — {len(missing) - len(still_missing)} from DB cache, {len(still_missing)} fetching from Spotify")
        if still_missing:
            for a in still_missing:
                try:
                    full = sp.artist(a["id"])
                    images = full.get("images", [])
                    a["image"] = images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)
                    a["image_thumb"] = images[-1]["url"] if images else None
                except Exception as e:
                    logger.info(f"[artists] Spotify fetch failed for {a['id']}: {e}")

            fetched = sum(1 for a in still_missing if a.get("image"))
            logger.info(f"[artists] individual fetch complete — {fetched}/{len(still_missing)} got images")
            cache_artists(still_missing)
    for a in ordered:
        a.pop("_fetch_error", None)
        a["score"] = round(a["score"], 1)

    # Persist ranked list to DB so it survives server restarts
    cur = get_cursor()
    cur.execute(
        "UPDATE users SET artists_data = %s, artists_cached_at = NOW() WHERE spotify_id = %s",
        (json.dumps(ordered), spotify_id)
    )
    conn.commit()
    cur.close()

    _artists_cache[spotify_id] = {"ts": time.time(), "data": ordered}
    logger.info(f"[artists] saved {len(ordered)} artists to cache for {spotify_id}")


def _start_refresh(background_tasks: BackgroundTasks, spotify_id: str, sp):
    if spotify_id not in _refresh_in_progress:
        _refresh_in_progress.add(spotify_id)
        background_tasks.add_task(_fetch_and_cache_artists, spotify_id, sp)


@app.get("/top-artists")
def top_artists(spotify_id: str, background_tasks: BackgroundTasks):
    pending = spotify_id in _refresh_in_progress

    # Check memory cache
    cached = _artists_cache.get(spotify_id)
    if cached:
        if time.time() - cached["ts"] >= ARTISTS_CACHE_TTL:
            sp, _ = _get_sp(spotify_id)
            if sp:
                _start_refresh(background_tasks, spotify_id, sp)
                pending = True
        return {"total": len(cached["data"]), "artists": cached["data"], "refresh_pending": pending}

    # Fall back to DB
    cur = get_cursor()
    cur.execute("SELECT artists_data, artists_cached_at FROM users WHERE spotify_id = %s", (spotify_id,))
    row = cur.fetchone()
    cur.close()
    if row and row["artists_data"]:
        artists = row["artists_data"]
        cached_ts = row["artists_cached_at"].timestamp() if row["artists_cached_at"] else time.time()
        _artists_cache[spotify_id] = {"ts": cached_ts, "data": artists}
        if time.time() - cached_ts >= ARTISTS_CACHE_TTL:
            sp, _ = _get_sp(spotify_id)
            if sp:
                _start_refresh(background_tasks, spotify_id, sp)
                pending = True
        return {"total": len(artists), "artists": artists, "refresh_pending": pending}

    return {"total": 0, "artists": [], "refresh_pending": pending}


@app.get("/profile")
def me(spotify_id: str):
    sp, _ = _get_sp(spotify_id)
    if not sp:
        return {"error": "User not found"}
    user = sp.current_user()
    images = user.get("images", [])
    return {
        "display_name": user.get("display_name"),
        "image": images[0]["url"] if images else None,
    }


@app.get("/albums")
def get_albums(spotify_id: str):
    cur = get_cursor()
    cur.execute(
        "SELECT id, name, artists, created_at FROM albums WHERE spotify_user_id = %s ORDER BY created_at DESC",
        (spotify_id,)
    )
    albums = [dict(row) for row in cur.fetchall()]
    cur.close()
    return {"albums": albums}


class RecommendationsRequest(BaseModel):
    spotify_id: str
    artists: list
    exclude: list = []


def _lastfm_similar(artist_name: str, limit: int, api_key: str):
    """Last.fm artist.getSimilar → [(name, mbid, score)]."""
    try:
        r = http_requests.get("https://ws.audioscrobbler.com/2.0/", params={
            "method": "artist.getSimilar",
            "artist": artist_name,
            "api_key": api_key,
            "autocorrect": 1,
            "limit": limit,
            "format": "json",
        }, timeout=15)
        items = r.json().get("similarartists", {}).get("artist", [])
        return [(s["name"], s.get("mbid", ""), float(s.get("match", 0))) for s in items]
    except Exception as e:
        logger.info(f"[recs] Last.fm failed for {artist_name}: {e}")
        return []


_MB_HEADERS = {"User-Agent": "Gemify/1.0 (gemify-app)"}


def _mb_lookup(mbid: str) -> tuple:
    """MBID → (name, area) via MusicBrainz."""
    r = http_requests.get(
        f"https://musicbrainz.org/ws/2/artist/{mbid}",
        params={"fmt": "json"},
        headers=_MB_HEADERS,
        timeout=10,
    )
    if r.status_code == 200:
        data = r.json()
        name = data.get("name")
        area = data.get("area") or {}
        return name, area.get("name")
    return None, None


def _mb_search(name: str) -> tuple:
    """MusicBrainz name search → (canonical_name, area) if confidence ≥ 95."""
    r = http_requests.get(
        "https://musicbrainz.org/ws/2/artist",
        params={"query": name, "fmt": "json", "limit": 1},
        headers=_MB_HEADERS,
        timeout=10,
    )
    if r.status_code == 200:
        artists = r.json().get("artists", [])
        if artists and artists[0].get("score", 0) >= 95:
            a = artists[0]
            area = a.get("area") or {}
            return a.get("name"), area.get("name")
    return None, None


def _resolve_names(entries: list) -> dict:
    """Last.fm names → canonical names+areas via MusicBrainz. {orig_name: {name, area}}. In-memory cached."""
    result = {}
    for entry in entries:
        name = entry["name"]
        mbid = entry.get("mbid", "")

        if name in _mbid_name_cache:
            result[name] = _mbid_name_cache[name]
            continue

        canonical, area = None, None
        try:
            if mbid:
                canonical, area = _mb_lookup(mbid)
            else:
                canonical, area = _mb_search(name)
        except Exception as e:
            logger.info(f"[mbid] failed for {name!r}: {e}")

        resolved = {"name": canonical or name, "area": area}
        if canonical and canonical != name:
            logger.info(f"[mbid] {name!r} -> {canonical!r} ({area})")
        _mbid_name_cache[name] = resolved
        result[name] = resolved

        time.sleep(1)  # MusicBrainz rate limit: 1 req/sec
    return result


@app.post("/recommendations")
def get_recommendations(req: RecommendationsRequest):
    print(f"[recs] ENDPOINT HIT spotify_id={req.spotify_id}", flush=True)
    _mbid_name_cache.clear()
    logger.info(f"[recs] starting — seeds={req.artists}")
    lastfm_key = os.getenv("LASTFM_API_KEY")

    G = nx.DiGraph()
    seed_lower = {a.lower() for a in req.artists}

    # ── Batch 0 ──
    for a in req.artists:
        G.add_node(a, batch_number=0, sum_weight=1.0, seeds=[a])

    # ── Batch 1 ──
    n_seed = len(req.artists)
    limit_b1 = 50 if n_seed == 1 else max(1, 100 // n_seed)

    b1_raw = []  # (seed, name, mbid, score)
    with ThreadPoolExecutor(max_workers=min(n_seed, 10)) as pool:
        futures = {pool.submit(_lastfm_similar, a, limit_b1, lastfm_key): a for a in req.artists}
        for future in as_completed(futures):
            seed = futures[future]
            for name, mbid, score in future.result():
                if name.lower() not in seed_lower:
                    b1_raw.append((seed, name, mbid, score))

    b1 = {}  # lower -> {name, mbid, sum_weight, seeds}
    for seed, name, mbid, score in b1_raw:
        key = name.lower()
        if key not in b1:
            b1[key] = {"name": name, "mbid": mbid or "", "sum_weight": 0.0, "seeds": []}
        if mbid and not b1[key]["mbid"]:
            b1[key]["mbid"] = mbid
        b1[key]["sum_weight"] += G.nodes[seed]["sum_weight"] * score
        if seed not in b1[key]["seeds"]:
            b1[key]["seeds"].append(seed)

    for d in b1.values():
        G.add_node(d["name"], batch_number=1, sum_weight=d["sum_weight"], seeds=d["seeds"])
    for seed, name, mbid, score in b1_raw:
        if name in G.nodes:
            G.add_edge(seed, name, weight=score)

    # Prune batch 1
    b1_sorted = sorted(b1.values(), key=lambda x: x["sum_weight"], reverse=True)
    above_b1 = [d for d in b1_sorted if d["sum_weight"] >= 0.1]
    surviving_b1 = above_b1 if len(above_b1) >= 10 else b1_sorted[:10]
    surviving_b1_lower = {d["name"].lower() for d in surviving_b1}
    b1_all_lower = set(b1.keys())
    logger.info(f"[recs] b1 total={len(b1)} surviving={len(surviving_b1)}")

    # ── Batch 2 ──
    b2_raw = []  # (parent, name, mbid, score)
    with ThreadPoolExecutor(max_workers=min(len(surviving_b1), 20)) as pool:
        futures = {pool.submit(_lastfm_similar, d["name"], 5, lastfm_key): d["name"] for d in surviving_b1}
        for future in as_completed(futures):
            parent = futures[future]
            for name, mbid, score in future.result():
                nl = name.lower()
                if nl in seed_lower or nl in b1_all_lower:
                    # structural edge only for surviving b1 nodes
                    if nl in surviving_b1_lower:
                        node_name = next(d["name"] for d in surviving_b1 if d["name"].lower() == nl)
                        G.add_edge(parent, node_name, weight=score)
                    continue
                b2_raw.append((parent, name, mbid, score))

    b2 = {}  # lower -> {name, mbid, sum_weight, seeds}
    for parent, name, mbid, score in b2_raw:
        key = name.lower()
        parent_node = G.nodes[parent]
        if key not in b2:
            b2[key] = {"name": name, "mbid": mbid or "", "sum_weight": 0.0, "seeds": []}
        if mbid and not b2[key]["mbid"]:
            b2[key]["mbid"] = mbid
        b2[key]["sum_weight"] += parent_node["sum_weight"] * score
        for seed in parent_node.get("seeds", []):
            if seed not in b2[key]["seeds"]:
                b2[key]["seeds"].append(seed)

    for d in b2.values():
        G.add_node(d["name"], batch_number=2, sum_weight=d["sum_weight"], seeds=d["seeds"])
    for parent, name, mbid, score in b2_raw:
        if name in G.nodes:
            G.add_edge(parent, name, weight=score)

    # Prune batch 2
    b2_sorted = sorted(b2.values(), key=lambda x: x["sum_weight"], reverse=True)
    above_b2 = [d for d in b2_sorted if d["sum_weight"] >= 0.2]
    surviving_b2 = above_b2 if len(above_b2) >= 10 else b2_sorted[:10]
    logger.info(f"[recs] b2 total={len(b2)} surviving={len(surviving_b2)}")

    # ── Listener lookup pass 1 (raw Last.fm names) ──
    all_surviving = surviving_b1 + surviving_b2
    listeners_map = {}
    try:
        names_lower = [d["name"].lower() for d in all_surviving]
        cur = get_cursor()
        cur.execute(
            "SELECT name, listeners FROM artist_listeners WHERE lower(name) = ANY(%s)",
            (names_lower,)
        )
        listeners_map = {row["name"].lower(): row["listeners"] for row in cur.fetchall()}
        cur.close()
        matched = sum(1 for d in all_surviving if d["name"].lower() in listeners_map)
        logger.info(f"[recs] listener pass 1: {matched}/{len(all_surviving)} matched")
    except Exception as e:
        logger.info(f"[recs] listener pass 1 failed: {e}")

    # ── MusicBrainz resolution for top 5 unmatched + any with non-standard chars ──
    _standard_chars = set("abcdefghijklmnopqrstuvwxyz0123456789 '-.")
    def _has_nonstandard(name):
        return any(c not in _standard_chars for c in name.lower())

    unmatched = [d for d in all_surviving if d["name"].lower() not in listeners_map]
    top5 = sorted(unmatched, key=lambda x: x["sum_weight"], reverse=True)[:5]
    nonstandard = [d for d in unmatched if _has_nonstandard(d["name"]) and d not in top5]
    unmatched_top5 = top5 + nonstandard
    if unmatched_top5:
        logger.info(f"[recs] resolving {len(unmatched_top5)} unmatched via MusicBrainz")
        mb_entries = [{"name": d["name"], "mbid": d.get("mbid", "")} for d in unmatched_top5]
        name_map = _resolve_names(mb_entries)
        for d in unmatched_top5:
            resolved = name_map.get(d["name"])
            d["canonical_name"] = resolved["name"] if resolved else d["name"]
            d["area"] = resolved.get("area") if resolved else None

        # Pass 2: re-query with canonical names
        newly_resolved = [d for d in unmatched_top5 if d.get("canonical_name", d["name"]).lower() != d["name"].lower()]
        if newly_resolved:
            try:
                canon_lower = [d["canonical_name"].lower() for d in newly_resolved]
                cur = get_cursor()
                cur.execute(
                    "SELECT name, listeners FROM artist_listeners WHERE lower(name) = ANY(%s)",
                    (canon_lower,)
                )
                for row in cur.fetchall():
                    listeners_map[row["name"].lower()] = row["listeners"]
                cur.close()
                for d in newly_resolved:
                    if d["canonical_name"].lower() in listeners_map:
                        listeners_map[d["canonical_name"].lower()] = listeners_map[d["canonical_name"].lower()]
                matched2 = sum(1 for d in newly_resolved if d["canonical_name"].lower() in listeners_map)
                logger.info(f"[recs] listener pass 2: {matched2}/{len(newly_resolved)} newly matched")
            except Exception as e:
                logger.info(f"[recs] listener pass 2 failed: {e}")

    # ── Pass 3: LIKE fuzzy lookup for non-standard char artists still unmatched ──
    still_unmatched_ns = [
        d for d in unmatched_top5
        if _has_nonstandard(d["name"])
        and d.get("canonical_name", d["name"]).lower() not in listeners_map
        and d["name"].lower() not in listeners_map
    ]
    if still_unmatched_ns:
        try:
            cur = get_cursor()
            for d in still_unmatched_ns:
                pattern = re.sub(r'[^a-z0-9 \'\-.]', '%', d["name"].lower())
                cur.execute(
                    "SELECT name, listeners FROM artist_listeners WHERE lower(name) LIKE %s LIMIT 1",
                    (pattern,)
                )
                row = cur.fetchone()
                if row:
                    listeners_map[row["name"].lower()] = row["listeners"]
                    d["canonical_name"] = row["name"]
                    logger.info(f"[recs] pass 3 LIKE matched {d['name']!r} -> {row['name']!r}")
            cur.close()
        except Exception as e:
            logger.info(f"[recs] listener pass 3 failed: {e}")

    # ── Popularity factor ──
    def popularity_factor(listeners):
        peak, sl, sr = 2_000_000, 4_680_000, 1_930_000
        n = 0 if listeners is None else listeners
        if n <= 500_000:
            return 0.85
        sigma = sl if n <= peak else sr
        return math.exp(-((n - peak) ** 2) / (2 * sigma ** 2))

    # ── Score, filter, sort ──
    results = []
    for d in all_surviving:
        if "&" in d["name"] or "," in d["name"]:
            continue
        display_name = d.get("canonical_name", d["name"])
        area = d.get("area")
        listeners = listeners_map.get(display_name.lower()) or listeners_map.get(d["name"].lower())
        pf = round(popularity_factor(listeners), 4)
        final_score = round(d["sum_weight"] * pf, 4)
        results.append({
            "name": display_name,
            "batch": G.nodes[d["name"]]["batch_number"],
            "sum_weight": round(d["sum_weight"], 4),
            "popularity_factor": pf,
            "final_score": final_score,
            "from": d["seeds"],
            "listeners": listeners,
            "area": area,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    results = results[:10]

    # ── Image lookup (top 10 only) ──
    # Pass 1: check Supabase cache
    try:
        names = [r["name"] for r in results]
        cur = get_cursor()
        cur.execute(
            "SELECT name, image_url FROM artists WHERE name = ANY(%s) AND image_url IS NOT NULL",
            (names,)
        )
        image_map = {row["name"]: row["image_url"] for row in cur.fetchall()}
        cur.close()
        for r in results:
            r["image"] = image_map.get(r["name"])
    except Exception as e:
        logger.info(f"[recs] image cache lookup failed: {e}")
        for r in results:
            r["image"] = None

    # Pass 2: fetch from Spotify for any still missing, then cache
    missing_image = [r for r in results if not r.get("image")]
    if missing_image:
        try:
            sp_cc = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET
            ))
            to_cache = []
            for r in missing_image:
                search = sp_cc.search(q=f"artist:{r['name']}", type="artist", limit=5)
                items = search.get("artists", {}).get("items", [])
                artist = next((a for a in items if a["name"].lower() == r["name"].lower()), None)
                if artist:
                    images = artist.get("images", [])
                    image_url = images[1]["url"] if len(images) > 1 else (images[0]["url"] if images else None)
                    if image_url:
                        r["image"] = image_url
                        to_cache.append((artist["id"], r["name"], image_url))
            if to_cache:
                cur = get_cursor()
                for spotify_id, name, image_url in to_cache:
                    cur.execute("""
                        INSERT INTO artists (spotify_id, name, image_url)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (spotify_id) DO UPDATE
                            SET name = EXCLUDED.name,
                                image_url = EXCLUDED.image_url,
                                cached_at = NOW()
                    """, (spotify_id, name, image_url))
                conn.commit()
                cur.close()
                logger.info(f"[recs] cached {len(to_cache)} new images to Supabase")
        except Exception as e:
            logger.info(f"[recs] Spotify image fetch failed: {e}")

    logger.info(f"[recs] done — top 5: {[r['name'] for r in results]}")

    # ── Save to recommendations table (before stripping, so ML features are available) ──
    exclude_lower = {n.lower() for n in req.exclude}
    display_results = [r for r in results if r["name"].lower() not in exclude_lower]
    try:
        cur = get_cursor()
        for rank, r in enumerate(display_results[:5], start=1):
            cur.execute("""
                INSERT INTO recommendations (
                    spotify_user_id, name, image_url, monthly_listeners, rank,
                    sum_weight, batch_number, popularity_factor, n_seeds
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (spotify_user_id, name) DO UPDATE
                    SET image_url         = EXCLUDED.image_url,
                        monthly_listeners = EXCLUDED.monthly_listeners,
                        rank              = EXCLUDED.rank,
                        sum_weight        = EXCLUDED.sum_weight,
                        batch_number      = EXCLUDED.batch_number,
                        popularity_factor = EXCLUDED.popularity_factor,
                        n_seeds           = EXCLUDED.n_seeds,
                        added_at          = NOW()
            """, (
                req.spotify_id, r["name"], r.get("image"), r.get("listeners"), rank,
                r.get("sum_weight"), r.get("batch"), r.get("popularity_factor"),
                len(r.get("from", []))
            ))
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"[recs] DB save failed: {e}", flush=True)
        conn.rollback()

    # ── Strip internal fields ──
    for r in results:
        r.pop("batch", None)
        r.pop("sum_weight", None)
        r.pop("final_score", None)
        r.pop("popularity_factor", None)

    return JSONResponse(
        content={"recommendations": results},
        headers={"Cache-Control": "no-store"}
    )


class FeedbackRequest(BaseModel):
    spotify_id: str
    artist_name: str
    label: int  # 1 = like, 0 = dislike


@app.post("/feedback")
def post_feedback(req: FeedbackRequest):
    cur = get_cursor()
    cur.execute("""
        INSERT INTO feedback (spotify_user_id, artist_name, label)
        VALUES (%s, %s, %s)
        ON CONFLICT (spotify_user_id, artist_name) DO UPDATE
            SET label = EXCLUDED.label,
                created_at = NOW()
    """, (req.spotify_id, req.artist_name, req.label))
    conn.commit()
    cur.close()
    return {"ok": True}


@app.delete("/feedback")
def delete_feedback(req: FeedbackRequest):
    cur = get_cursor()
    cur.execute(
        "DELETE FROM feedback WHERE spotify_user_id = %s AND artist_name = %s",
        (req.spotify_id, req.artist_name)
    )
    conn.commit()
    cur.close()
    return {"ok": True}


@app.get("/history")
def get_recommendation_history(spotify_id: str):
    cur = get_cursor()
    cur.execute(
        "SELECT name, image_url, monthly_listeners FROM recommendations WHERE spotify_user_id = %s ORDER BY added_at DESC LIMIT 100",
        (spotify_id,)
    )
    recs = [dict(row) for row in cur.fetchall()]
    cur.close()
    return {"recommendations": recs}


class AlbumCreate(BaseModel):
    spotify_id: str
    name: str
    artists: list


@app.post("/albums")
def create_album(body: AlbumCreate):
    cur = get_cursor()
    cur.execute(
        "INSERT INTO albums (spotify_user_id, name, artists) VALUES (%s, %s, %s) RETURNING id, name, artists, created_at",
        (body.spotify_id, body.name, json.dumps(body.artists))
    )
    row = dict(cur.fetchone())
    conn.commit()
    cur.close()
    return row


@app.delete("/albums/{album_id}")
def delete_album(album_id: int, spotify_id: str):
    cur = get_cursor()
    cur.execute(
        "DELETE FROM albums WHERE id = %s AND spotify_user_id = %s",
        (album_id, spotify_id)
    )
    conn.commit()
    cur.close()
    return {"ok": True}


@app.get("/debug/artists")
def debug_artists(spotify_id: str):
    cached = _artists_cache.get(spotify_id)
    source = "memory"
    artists = cached["data"] if cached else []

    if not artists:
        cur = get_cursor()
        cur.execute("SELECT artists_data FROM users WHERE spotify_id = %s", (spotify_id,))
        row = cur.fetchone()
        cur.close()
        artists = row["artists_data"] if row and row["artists_data"] else []
        source = "db"

    total = len(artists)
    with_image = sum(1 for a in artists if a.get("image"))
    no_image = [{"id": a["id"], "name": a["name"], "sources": a.get("sources", [])} for a in artists if not a.get("image")]
    return {
        "source": source,
        "total": total,
        "with_image": with_image,
        "without_image": len(no_image),
        "refresh_in_progress": spotify_id in _refresh_in_progress,
        "no_image_artists": no_image[:20],
    }

@app.get('/ping')
def ping(): return {'pong': True}
