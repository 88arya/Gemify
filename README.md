# Gemify

Gemify is a music discovery app that finds artists you'll actually like — not just more of what you already listen to. Unlike Spotify's recommendation algorithm, which optimizes for engagement and familiar sounds, Gemify uses a graph-based pipeline seeded by your own taste to surface genuinely new artists, then gets sharper over time as you give feedback.

## How It Works

1. **Authenticate** with Spotify — your followed artists and listening history are used to build your taste profile.
2. **Select seed artists** from your top and followed artists.
3. **Recommendation pipeline** expands outward in two graph batches using Last.fm similarity data:
   - **Batch 1:** Similar artists to your seeds, aggregated and weighted by similarity score
   - **Batch 2:** Similar artists to the Batch 1 survivors, filtered for novelty
4. **Popularity scoring** ranks results using a Gaussian curve that peaks around 2M Spotify listeners — rewarding artists who are popular enough to be worth discovering, but not already everywhere.
5. **Personalization** kicks in after 20+ likes/dislikes — a per-user XGBoost model re-ranks future recommendations based on your feedback history.
6. **Save artists** you want to revisit into named albums.

## Tech Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI + Uvicorn |
| Database | PostgreSQL (Supabase) |
| Graph | NetworkX (directed graph for recommendation pipeline) |
| ML | XGBoost (per-user classifier) |
| APIs | Spotify, Last.fm, MusicBrainz, kworb.net (scraped) |
| Frontend | Vanilla JS, single-page app |

## Getting Started

> [!NOTE]
> As of Feb 2026, Spotify has restricted the ability to deploy web applications. As a result, Gemify must be run locally and requires Spotify Premium.

### Prerequisites

- Python 3.10+
- A Supabase PostgreSQL database
- API credentials for Spotify and Last.fm

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Create `backend/.env`:

```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8000/callback
LASTFM_API_KEY=your_lastfm_api_key
DB_PASSWORD=your_supabase_db_password
```

### Run

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/app` in your browser and log in with Spotify.

## Repository Layout

```
Gemify/
├── backend/
│   ├── main.py              # FastAPI app — all endpoints and pipeline logic
│   ├── db.py                # Database connection
│   ├── migrate.py           # Schema migration utility
│   └── scraper/
│       └── scrape_listeners.py  # kworb.net listener count scraper
├── frontend/
│   └── index.html           # Single-page app (vanilla JS)
└── requirements.txt
```
