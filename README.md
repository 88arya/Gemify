# Gemify

Spotify-based artist recommendation engine.

---

## What it does

You log in with Spotify. The app scores your listening history across multiple time ranges and lets you seed a recommendation engine with artists you pick. It builds a 3-layer similarity graph via Last.fm and returns 5 ranked artist recommendations. A feedback system (like/dislike) trains an XGBoost model to personalize results over time.

---

## Tech stack

- **Frontend:** Vanilla JS / HTML / CSS
- **Backend:** FastAPI (Python)
- **Database:** PostgreSQL via Supabase
- **APIs:** Spotify, Last.fm, MusicBrainz
- **ML:** XGBoost, NetworkX
- **Scraper:** BeautifulSoup (kworb.net listener counts)

---

## How the recommendation works

Starting from your selected seed artists, the engine runs 3 batches of Last.fm similar-artist lookups, building a directed graph where edge weights decay by batch depth. Each candidate is scored by aggregating weighted path strengths, then adjusted by a popularity factor derived from a Gaussian curve over monthly listener counts — this pushes results toward artists with similar popularity to your seeds rather than always surfacing the most-streamed names.

Once you've liked or disliked 20+ artists, an XGBoost classifier trained on your feedback history adjusts each score by a learned probability multiplier. Below 20 ratings, the raw graph score is used as-is (cold start fallback). Between 20–49, a blended multiplier softens the ML influence until there's enough signal.

---

## Setup

```bash
git clone https://github.com/88arya/Gemify.git
cd Gemify
python -m pip install -r requirements.txt
```

Create a `.env` file in the root directory:

```
DB_PASSWORD=your_supabase_password
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
LASTFM_API_KEY=your_lastfm_api_key
```

Run the server:

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open: [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app)

---

## Environment variables

| Variable | Description |
|---|---|
| `DB_PASSWORD` | Supabase PostgreSQL password |
| `SPOTIFY_CLIENT_ID` | Spotify developer app client ID |
| `SPOTIFY_CLIENT_SECRET` | Spotify developer app client secret |
| `LASTFM_API_KEY` | Last.fm API key |

---

## Repo structure

```
Gemify/
├── backend/
│   ├── main.py                    FastAPI app — routes, recommendation engine, ML pipeline
│   ├── db.py                      Supabase connection
│   ├── spotify_oauth.py           Spotify OAuth flow
│   └── scraper/
│       └── scrape_listeners.py    kworb.net monthly listener scraper (runs via CI)
├── frontend/
│   └── index.html                 single-page app (vanilla JS)
└── .github/
    └── workflows/
        └── scrape_listeners.yml   monthly cron job (1st of each month)
```

---

## Notes

- XGBoost personalisation kicks in after 20 liked/disliked artists
- Monthly listener data is scraped and cached via a GitHub Actions cron job
- Single-user design — built for personal local deployment
