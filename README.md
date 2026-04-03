# Gemify

Gemify is an artist recommendation web app focused on discovering underground artists tailored to your taste.  Unlike Spotify, which largely promotes popular or trending artists, Gemify prioritizes lesser-known artists that closely match your listening preferences. As you interact with recommendations, the algorithm learns your preferences over time and continuously improves future suggestions.

## How It Works

1. Authenticate with Spotify to retrieve your top artists and build your initial taste profile.
2. Select seed artists that best represent your music preferences.
3. Candidates are generated in stages, first finding artists similar to your seeds, then exploring artists similar to those results.
4. Candidates are ranked using a combination of similarity and popularity.
5. After 20 likes or dislikes, recommendation weights are constantly adjusted to better match your taste.

## Tech Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI + Uvicorn |
| Database | Supabase |
| Recommendation Algorithm | NetworkX + XGBoost |
| APIs | Spotify, Last.fm, MusicBrainz |
| Frontend | Vanilla JS |
| Scraping | Beautiful Soup |

## Getting Started

> [!NOTE]
> As of Feb 2026, Spotify has restricted the ability to deploy web applications. As a result, Gemify must be run locally and requires Spotify Premium.

### Prerequisites

- Python 3.10+
- Supabase database
- API credentials for Spotify and Last.fm

### Install dependencies

```bash
pip install -r requirements.txt
```

### Environment variables

Create `backend/.env`:

```env
SPOTIFY_CLIENT_ID=
SPOTIFY_CLIENT_SECRET=
SPOTIFY_REDIRECT_URI=
LASTFM_API_KEY=
DB_PASSWORD=
```

### Run

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/app` in your browser and log in with Spotify.

## Repository Layout

```
Gemify/
├── backend/
│   ├── main.py              
│   ├── db.py                
│   ├── migrate.py          
│   └── scraper/
│       └── scrape_listeners.py  
├── frontend/
│   └── index.html           
└── requirements.txt
```
