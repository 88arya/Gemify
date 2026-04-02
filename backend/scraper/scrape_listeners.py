import requests
from bs4 import BeautifulSoup
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv
import os
import re
import time

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

DB_PASSWORD = os.getenv("DB_PASSWORD")  # .env locally, GitHub secret in CI
DATABASE_URL = f"postgresql://postgres.ocebzopndfyhrnzujlwo:{DB_PASSWORD}@aws-1-us-east-1.pooler.supabase.com:6543/postgres"

def scrape_page(url, retries=3):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            r.raise_for_status()
            r.encoding = 'utf-8'
            soup = BeautifulSoup(r.text, "html.parser")
            rows = []
            for tr in soup.select("table tbody tr"):
                cols = tr.find_all("td")
                if len(cols) < 3:
                    continue
                link = cols[1].find("a")
                if not link:
                    continue
                name = link.get_text(strip=True)
                href = link.get("href", "")
                m = re.search(r"artist/([^_]+)_songs\.html", href)
                spotify_id = m.group(1) if m else None
                listeners_raw = cols[2].get_text(strip=True).replace(",", "")
                if not listeners_raw.isdigit():
                    continue
                rows.append((name, spotify_id, int(listeners_raw)))
            return rows
        except Exception as e:
            print(f"  Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    print(f"  Skipping {url} after {retries} failed attempts")
    return []

def main():
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=15)
    cur = conn.cursor()

    # supabase default is lower
    cur.execute("SET statement_timeout = '60s'")
    conn.commit()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS artist_listeners (
            spotify_id  text PRIMARY KEY,
            name        text NOT NULL,
            rank        int NOT NULL,
            listeners   bigint NOT NULL,
            updated_at  timestamptz DEFAULT NOW()
        )
    """)
    conn.commit()

    all_rows = []
    urls = ["https://kworb.net/spotify/listeners.html"] + \
           [f"https://kworb.net/spotify/listeners{i}.html" for i in range(2, 11)]

    for i, url in enumerate(urls, start=1):
        print(f"Scraping page {i}/10: {url}")
        rows = scrape_page(url)
        print(f"  Got {len(rows)} rows")
        all_rows.extend(rows)
        time.sleep(2)

    records = [
        {"spotify_id": spotify_id, "name": name, "rank": rank, "listeners": listeners}
        for rank, (name, spotify_id, listeners) in enumerate(all_rows, start=1)
        if spotify_id
    ]

    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            execute_values(cur, """
                INSERT INTO artist_listeners (spotify_id, name, rank, listeners, updated_at)
                VALUES %s
                ON CONFLICT (spotify_id) DO UPDATE
                    SET name       = EXCLUDED.name,
                        rank       = EXCLUDED.rank,
                        listeners  = EXCLUDED.listeners,
                        updated_at = NOW()
            """, [(r["spotify_id"], r["name"], r["rank"], r["listeners"]) for r in batch],
            template="(%s, %s, %s, %s, NOW())")
            conn.commit()
            print(f"  Inserted {min(i + batch_size, len(records))}/{len(records)}")
        except Exception as e:
            print(f"  Batch {i}-{i+batch_size} failed: {e}")
            conn.rollback()

    cur.close()
    conn.close()
    print(f"Done — upserted {len(records)} artists")

if __name__ == "__main__":
    main()
