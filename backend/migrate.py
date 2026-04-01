import psycopg2
from dotenv import load_dotenv
import os
load_dotenv()
DB_PASSWORD = os.getenv("DB_PASSWORD")
# Use direct connection (port 5432) to bypass pgBouncer statement timeout
conn = psycopg2.connect(
    f"postgresql://postgres.ocebzopndfyhrnzujlwo:{DB_PASSWORD}@aws-1-us-east-1.pooler.supabase.com:5432/postgres",
    connect_timeout=15
)

cur = conn.cursor()

cols = [
    ("recommendations", "sum_weight", "FLOAT"),
    ("recommendations", "batch_number", "SMALLINT"),
    ("recommendations", "popularity_factor", "FLOAT"),
    ("recommendations", "n_seeds", "SMALLINT"),
    ("users", "xgb_model", "BYTEA"),
    ("users", "xgb_trained_at", "TIMESTAMPTZ"),
]

for table, col, typ in cols:
    try:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {typ}")
        conn.commit()
        print(f"{table}.{col} OK")
    except Exception as e:
        conn.rollback()
        print(f"{table}.{col} FAILED: {e}")

cur.close()
conn.close()
print("done")
