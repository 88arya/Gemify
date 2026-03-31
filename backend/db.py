import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DB_PASSWORD = os.getenv("DB_PASSWORD")
DATABASE_URL = f"postgresql://postgres.ocebzopndfyhrnzujlwo:{DB_PASSWORD}@aws-1-us-east-1.pooler.supabase.com:6543/postgres"

conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
print("Connected to Supabase")
