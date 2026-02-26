
import sqlalchemy
from sqlalchemy import create_engine

db_url = "postgresql://postgres.xnrnzposzbmvpettfxpu:45FDG59LMz1z35t5@aws-0-us-west-2.pooler.supabase.com:5432/postgres?connect_timeout=10&sslmode=require"

try:
    print(f"Testing URL: {db_url}")
    engine = create_engine(db_url)
    connection = engine.connect()
    print("✅ Success!")
    connection.close()
except Exception as e:
    print(f"❌ Failed: {e}")

db_url_psycopg2 = "postgresql+psycopg2://postgres.xnrnzposzbmvpettfxpu:45FDG59LMz1z35t5@aws-0-us-west-2.pooler.supabase.com:5432/postgres?connect_timeout=10&sslmode=require"

try:
    print(f"\nTesting URL with psycopg2: {db_url_psycopg2}")
    engine = create_engine(db_url_psycopg2)
    connection = engine.connect()
    print("✅ Success!")
    connection.close()
except Exception as e:
    print(f"❌ Failed: {e}")

db_url_asyncpg = "postgresql+asyncpg://postgres.xnrnzposzbmvpettfxpu:45FDG59LMz1z35t5@aws-0-us-west-2.pooler.supabase.com:5432/postgres?connect_timeout=10&sslmode=require"
# Note: connect() won't work easily with asyncpg in a sync script, but create_engine should at least not fail immediately if the scheme is recognized.
try:
    print(f"\nTesting URL with asyncpg (create_engine only): {db_url_asyncpg}")
    engine = create_engine(db_url_asyncpg)
    print("✅ create_engine Success!")
except Exception as e:
    print(f"❌ Failed: {e}")
