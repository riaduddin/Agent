
import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Use the URL provided in the user request, but modified for asyncpg
db_url = "postgresql+asyncpg://postgres.xnrnzposzbmvpettfxpu:45FDG59LMz1z35t5@aws-0-us-west-2.pooler.supabase.com:5432/postgres"

async def test_connection():
    print(f"Testing Async URL: {db_url}")
    try:
        engine = create_async_engine(db_url)
        print("Engine created. Attempting connection...")
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            print(f"✅ Connection successful! Result: {result.scalar()}")
        await engine.dispose()
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_connection())
