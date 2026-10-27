# backend/db/mongo.py
# ─────────────────────────────────────────────────────────
# MongoDB connection manager.
# Local MongoDB — no SSL/TLS needed.
# ─────────────────────────────────────────────────────────

import sys
import os
from motor.motor_asyncio import AsyncIOMotorClient

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

client: AsyncIOMotorClient = None
db = None


async def connect_db():
    global client, db
    client = AsyncIOMotorClient(settings.MONGODB_URL)
    db = client[settings.DATABASE_NAME]
    await client.admin.command("ping")
    print(f"[OK] Connected to MongoDB - database: '{settings.DATABASE_NAME}'")


async def close_db():
    global client
    if client:
        client.close()
        print("[INFO] MongoDB connection closed.")


def get_db():
    return db