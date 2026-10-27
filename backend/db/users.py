# backend/db/users.py
# ─────────────────────────────────────────────────────────
# All MongoDB operations related to users.
# The auth router imports and calls these functions.
# No route logic here — only database read/write operations.
# ─────────────────────────────────────────────────────────

import sys
import os
from datetime import datetime, timezone

# passlib handles password hashing using bcrypt algorithm
# bcrypt is one-way — you can never reverse it back to plain text
# This means even if the database is stolen, passwords are safe
from passlib.context import CryptContext

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings
from db.mongo import get_db

# ── Password hashing setup ────────────────────────────────
# CryptContext handles hashing and verification
# bcrypt is the algorithm — industry standard for passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """
    Converts plain text password to a bcrypt hash.
    Example: "admin123" → "$2b$12$eImiTXuWVxfM37uY4JANjQ..."
    The hash is different every time even for the same password.
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Checks if a plain text password matches a stored hash.
    Returns True if match, False if not.
    """
    return pwd_context.verify(plain_password, hashed_password)


# ── Database operations ───────────────────────────────────

async def get_user_by_username(username: str) -> dict | None:
    """
    Finds a user in MongoDB by username.
    Returns the full user document or None if not found.

    The document looks like:
    {
        "_id": ObjectId(...),
        "username": "admin",
        "password": "$2b$12$...",   ← hashed
        "name": "Administrator",
        "role": "admin",
        "created_at": datetime(...)
    }
    """
    database = get_db()
    # find_one returns None automatically if no match
    user = await database.users.find_one({"username": username.lower().strip()})
    return user


async def create_user(username: str, password: str, name: str, role: str = "user") -> dict:
    """
    Creates a new user in MongoDB.
    Password is hashed before storing — plain text never touches the database.

    Returns:
        dict with success status and message
    """
    database = get_db()

    # Check if username already exists
    existing = await get_user_by_username(username)
    if existing:
        return {"success": False, "message": "Username already exists."}

    # Build the user document
    user_doc = {
        "username": username.lower().strip(),
        "password": hash_password(password),   # ← hashed, never plain text
        "name": name.strip(),
        "role": role,
        "created_at": datetime.now(timezone.utc),
    }

    await database.users.insert_one(user_doc)
    return {"success": True, "message": f"User '{username}' created successfully."}


async def delete_user(username: str) -> dict:
    """
    Deletes a user from MongoDB by username.
    The default admin account cannot be deleted.
    """
    if username.lower() == "admin":
        return {"success": False, "message": "Cannot delete the default admin account."}

    database = get_db()
    result = await database.users.delete_one({"username": username.lower()})

    if result.deleted_count == 0:
        return {"success": False, "message": "User not found."}

    return {"success": True, "message": f"User '{username}' deleted."}


async def list_all_users() -> list:
    """
    Returns all users as a list of dicts.
    Password field is excluded — never expose hashes in responses.
    """
    database = get_db()

    # The second argument to find() is a projection
    # 0 means exclude that field, 1 means include
    # We exclude password and MongoDB's internal _id
    cursor = database.users.find(
        {},                                    # empty filter = all documents
        {"password": 0, "_id": 0}             # exclude password and _id
    )
    users = await cursor.to_list(length=100)  # max 100 users
    return users


async def authenticate_user(username: str, password: str) -> dict | None:
    """
    The main login function.
    Finds the user then checks if the password matches the hash.

    Returns:
        Full user dict (without password) if credentials are correct
        None if username not found or password is wrong
    """
    user = await get_user_by_username(username)

    # User not found
    if not user:
        return None

    # Password doesn't match the stored hash
    if not verify_password(password, user["password"]):
        return None

    # Return user info without the password field
    return {
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
    }


async def seed_default_users():
    """
    Creates default admin and user accounts if they don't exist.
    Called once on app startup.
    This ensures the app always has at least one admin account.

    Default credentials:
        admin / admin123  (role: admin)
        user1 / user123   (role: user)

    Does nothing when settings.SEED_DEFAULT_USERS is false — see that
    setting in config.py for why a deployment would want that off. The
    check lives HERE rather than at the call site in main.py so that any
    future caller is covered by it too; a gate that only guards one call
    is one refactor away from being bypassed.
    """
    if not settings.SEED_DEFAULT_USERS:
        print("⏭️  SEED_DEFAULT_USERS is false — skipping default user seeding")
        return

    database = get_db()
    count = await database.users.count_documents({})

    # Only seed if database is completely empty
    if count == 0:
        defaults = [
            {
                "username": "admin",
                "password": hash_password("admin123"),
                "name": "Administrator",
                "role": "admin",
                "created_at": datetime.now(timezone.utc),
            },
            {
                "username": "user1",
                "password": hash_password("user123"),
                "name": "Govt Official",
                "role": "user",
                "created_at": datetime.now(timezone.utc),
            },
        ]
        await database.users.insert_many(defaults)
        print("[OK] Default users seeded - admin/admin123 and user1/user123")
    else:
        print(f"[OK] Users collection has {count} existing user(s) - skipping seed")