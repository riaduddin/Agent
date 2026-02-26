import asyncio
from collections import defaultdict
import os
import logging

logger = logging.getLogger(__name__)

# Global token tracker (keyed by p_id)
_token_tracker = defaultdict(lambda: {
    "input_tokens": 0,
    "output_tokens": 0,
    "thoughts_tokens": 0
})

# Locks for thread-safe access to token tracker per p_id
_token_locks = defaultdict(lambda: asyncio.Lock())

token_api_base_url = os.getenv("TOKEN_API_BASE_URL")

async def get_token_tracker(p_id: str):
    """Get token tracker for a specific p_id (thread-safe)"""
    async with _token_locks[p_id]:
        return _token_tracker[p_id].copy()  # Return a copy to avoid external modification

async def add_tokens(p_id: str, input_tokens: int = 0, output_tokens: int = 0, thoughts_tokens: int = 0):
    """Add tokens to the tracker for a specific p_id (thread-safe)"""
    # Safety: ensure we don't try to add None
    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    thoughts_tokens = thoughts_tokens or 0
    
    async with _token_locks[p_id]:
        tracker = _token_tracker[p_id]
        tracker["input_tokens"] += input_tokens
        tracker["output_tokens"] += output_tokens
        tracker["thoughts_tokens"] += thoughts_tokens

async def get_total_tokens(p_id: str):
    """Get total tokens for a specific p_id (thread-safe)"""
    async with _token_locks[p_id]:
        tracker = _token_tracker[p_id]
        return {
            "input_tokens": tracker["input_tokens"],
            "output_tokens": tracker["output_tokens"],
            "thoughts_tokens": tracker["thoughts_tokens"],
            "total_tokens": tracker["input_tokens"] + tracker["output_tokens"]
        }

async def reset_token_tracker(p_id: str):
    """Reset token tracker for a specific p_id (thread-safe)"""
    async with _token_locks[p_id]:
        _token_tracker[p_id] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "thoughts_tokens": 0
        }
