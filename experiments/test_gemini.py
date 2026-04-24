#!/usr/bin/env python3
"""Test script for Gemini API."""

import os
from pathlib import Path

for env_path in [
    Path(__file__).parent / ".env",
    Path(__file__).parent.parent / ".env",
]:
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())
        break

from google import genai

client = genai.Client()

response = client.models.generate_content(
    model="gemini-2.0-flash-001", contents="Explain how AI works in a few words"
)
print(response.text)
