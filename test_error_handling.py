#!/usr/bin/env python
"""Test that 410 errors are properly handled with clear messages."""
from backend.rag.model import get_llm

print("Testing LLM error handling...")
print("=" * 60)

try:
    llm = get_llm()
    print(f"[OK] LLM initialized: {type(llm).__name__}")
except ValueError as e:
    print("[ERROR] LLM initialization failed:")
    print("-" * 60)
    print(str(e))
    print("-" * 60)
    print("\n[INFO] This is expected if HuggingFace models return 410 Gone")
    print("[INFO] Solution: Install Ollama (see FIX_LLM_ERROR.md)")
except Exception as e:
    print(f"[UNEXPECTED ERROR] {type(e).__name__}: {e}")

print("=" * 60)


