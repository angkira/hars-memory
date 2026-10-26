"""Shadows the real tiktoken for SUBPROCESSES of the test suite (put first on PYTHONPATH by conftest.py):
importing it fails, so every script falls back to the chars/3.6 estimator, never to a locally cached or
downloaded encoding."""
raise ImportError("tiktoken disabled for the offline test suite (tests/conftest.py)")
