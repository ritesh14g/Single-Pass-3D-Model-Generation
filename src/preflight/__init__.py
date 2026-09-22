"""Stage 0 — input check: prove the input can produce every output before spending the budget.

``analyze_input`` (``analyzer.py``) reads the video and its telemetry in seconds and returns
PASS / WARN / BLOCK per check with a fix for each problem. The pipeline runs it first and
stops on BLOCK (unless told to accept the input anyway).
"""

from src.preflight.analyzer import InputReport, InputRejected, analyze_input  # noqa: F401
