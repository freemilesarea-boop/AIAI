"""Audio post-processing worker service.

Phase 0: queue plumbing only. Phase 4 implements decode → validation →
resample → WAV/MP3 encode → SHA256 → upload. No destructive mastering
in the MVP.
"""
