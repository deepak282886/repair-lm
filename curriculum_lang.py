"""
7-Stage Language Curriculum

Stage 1 — Atoms            : single words, basic co-occurrence
Stage 2 — Bigrams          : word pairs, simple structure
Stage 3 — Simple sentences : SVO, gym-generated
Stage 4 — Real text short  : 4-6 token windows, top 500 vocab
Stage 5 — Real text medium : 5-9 token windows, top 1000 vocab
Stage 6 — Real text long   : 6-12 token windows, top 2000 vocab
Stage 7 — Open             : 8-16 token windows, full vocab

Readiness thresholds calibrated to vocabulary size per stage.
"""

from core import Stage


def build_language_curriculum():
    return [
        # ── Stage 1: Atoms ─────────────────────────────────────────
        # ~40 words, 4-node sequences
        # Expect: ~5 chunks, 3 highways after 1000 ticks
        Stage(
            name             = "1_atoms",
            target_density   = 0.10,    # 4 chunks / 40 atoms
            target_highways  = 2,
            suggested_epsilon= 0.60,    # high exploration — graph is empty
            min_ticks        = 1000,
        ),

        # ── Stage 2: Bigrams ───────────────────────────────────────
        # ~65 words, 4-6 node sequences
        # Expect: ~15 chunks, 6 highways after 1500 ticks
        Stage(
            name             = "2_bigrams",
            target_density   = 0.18,
            target_highways  = 5,
            suggested_epsilon= 0.45,
            min_ticks        = 1500,
        ),

        # ── Stage 3: Simple sentences ──────────────────────────────
        # ~85 words, 5-8 node sequences
        # Expect: ~25 chunks, 10 highways after 2000 ticks
        Stage(
            name             = "3_sentences",
            target_density   = 0.25,
            target_highways  = 8,
            suggested_epsilon= 0.35,
            min_ticks        = 2000,
        ),

        # ── Stage 4: Real text short ───────────────────────────────
        # 500 word vocab, 4-6 token windows
        # Expect: ~80 chunks, 20 highways after 3000 ticks
        Stage(
            name             = "4_real_short",
            target_density   = 0.15,    # relative to 500 atoms
            target_highways  = 15,
            suggested_epsilon= 0.30,
            min_ticks        = 3000,
        ),

        # ── Stage 5: Real text medium ──────────────────────────────
        # 1000 word vocab, 5-9 token windows
        Stage(
            name             = "5_real_medium",
            target_density   = 0.12,
            target_highways  = 30,
            suggested_epsilon= 0.22,
            min_ticks        = 4000,
        ),

        # ── Stage 6: Real text long ────────────────────────────────
        # 2000 word vocab, 6-12 token windows
        Stage(
            name             = "6_real_long",
            target_density   = 0.10,
            target_highways  = 50,
            suggested_epsilon= 0.15,
            min_ticks        = 5000,
        ),

        # ── Stage 7: Open conversation ─────────────────────────────
        # Full vocab, 8-16 token windows
        # No density target — run for full min_ticks
        Stage(
            name             = "7_open",
            target_density   = 0.08,
            target_highways  = 80,
            suggested_epsilon= 0.10,
            min_ticks        = 6000,
        ),
    ]