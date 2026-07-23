# Re-Pair Language Model

A grammar-compression-based language model built on the Re-Pair Forest architecture. No neural networks — the model learns structure entirely from data through joint Re-Pair compression.

## Architecture

Query resolution runs in order of decreasing confidence:

| Stage | Component | Description |
|---|---|---|
| 1 | `memory.py` | Exact normalised question → answer lookup |
| 2 | `repair_forest.py` | Entity-driven multi-hop composition |
| 3 | `soft_match.py` | Content-word overlap + entity bonus + predicate gate |
| 4 | `discovery_graph.py` | Word-overlap similarity search (last resort) |
| 5 | `repair_grammar.py` | Re-Pair grammar generation (open-ended) |

### Modules

- **`repair_grammar.py`** — Joint Re-Pair compression engine. Repeatedly replaces the most frequent adjacent token pair with a new non-terminal. Running jointly over the whole corpus means shared phrasings (e.g. "what nationality was", "who wrote") become reusable non-terminals. Rule frequencies are preserved as generation probabilities.

- **`memory.py`** — Append-only exact Q/A table. Normalised lookup (lowercase, strip punctuation). Never overwrites stored facts.

- **`entity_extractor.py`** — Identifies entity spans: capitalised proper-noun spans including multi-word names with lowercase particles (da, van, von), with fallback to last content word for lowercase objects like "penicillin".

- **`fact_corpus.py`** — Manages (template, value) pairs derived from Q/A facts. Template = question with entity abstracted to `E`. Value = novel entity the answer introduces.

- **`relation_clustering.py`** — Data-driven DIRT-style clustering. Two templates are merged when they produce overlapping answer values across the corpus. Union-find based. Requires ~50+ facts for reliable signal.

- **`predicate_gate.py`** — Predicate mismatch penalty. Hand-written keyword → group dictionary (production default) with optional data-driven clustering overlay. Suppresses false positives like "who discovered X" matching "who composed X".

- **`soft_match.py`** — Scored approximate matching: Jaccard similarity + entity-presence bonus × predicate-mismatch penalty. Gated by absolute threshold and margin over second-best candidate.

- **`repair_forest.py`** — Multi-hop composition pipeline. Order-agnostic clause splitting, novelty-based entity selection, recursive substitution. Supports 3+ hop chains.

- **`discovery_graph.py`** — Word-overlap similarity search over all memorised questions. Last resort before returning no answer.

- **`model.py`** — Top-level model wiring all components together.

## Training

### Pretraining

Runs joint Re-Pair compression over a large text corpus (FineWeb-Edu by default):

```bash
pip install -r requirements.txt
python pretrain.py
python pretrain.py --sentences 50000 --rules 2000 --save pretrained.pkl
```

More data → more rule reuse → richer non-terminals → better generation.

### Finetuning

Loads QA/instruction pairs, populates memory, runs relation clustering:

```bash
python finetune.py --load pretrained.pkl --save finetuned.pkl
python finetune.py --load pretrained.pkl --dataset tatsu-lab/alpaca --pairs 5000
```

Relation clustering runs automatically once the corpus reaches 50+ facts.

## Generation

```bash
python generate.py --load finetuned.pkl
python generate.py --load finetuned.pkl --prompt "Who discovered penicillin"
python generate.py --load finetuned.pkl --mode generate --batch 10
```

**Modes:**
- `qa` — full resolution pipeline (memory → composition → soft match → discovery → grammar)
- `generate` — grammar-only open-ended generation
- `both` — both outputs side by side

## Scaling

The system improves purely through data:

- **More sentences** → more frequent pairs → richer grammar rules → better generation
- **More facts** → larger discovery graph → better fallback coverage
- **More diverse facts** → more shared-value evidence → more relation cluster merges → better predicate gating

## Recommended workflow

```bash
# 1. Pretrain on web text (builds the grammar)
python pretrain.py --sentences 50000 --rules 2000 --save pretrained.pkl

# 2. Finetune on instruction pairs (builds memory + relation structure)
python finetune.py --load pretrained.pkl --dataset tatsu-lab/alpaca --pairs 5000 --save finetuned.pkl

# 3. Generate
python generate.py --load finetuned.pkl
```

## Install as package

```bash
pip install -e .
```

## Requirements

- Python 3.10+
- `datasets` (HuggingFace)
- `huggingface-hub`
