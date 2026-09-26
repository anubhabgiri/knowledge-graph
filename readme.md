
### Notes

I am trying to build an agentic system that can efficiently navigate a knowledge graph and answer questions based on deep relations and heuristics




### Schema Free Knowledge Graph Extraction

Extract, Define and Canonicalize

The entire process is based on this [paper](https://arxiv.org/pdf/2404.03868)

## Pipeline Overview

The notebook shows three distinct LLM-based extraction steps:

```
[Raw Text]
    │
    ▼ Chunking
[Text Chunks]
    │
    ▼ Step 1 — Open Triplet Extraction
[Raw Triplets: (Subject, Relation, Object)]
    │
    ▼ Step 2 — Relation Definition Generation
[Triplets + Definitions]
    │
    ▼ Step 3 — Relation Canonicalization (via vector similarity)
[Canonical Triplets]
    │
    ▼ Merge & Deduplicate
[Final Knowledge Graph → output.json]
```

---

## Chunking Strategy

Large texts cannot be sent wholesale to an LLM due to context limits and for quality reasons (the LLM should stay focused). The plan is a **sliding window chunker with sentence-aware boundaries**:

| Parameter | Default | Notes |
|---|---|---|
| `chunk_size` | 512 tokens | Fits comfortably in most model context windows alongside the system/user prompt |
| `chunk_overlap` | 64 tokens | Preserves cross-sentence context at chunk boundaries |
| Boundary type | Sentence | Uses `nltk.sent_tokenize` so chunks never split mid-sentence |

The `chunk_overlap` window ensures that entities/relations that span a chunk boundary (e.g., a pronoun resolved in the next sentence) are still captured.

---

## Pipeline Usage

```bash
# Gemini (default)
python creation/without-schema/main.py --input doc.txt --provider gemini --model gemini-2.5-flash

# OpenAI
python creation/without-schema/main.py --input doc.txt --provider openai --model gpt-4o

# Ollama (local)
python creation/without-schema/main.py --input doc.txt --provider ollama --model llama3.2
```

```bash
# For general domain text (default)
python creation/without-schema/main.py -i input.txt -o output.json --domain general

# For biomedical / BC5CDR text
python creation/without-schema/main.py -i pubmed_abstract.txt -o output.json --domain biomedical
```


## Directory Structure 

```text
evaluation/
├── __init__.py
├── models.py                  # Ground-truth, sample, match, and report data models
├── matcher.py                 # Hungarian maximum-weight matching + entity & predicate similarity
├── metrics.py                 # Micro/Macro Precision, Recall, F1 & report formatting
├── runner.py                  # Benchmark orchestrator: dataset loader -> pipeline -> scorer
├── main.py                    # CLI entrypoint for running benchmarks
├── data/                      # Bundled sample benchmark datasets (zero-download testing)
│   ├── bc5cdr_sample.pubtator # Curated BC5CDR PubMed abstracts with gold CID relations
│   └── webnlg_sample.json     # Curated WebNLG general-domain RDF instances
└── datasets/                  # Extensible benchmark loaders
    ├── base.py                # Abstract BaseDatasetLoader
    ├── bc5cdr.py              # BioCreative V PubTator format parser + MeSH mapper
    └── webnlg.py              # WebNLG JSON format loader
```

### Evaluation Pipeline Usage

```bash
# Evaluate on BC5CDR samples with your local Ollama model (no API costs)
.venv/bin/python evaluation/main.py --dataset bc5cdr --limit 3 --provider ollama --model qwen2.5-coder:7b

# Or with Gemini:
.venv/bin/python evaluation/main.py --dataset bc5cdr --limit 3 --provider gemini --model gemini-2.5-flash

.venv/bin/python evaluation/main.py --dataset webnlg --limit 2 --provider ollama --model qwen2.5-coder:7b

# Custom dataset
.venv/bin/python evaluation/main.py \
  --dataset bc5cdr \
  --data-path /path/to/CDR_TestSet.PubTator.txt \
  --limit 10 \
  --threshold 0.70 \
  --output output/bc5cdr_eval_report.json

  Options:
  -i, --input FILE                Path to the input text file.  [required]
  -o, --output PATH               Destination path for the output JSON file.
                                  [default: output.json]
  --chunk-size INTEGER            Maximum tokens per chunk.  [default: 512]
  --chunk-overlap INTEGER         Overlap tokens between consecutive chunks.
                                  [default: 64]
  --top-k INTEGER                 Number of canonical-relation candidates
                                  surfaced per LLM call.  [default: 5]
  --min-similarity FLOAT          Minimum cosine similarity threshold to
                                  consider candidate relations for
                                  canonicalization.  [default: 0.45]
  -d, --domain [general|biomedical]
                                  Domain preset for prompts and reasoning
                                  rules ('general' or 'biomedical').
                                  [default: general]
  --provider [gemini|openai|ollama]
                                  LLM provider to use for all pipeline stages.
                                  [default: gemini]
  --model TEXT                    Model identifier for the chosen provider.
                                  Defaults: gemini→gemini-2.5-flash,
                                  openai→gpt-4o-mini, ollama→llama3.2
  --help                          Show this message and exit.
```

