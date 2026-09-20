
### Notes

I am trying to build an agentic system that can efficiently navigate a knowledge graph and answer questions based on deep relations and heuristics

## Ollama setup

```
ollama run qwen2.5-coder:7b
```

### Neo4j connection

The graph tools query a live Neo4j database over Bolt; there is no in-memory fixture.
Configure the process with these environment variables (a `.env` file is supported):

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j
# Optional: human-readable node identifier property (default: name)
NEO4J_NODE_ID_PROPERTY=name
```

For the Docker image in this repository, `NEO4J_AUTH=neo4j/password` is also
accepted as a convenience; prefer `NEO4J_PASSWORD` for an external instance.

`search_nodes` returns a `node_id` plus Neo4j's `element_id`; either can be supplied
to `get_neighbors` and `get_node_details`. Neighbor results include both directions.

### Load the example graph

The original example companies, product, supplier, and four relationships can be
loaded into your configured database with:

```
python seed_graph.py
```

The command is idempotent: it adds a `KnowledgeGraphNode` label, creates a unique
`name` constraint for that label, and uses `MERGE`, so it is safe to run again.

### Optional local Neo4j container

Build and start a fresh local database with the default credentials
`neo4j/password`:

```
docker build -t local-neo4j .
docker run --name neo4j-local -d -p 7474:7474 -p 7687:7687 local-neo4j
```

Neo4j stores its password in the container's `/data` volume on first startup.
Changing `NEO4J_AUTH` afterward does **not** change an existing password. For this
example, if no graph data needs to be retained, recreate the local container and
its anonymous volumes to reset it to the Dockerfile default:

```
docker rm -fv neo4j-local
docker run --name neo4j-local -d -p 7474:7474 -p 7687:7687 local-neo4j
```

Then configure the client and load the example graph:

```
NEO4J_AUTH=neo4j/password python seed_graph.py
```


### Cypher query

Return all nodes and relationship

```sql
MATCH (n)
OPTIONAL MATCH (n)-[r]->(m)
RETURN n, r, m
```

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