
### Notes

I am trying to build an agentic system that can efficiently navigate a knowledge graph and answer questions based on deep relations and heuristics

## Ollama setup

```
ollama run qwen2.5-coder:7b
```

### Neo4j setup
```
docker build -t local-neo4j .
docker run --name neo4j-local -d -p 7474:7474 -p 7687:7687 local-neo4j
```