from langchain_core.tools import tool
from knowledge_graph import KNOWLEDGE_GRAPH
import json

# Define standard graph tools accessible to the agent zero-shot
@tool
def search_nodes(query: str) -> str:
    """Finds nodes by name and by values stored in the graph, including relation labels and properties."""
    if not query or not query.strip():
        return json.dumps({"matched_nodes": [], "message": "Empty search query."})

    normalized = query.strip().lower()
    normalized = normalized.replace("acquisitions", "acquired").replace("acquired by", "acquired")
    tokens = {part for part in normalized.replace("-", " ").split() if part}
    if not tokens:
        return json.dumps({"matched_nodes": [], "message": f"No nodes found matching '{query}'"})

    matches = []
    seen = set()

    for node_id, node_data in KNOWLEDGE_GRAPH.items():
        haystack = [
            node_id,
            node_id.lower(),
            node_data.get("type", ""),
        ]
        haystack.extend(str(value).lower() for value in node_data.get("properties", {}).values())
        for edge in node_data.get("edges", []):
            haystack.append(str(edge.get("relation", "")).lower())
            haystack.append(str(edge.get("target", "")).lower())

        combined = " ".join(haystack).lower()
        if any(token in combined for token in tokens):
            if node_id not in seen:
                matches.append(node_id)
                seen.add(node_id)

    if matches:
        return json.dumps({"matched_nodes": matches})
    return json.dumps({"matched_nodes": [], "message": f"No nodes found matching '{query}'"})

@tool
def get_neighbors(node_id: str) -> str:
    """Retrieves 1-hop outgoing connections/relationships and targets for a specific node."""
    if node_id not in KNOWLEDGE_GRAPH:
        return f"Error: Node '{node_id}' does not exist in the graph."
    
    node_data = KNOWLEDGE_GRAPH[node_id]
    return json.dumps({
        "node": node_id,
        "type": node_data["type"],
        "edges": node_data["edges"]
    })

@tool
def get_node_details(node_id: str) -> str:
    """Fetches the internal properties and attributes of a specific node."""
    if node_id not in KNOWLEDGE_GRAPH:
        return f"Error: Node '{node_id}' does not exist in the graph."
    
    return json.dumps({
        "node": node_id,
        "properties": KNOWLEDGE_GRAPH[node_id]["properties"]
    })