import networkx as nx
from pydantic import BaseModel, Field
from typing import List, Literal

# ==========================================
# 1. Define the Predefined Schema
# ==========================================

# Restrict the types of entities the graph can contain
EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT"]

# Restrict the types of relationships allowed between entities
RelationType = Literal["FOUNDED_BY", "CEO_OF", "HEADQUARTERED_IN", "PRODUCES"]

class Node(BaseModel):
    id: str = Field(description="Unique identifier for the entity (e.g., 'Apple_Inc')")
    label: str = Field(description="Display name of the entity")
    type: EntityType = Field(description="The category of the entity")

class Edge(BaseModel):
    source: str = Field(description="The ID of the source node")
    target: str = Field(description="The ID of the target node")
    type: RelationType = Field(description="The type of relationship")

class KnowledgeGraphSchema(BaseModel):
    nodes: List[Node]
    edges: List[Edge]

# ==========================================
# 2. the Extraction Process
# ==========================================

def extract_knowledge_graph(text: str) -> KnowledgeGraphSchema:
    """
    In a real-world scenario, you would pass the text and the Pydantic schema 
    to an LLM (e.g., OpenAI, Anthropic) using structured outputs or function calling.
    For this runnable example, we simulate the LLM's JSON response based on the input text.
    """
    
    # Simulating an LLM parsing the text: 
    # "Tim Cook is the CEO of Apple. Apple is headquartered in Cupertino and produces the iPhone."
    
    simulated_llm_response = {
        "nodes": [
            {"id": "tim_cook", "label": "Tim Cook", "type": "PERSON"},
            {"id": "apple", "label": "Apple", "type": "ORGANIZATION"},
            {"id": "cupertino", "label": "Cupertino", "type": "LOCATION"},
            {"id": "iphone", "label": "iPhone", "type": "PRODUCT"}
        ],
        "edges": [
            {"source": "tim_cook", "target": "apple", "type": "CEO_OF"},
            {"source": "apple", "target": "cupertino", "type": "HEADQUARTERED_IN"},
            {"source": "apple", "target": "iphone", "type": "PRODUCES"}
        ]
    }
    
    # Enforce the predefined schema on the extracted data
    return KnowledgeGraphSchema(**simulated_llm_response)

# ==========================================
# 3. Build and Query the Graph
# ==========================================

def main():
    text_input = "Tim Cook is the CEO of Apple. Apple is headquartered in Cupertino and produces the iPhone."
    print(f"Input Text: '{text_input}'\n")

    # Extract structured data
    kg_data = extract_knowledge_graph(text_input)
    
    # Initialize a directed graph
    G = nx.DiGraph()
    
    # Add Nodes with their attributes
    for node in kg_data.nodes:
        G.add_node(node.id, label=node.label, type=node.type)
        
    # Add Edges with their attributes
    for edge in kg_data.edges:
        G.add_edge(edge.source, edge.target, relation=edge.type)

    # Output the resulting Graph
    print("--- Graph Nodes ---")
    for node_id, attributes in G.nodes(data=True):
        print(f"ID: {node_id:<12} | Label: {attributes['label']:<10} | Type: {attributes['type']}")
        
    print("\n--- Graph Edges ---")
    for source, target, attributes in G.edges(data=True):
        source_label = G.nodes[source]['label']
        target_label = G.nodes[target]['label']
        relation = attributes['relation']
        print(f"({source_label}) --[{relation}]--> ({target_label})")

if __name__ == "__main__":
    main()