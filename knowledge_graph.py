# ==========================================
# 1. Mock Knowledge Graph & Interfaces
# ==========================================
# A mock graph representing a tech supply chain & company relationships
KNOWLEDGE_GRAPH = {
    "AcmeCorp": {
        "type": "Company",
        "properties": {"founded": 2010, "sector": "Hardware"},
        "edges": [
            {"relation": "ACQUIRED", "target": "ChipTech"},
            {"relation": "SUPPLIES_TO", "target": "GlobalTech"},
        ],
    },
    "ChipTech": {
        "type": "Company",
        "properties": {"specialty": "AI Chips", "location": "Austin"},
        "edges": [
            {"relation": "DESIGNS", "target": "NeuralEngineX"},
            {"relation": "PARTNERED_WITH", "target": "FabFoundry"},
        ],
    },
    "NeuralEngineX": {
        "type": "Product",
        "properties": {"architecture": "5nm ASIC", "tflops": 250},
        "edges": [],
    },
    "FabFoundry": {
        "type": "Supplier",
        "properties": {"country": "Taiwan", "capacity": "High"},
        "edges": [],
    },
    "GlobalTech": {
        "type": "Company",
        "properties": {"sector": "Consumer Electronics"},
        "edges": [],
    },
}