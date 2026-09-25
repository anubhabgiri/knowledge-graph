import json

import tools


class FakeGraph:
    def search_nodes(self, query):
        assert query == "chip"
        return [{"node_id": "ChipTech", "element_id": "4:abc:1", "labels": ["Company"]}]

    def get_neighbors(self, node_id):
        assert node_id == "ChipTech"
        edge = {
            "relation": "DESIGNS",
            "direction": "outgoing",
            "target": "NeuralEngineX",
        }
        return [{"node_id": node_id, "edges": [None, edge]}]

    def get_node_details(self, node_id):
        assert node_id == "ChipTech"
        return [{"node_id": node_id, "labels": ["Company"], "properties": {"sector": "Hardware"}}]


def test_tools_return_live_graph_result_shape(monkeypatch):
    monkeypatch.setattr(tools, "graph", FakeGraph())

    result = json.loads(tools.search_nodes.invoke({"query": "chip"}))
    assert result["matched_nodes"][0]["node_id"] == "ChipTech"
    neighbors = json.loads(tools.get_neighbors.invoke({"node_id": "ChipTech"}))
    assert neighbors["edges"] == [{"relation": "DESIGNS", "direction": "outgoing", "target": "NeuralEngineX"}]
    details = json.loads(tools.get_node_details.invoke({"node_id": "ChipTech"}))
    assert details["properties"]["sector"] == "Hardware"
