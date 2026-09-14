import json

from langchain_core.messages import AIMessage

from main import parse_tool_calls_from_message


def test_parse_tool_calls_from_json_content():
    raw = '{"name": "search_nodes", "arguments": {"query": "AcmeCorp acquisitions"}}'
    message = AIMessage(content=raw)

    tool_calls = parse_tool_calls_from_message(message)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "search_nodes"
    assert tool_calls[0]["args"]["query"] == "AcmeCorp acquisitions"


def test_parse_tool_calls_when_text_precedes_json():
    raw = 'It appears there are no direct matches. {"name": "search_nodes", "arguments": {"query": "AcmeCorp acquisitions target companies"}}'
    message = AIMessage(content=raw)

    tool_calls = parse_tool_calls_from_message(message)

    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "search_nodes"
    assert tool_calls[0]["args"]["query"] == "AcmeCorp acquisitions target companies"
