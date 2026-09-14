import json
import os
import uuid
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from knowledge_graph import KNOWLEDGE_GRAPH
from tools import (
    search_nodes,
    get_neighbors,
    get_node_details,
)
from prompts import (
    SYSTEM_PROMPT,
)
from dotenv import load_dotenv
load_dotenv()

tools = [search_nodes, get_neighbors, get_node_details]
tools_by_name = {t.name: t for t in tools}

# ==========================================
# 2. Define State Schema & System Prompts
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================
# 3. Define Graph Nodes & Edges
# ==========================================
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(
    model="qwen2.5-coder:7b",
    base_url=ollama_base_url,
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)


def parse_tool_calls_from_message(message: BaseMessage) -> list[dict]:
    """Handle native tool_calls as well as Ollama text payloads that include a JSON tool call."""
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        return tool_calls

    content = getattr(message, "content", "")
    if not isinstance(content, str):
        return []

    stripped = content.strip()
    if not stripped:
        return []

    json_candidates = []
    start_index = stripped.find("{")
    if start_index != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start_index, len(stripped)):
            ch = stripped[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        json_candidates.append(stripped[start_index : i + 1])
                        break

    for candidate in json_candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict) and "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
            return parsed["tool_calls"]

        if isinstance(parsed, dict) and "name" in parsed:
            name = parsed.get("name")
            arguments = parsed.get("arguments", {}) or {}
            if isinstance(arguments, dict):
                return [{
                    "id": f"call_{uuid.uuid4().hex}",
                    "name": name,
                    "args": arguments,
                }]

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict) and "tool_calls" in parsed and isinstance(parsed["tool_calls"], list):
        return parsed["tool_calls"]

    if isinstance(parsed, dict) and "name" in parsed:
        name = parsed.get("name")
        arguments = parsed.get("arguments", {}) or {}
        if isinstance(arguments, dict):
            return [{
                "id": f"call_{uuid.uuid4().hex}",
                "name": name,
                "args": arguments,
            }]

    return []


def agent_node(state: AgentState) -> dict:
    """Evaluates the conversation history and decides whether to call a tool or finalize."""
    messages = state["messages"]
    
    # Inject system prompt at start
    if not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_executor_node(state: AgentState) -> dict:
    """Executes tool requests issued by the LLM and appends output messages."""
    last_message = state["messages"][-1]
    tool_calls = parse_tool_calls_from_message(last_message)
    tool_results = []

    for tool_call in tool_calls:
        tool_fn = tools_by_name[tool_call["name"]]
        output = tool_fn.invoke(tool_call["args"])
        tool_results.append(
            ToolMessage(
                content=str(output),
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": tool_results}

def should_continue(state: AgentState) -> str:
    """Router: Checks if the LLM called any tools or finished answering."""
    last_message = state["messages"][-1]
    if parse_tool_calls_from_message(last_message):
        tool_call_count = sum(
            1 for message in state["messages"] if isinstance(message, ToolMessage)
        )
        if tool_call_count >= 12:
            return END
        return "execute_tools"
    return END

# ==========================================
# 4. Compile the LangGraph Runtime
# ==========================================
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent_node)
workflow.add_node("execute_tools", tool_executor_node)

# Add Edges
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["execute_tools", END])
workflow.add_edge("execute_tools", "agent")

app = workflow.compile()

# ==========================================
# 5. Execution Demo
# ==========================================
if __name__ == "__main__":
    query = "What products or architecture is engineered by companies acquired by AcmeCorp?"
    print(f"User Query: {query}\n" + "="*50)

    initial_state = {"messages": [HumanMessage(content=query)]}

    for event in app.stream(initial_state, stream_mode="values"):
        latest_msg = event["messages"][-1]
        tool_calls = parse_tool_calls_from_message(latest_msg)

        if isinstance(latest_msg, HumanMessage):
            continue
        elif tool_calls:
            for tc in tool_calls:
                print(f"[Agent Action] Calling Tool '{tc['name']}' with args: {tc['args']}")
        elif isinstance(latest_msg, ToolMessage):
            print(f"[Tool Output] -> {latest_msg.content}")
        else:
            print(f"\n[Final Response]\n{latest_msg.content}")