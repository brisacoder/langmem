# Notebook: import-only. Implementation lives in scripts.extract_pdf_section
# Make sure the repository root is on sys.path so local folders are importable
from dataclasses import dataclass
import logging as log
import sys
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Literal, TypedDict
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from langmem import create_manage_memory_tool, create_search_memory_tool

repo_root = Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from extract_pdf_section import (
    extract_pages_from_printed_range,
    extract_text_from_printed_range,
)

load_dotenv(override=True)


@dataclass
class ContextSchema:
    """Container for context values passed into runnables.

    llm is typed as an optional callable accepting any args/kwargs so you
    can assign plain functions, bound methods, or chat-model objects.
    """

    input_pdf: Path
    llm: Callable[..., Any] | None = None


context_schema = ContextSchema(
    input_pdf=Path(repo_root / "25_U717_Expedition_OM_ENG_V2.pdf"),
    llm=init_chat_model("openai:gpt-4.1"),
)

checkpointer = InMemorySaver()
store = InMemoryStore()


class PreferenceMemory(BaseModel):
    """Store preferences about the user."""

    category: str
    preferences: list[str]
    context: str
    recent_memories: list[str] = []


class UserProfile(BaseModel):
    name: str
    age: int | None = None
    preferences: dict | None = None
    recent_memories: list[str] = []


store = InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})

MEMORY_MANAGEMENT_INSTRUCTIONS = (
    "Proactively call this tool when you:\n\n"
    "1. Identify a new USER preference.\n"
    "2. Receive an explicit USER request to remember something or otherwise alter your behavior.\n"
    "3. Are working and want to record important context.\n"
    "4. Identify that an existing MEMORY is incorrect or outdated.\n"
    "5. Always store recent memories proactively."
)

MEMORY_SEARCH_INSTRUCTIONS = """Before generating any reply, you MUST call the memory-search tool and wait for its results if available.
The tool is the canonical source of user-specific context and must be treated as authoritative for all personalization, preferences, history, and prior interactions.
Always include the tool call on every user turn, even if the current conversation appears to contain the information.


Hard rules (must follow):
1. Always call the memory-search tool first and wait for results before producing a final reply.
2. Treat returned memory entries as authoritative for user-specific facts, preferences, and past interactions. When memory conflicts with any external source (web search, agent outputs, APIs, or the model's internal knowledge), prefer the memory content.
3. If memory entries conflict among themselves, surface the conflict to the user, prefer recent/high-score items, and ask for confirmation before taking irreversible actions.
4. If the tool returns no relevant memories, state that explicitly and proceed using standard knowledge; offer to store the current info as a new memory if appropriate.
5. Always include a one-line provenance statement in the assistant's reply indicating the namespace/timeframe searched (for example: 'Using stored memories (namespace=agent_memories, recent)').

How to format calls and justification:
- Provide a 1–2 sentence justification with the tool call explaining why you need these memories.
- Keep queries concise, mention relevant entities, timeframe, and kind when applicable.

Failure modes & safe behavior:
- If results are low-confidence or ambiguous, explicitly state uncertainty and ask a short clarifying question before acting.
- Do not expose raw internal metadata; summarize returned memories when presenting them to the user.
- If a memory appears logically impossible, note the contradiction and request confirmation, but default to treating the memory as authoritative until the user updates it.
"""

SYSTEM_PROMPT = """You are the assistant that answers general
questions, including cars and automotive topics.
Before producing any final reply, follow these rules in order.

1) Always call the memory-search tool first and wait for results.
The memory-search tool is the canonical source of user-specific
context, experiences, vehicle history, and preferences. Do not
produce a final answer before receiving and incorporating those
results.

2) The User is a Expert Ford employee therefore treat
returned memories as authoritative for Vehicle facts. When memories
conflict with external sources (web, agent outputs, APIs, or the 
model's knowledge),
prefer memory content. If external sources appear more up-to-date, surface the discrepancy and ask the
user whether to update stored memories.

3) User provides memories and facts should be weaved in the response
for consistency and continuity. You should not add user memories
separate from the main response.

4) If memory results conflict among themselves, summarize conflicts,
prefer recent/high-relevance entries, and ask the user to confirm
before irreversible actions (scheduling, purchases, cancellations).

5) Always include a one-line provenance statement in replies stating
which namespace and timeframe you searched, for example:
"Using stored memories (namespace=agent_memories, recent)". Do not
expose raw internal metadata or secrets—summarize memories for
user-facing output.

6) If the memory-search returns no relevant memories, state that
explicitly (for example: "No relevant stored memories found") and
proceed using general knowledge; offer to store the new information.

Tool Call formatting rules:
- Prepend each tool call with a 1–2 sentence justification.
- Keep queries focused: mention vehicle, symptom, timeframe, or
  preference.
- Use require_summary=true when a short synthesis is helpful.

Failure modes and safe fallbacks:
- Low-confidence: present top memory summaries, state uncertainty,
  and ask one focused question.
- No memory: say "No relevant stored memories found" and offer to
  save the interaction.
- Logically impossible memory: flag it and request confirmation before
  changing behavior.

Per-turn sequence:
1) Formulate concise memory-search query + 1–2 sentence justification;
   call the tool.
2) Receive results and optional summary; resolve conflicts as above.
3) Synthesize reply using memories as authoritative; include the
   provenance line.

End of prompt."""


namespace = ("agent_memories",)
memory_tools = [
    create_manage_memory_tool(
        instructions=MEMORY_MANAGEMENT_INSTRUCTIONS,
        namespace=namespace,
        schema=PreferenceMemory,
        actions_permitted=("create", "update", "delete"),
    ),
    create_search_memory_tool(
        instructions=MEMORY_SEARCH_INSTRUCTIONS, namespace=namespace
    ),
]
checkpointer = InMemorySaver()
memory_tools_dict = {t.name: t for t in memory_tools}


class FordStateGraph(BaseModel):
    data_privacy_pdf: Path = Field(default_factory=Path)
    data_privacy_text: str = Field(default="")
    data_privacy_items: Dict[str, Any] = Field(default_factory=dict)
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)


def read_expedition_manual(
    state: FordStateGraph, config: RunnableConfig, store: BaseStore
) -> Command[Literal["extract_privacy_section", "llm_chain"]]:
    # Demo: extract Data Privacy using the API
    # Use the repo_root (set in the import cell) so paths are absolute and not relative to the kernel CWD
    if state.data_privacy_pdf == Path():
        input_pdf = context_schema.input_pdf
        out_pdf = input_pdf.parent / "Data_Privacy_24-28_demo.pdf"
        # Example: direct page-range extraction (1-based)
        extract_pages_from_printed_range(str(input_pdf), "24", "28", str(out_pdf))
        log.info(f"Read Ford Expedition Manual {out_pdf.suffix}.")
        return Command(
            # this is the state update
            update={"data_privacy_pdf": out_pdf.as_posix()},
            # this is a replacement for an edge
            goto="extract_privacy_section",
        )
    log.info(f"Ford Expedition Manual already processed")
    return Command(
        # this is a replacement for an edge
        goto="llm_chain",
    )


def extract_privacy_section(
    state: FordStateGraph, config: RunnableConfig, store: BaseStore
) -> Command[Literal["create_memory"]]:
    # Example: direct page-range extraction (1-based)
    text = extract_text_from_printed_range(str(state.data_privacy_pdf), "24", "28")
    out_text = context_schema.input_pdf.parent / "Data_Privacy_24-28_demo.txt"
    out_text.write_text(text)
    return Command(
        # this is the state update
        update={
            "messages": HumanMessage(
                content=f"Use this context to help you answer the question: {text}"
            ),
            "data_privacy_text": text,
        },
        # this is a replacement for an edge
        goto="create_memory",
    )


def create_memory(
    state: FordStateGraph, config: RunnableConfig, store: BaseStore
) -> Command[Literal["filter_messages"]]:

    # Retrieve existing memory from the store
    namespace = ("ford", "expedition", "manual")

    # Overwrite the existing memory in the store
    key = "data_privacy"

    # Write value as a dictionary with a memory key
    store.put(namespace, key, {"memory": state.data_privacy_text})
    log.info("Created Episodic Memory for Data Privacy.")
    log.info(f"Updated Episodic Memory for {namespace} with key {key}.")
    return Command(
        # this is a replacement for an edge
        goto="filter_messages",
    )


def filter_messages(
    state: FordStateGraph, config: RunnableConfig, store: BaseStore
) -> Command[Literal["llm_chain"]]:
    # Delete all but the 2 most recent messages
    delete_messages = [
        RemoveMessage(id=m.id) for m in state.messages[2:] if m.id is not None
    ]
    return Command(
        # this is the state update
        update={"messages": delete_messages},
        # this is a replacement for an edge
        goto="llm_chain",
    )


def llm_chain(
    state: FordStateGraph, config: RunnableConfig, store: BaseStore
) -> Command[Literal["search_memory", "manage_memory", END]]:
    # Create a chain of LLM calls based on the state and config
    llm = context_schema.llm
    llm_with_tools = llm.bind_tools(memory_tools)
    resp = llm_with_tools.invoke(input=state.messages)

    if not isinstance(resp, AIMessage):
        raise TypeError("The last message in state.messages must be an AIMessage.")
    if hasattr(resp, "tool_calls") and resp.tool_calls:
        match (resp.tool_calls[0].get("name")):
            case "search_memory":
                return Command(
                    update={"messages": resp},
                    goto="search_memory",
                )
            case "manage_memory":
                return Command(
                    update={"messages": resp},
                    goto="manage_memory",
                )
    return Command(
        # this is the state update
        update={"messages": resp},
        # this is a replacement for an edge
        goto=END,
    )


def search_memory(
    state: FordStateGraph, config: RunnableConfig, store: BaseStore
) -> Command[Literal["llm_chain", END]]:
    # Ensure the last message is an AIMessage
    last_msg = state.messages[-1]
    if not isinstance(last_msg, AIMessage):
        raise TypeError("The last message in state.messages must be an AIMessage.")
    ai_msg: AIMessage = last_msg
    messages = []
    # Example tool call handling (customize as needed)
    for tool_call in ai_msg.tool_calls:
        selected_tool = memory_tools_dict[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    # Update the state with the new messages
    return Command(
        # this is the state update
        update={"messages": messages},
        # this is a replacement for an edge
        goto="llm_chain",
    )


def manage_memory(
    state: FordStateGraph, config: RunnableConfig, store: BaseStore
) -> Command[Literal["llm_chain", END]]:
    # Ensure the last message is an AIMessage
    last_msg = state.messages[-1]
    if not isinstance(last_msg, AIMessage):
        raise TypeError("The last message in state.messages must be an AIMessage.")
    ai_msg: AIMessage = last_msg
    messages = []
    # Example tool call handling (customize as needed)
    for tool_call in ai_msg.tool_calls:
        selected_tool = memory_tools_dict[tool_call["name"].lower()]
        tool_msg = selected_tool.invoke(tool_call)
        messages.append(tool_msg)
    # Update the state with the new messages
    return Command(
        # this is the state update
        update={"messages": messages},
        # this is a replacement for an edge
        goto="llm_chain",
    )


def build_graph():

    builder = StateGraph(FordStateGraph)

    builder.add_node("read_expedition_manual", read_expedition_manual)
    builder.add_node("extract_privacy_section", extract_privacy_section)
    builder.add_node("create_memory", create_memory)
    builder.add_node("filter_messages", filter_messages)
    builder.add_node("llm_chain", llm_chain)
    builder.add_node("search_memory", search_memory)
    builder.add_node("manage_memory", manage_memory)

    builder.add_edge(START, "read_expedition_manual")
    graph = builder.compile(checkpointer=checkpointer, store=store)
    return graph


def main():

    config: RunnableConfig = {
        "configurable": {
            "thread_id": "1",
            "user_id": "1",
        }
    }

    input_messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content="Hi, What is Ford Expedition Data Privacy Policy?"),
    ]

    initial_state = FordStateGraph(
        data_privacy_pdf=Path(),  # Empty Path, equivalent to None for Path type
        data_privacy_text="",
        data_privacy_items={},  # Provide an empty dict for initialization
        messages=input_messages,
    )

    graph = build_graph()

    for chunk in graph.stream(input=initial_state, config=config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()


# main()
