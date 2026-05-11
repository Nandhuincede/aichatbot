from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import ChatState
from nodes import (
    greeting_node,
    collect_name_node,
    collect_phone_node,
    collect_email_node,
    collect_description_node,
    dispatch_node,
    route_from_dispatch,
    out_of_context_node,
    edit_handler_node,
)

_checkpointer = MemorySaver()


def build_graph():
    graph = StateGraph(ChatState)

    # Register nodes
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("collect_name", collect_name_node)
    graph.add_node("collect_phone", collect_phone_node)
    graph.add_node("collect_email", collect_email_node)
    graph.add_node("collect_description", collect_description_node)
    graph.add_node("out_of_context", out_of_context_node)
    graph.add_node("edit_handler", edit_handler_node)

    # Entry point
    graph.set_entry_point("dispatch")

    # Conditional routing from dispatch
    graph.add_conditional_edges(
        "dispatch",
        route_from_dispatch,
        {
            "greeting": "greeting",
            "collect_name": "collect_name",
            "collect_phone": "collect_phone",
            "collect_email": "collect_email",
            "collect_description": "collect_description",
            "out_of_context": "out_of_context",
            "edit_handler": "edit_handler",
            END: END,
        }
    )

    # All nodes end after their turn
    graph.add_edge("greeting", END)
    graph.add_edge("collect_name", END)
    graph.add_edge("collect_phone", END)
    graph.add_edge("collect_email", END)
    graph.add_edge("collect_description", END)
    graph.add_edge("out_of_context", END)
    graph.add_edge("edit_handler", END)

    compiled = graph.compile(checkpointer=_checkpointer)

    # Save graph image
    try:
        graph_bytes = compiled.get_graph().draw_mermaid_png()
        with open("graph_imagee.png", "wb") as f:
            f.write(graph_bytes)
        print("[GRAPH] graph_image.png saved successfully.")
    except Exception as e:
        print(f"[GRAPH] Could not save graph image: {e}")

    return compiled


chatbot_graph = build_graph()