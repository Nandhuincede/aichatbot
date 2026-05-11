from langgraph.graph import END
from state import ChatState
from nodes.out_of_context import is_out_of_context

def dispatch_node(state: ChatState) -> ChatState:
    """
    Real entry point — reads state['step'] and acts as a transparent
    pass-through so conditional_edges can route to the correct collector node.
    """
    return state


from nodes.edit_handler import _detect_edit_intent

def route_from_dispatch(state: ChatState) -> str:
    step = state.get("step", "greeting")
    user_input = state.get("current_input", "") or ""

    # Check edit intent first — only if at least one field has been collected
    if step not in ("greeting", "done") and user_input:
        field = _detect_edit_intent(user_input)
        if field:
            # Only allow editing fields that have already been collected
            collected = {
                "name": state.get("name"),
                "phone": state.get("phone"),
                "email": state.get("email"),
            }
            if collected.get(field):
                return "edit_handler"

    # OOC check
    if step not in ("greeting", "done", "collect_description") and user_input:
        if is_out_of_context(user_input):
            return "out_of_context"

    valid_nodes = {"greeting", "collect_name", "collect_phone", "collect_email", "collect_description"}
    if step in valid_nodes:
        return step
    return END