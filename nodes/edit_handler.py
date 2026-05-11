import re
import logging
from langchain_core.messages import AIMessage
from state import ChatState

logger = logging.getLogger(__name__)

EDIT_PATTERNS = {
    "name": {
        "keywords": ["name", "my name"],
        "step": "collect_name",
        "prompt": "Sure! Please enter your **full name** again."
    },
    "phone": {
        "keywords": ["phone", "number", "mobile", "contact number"],
        "step": "collect_phone",
        "prompt": "Sure! Please enter your **phone number** again."
    },
    "email": {
        "keywords": ["email", "mail", "email address"],
        "step": "collect_email",
        "prompt": "Sure! Please enter your **email address** again."
    },
}

EDIT_TRIGGERS = [
    r"\b(edit|change|update|correct|fix|modify|redo|re-enter|wrong|mistake|mistakenly|incorrect)\b",
]


def _detect_edit_intent(user_input: str) -> str | None:
    """
    Returns the field name ('name', 'phone', 'email') if user wants to edit,
    returns 'unknown' if edit intent detected but no field specified,
    returns None if no edit intent at all.
    """
    lowered = user_input.strip().lower()

    is_edit_intent = any(
        re.search(pattern, lowered)
        for pattern in EDIT_TRIGGERS
    )
    if not is_edit_intent:
        return None

    # Find which field they want to edit
    for field, config in EDIT_PATTERNS.items():
        for keyword in config["keywords"]:
            if keyword in lowered:
                return field

    # Edit intent detected but no specific field mentioned
    return "unknown"


def _next_step_after_edit(field: str, state: ChatState) -> str:
    """
    After editing a field, figure out where to go next.
    Skip steps that are already correctly filled AFTER the edited field.
    """
    # Define the order of collection
    order = ["name", "phone", "email", "description"]
    field_to_step = {
        "name": "collect_name",
        "phone": "collect_phone",
        "email": "collect_email",
        "description": "collect_description",
    }
    state_keys = {
        "name": state.get("name"),
        "phone": state.get("phone"),
        "email": state.get("email"),
        "description": state.get("description"),
    }

    # Start from the edited field — the edited field itself needs re-collection
    edit_index = order.index(field)

    # Find the first unfilled field at or after the edited field
    for f in order[edit_index:]:
        if not state_keys.get(f):
            return field_to_step[f]

    # All fields after edit are filled — go back to just re-collect the edited field
    return field_to_step[field]


def edit_handler_node(state: ChatState) -> ChatState:
    user_input = state.get("current_input", "")
    field = _detect_edit_intent(user_input)

    # User said "edit" but didn't say which field
    if field == "unknown":
        collected = []
        if state.get("name"):
            collected.append("**name**")
        if state.get("phone"):
            collected.append("**phone**")
        if state.get("email"):
            collected.append("**email**")

        fields_str = ", ".join(collected) if collected else "nothing yet"
        msg = (
            f"Sure! Which detail would you like to edit?\n"
            f"You have provided: {fields_str}.\n"
            f"Please say something like **'edit name'**, **'edit phone'**, or **'edit email'**."
        )
        return {
            **state,
            "last_bot_message": msg,
            "messages": state["messages"] + [AIMessage(content=msg)]
            # step unchanged — wait for them to specify
        }

    # Specific field detected — clear that field and route back to re-collect it
    if field in ("name", "phone", "email"):
        # Clear the field being edited
        cleared = {field: None}
        next_step = EDIT_PATTERNS[field]["step"]
        prompt = EDIT_PATTERNS[field]["prompt"]

        logger.info(f"[EDIT] Clearing '{field}', routing to '{next_step}'")

        return {
            **state,
            **cleared,
            "step": next_step,
            "last_bot_message": prompt,
            "messages": state["messages"] + [AIMessage(content=prompt)]
        }

    # Fallback
    msg = "What would you like to edit? You can say **'edit name'**, **'edit phone'**, or **'edit email'**."
    return {
        **state,
        "last_bot_message": msg,
        "messages": state["messages"] + [AIMessage(content=msg)]
    }