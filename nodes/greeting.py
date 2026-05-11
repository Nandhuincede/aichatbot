from langchain_core.messages import AIMessage
from state import ChatState


def greeting_node(state: ChatState) -> ChatState:
    msg = (
        "Hi there! Welcome to **Incede Technology**! "
        "I'm here to collect your contact details. Let's get started!\n\n"
        "Could you please tell me your **full name**?"
    )
    return {
        **state,
        "step": "collect_name",
        "last_bot_message": msg,
        "messages": state["messages"] + [AIMessage(content=msg)]
    }
