
import os
import re
import json
import logging
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import (
    ChatState,
    ExtractedName,
    ExtractedPhone,
    ExtractedEmail,
    NameValidation,
    PhoneValidation,
    EmailValidation,
)
from tools import (
    ALL_TOOLS,
    validate_name_tool,
    validate_phone_tool,
    validate_email_tool,
)

logger = logging.getLogger(__name__)


def _base_llm():
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in .env file")
    return ChatGroq(
        groq_api_key=api_key,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0,
        max_tokens=300
    )

def get_tool_calling_llm(tools=None):
    """LLM with bind_tools() — used in Step 2. The LLM decides which tool to call.
    Pass a specific tool list so the LLM only has access to the relevant tool.
    """
    bound_tools = tools if tools is not None else ALL_TOOLS
    return _base_llm().bind_tools(bound_tools)

def get_extraction_llm(schema):
    """LLM with structured output for a given extraction schema — used in Step 1."""
    return _base_llm().with_structured_output(schema)

def get_structured_llm(schema):
    """LLM with structured output for a given verdict schema — used in Step 3."""
    return _base_llm().with_structured_output(schema)



_NAME_RESERVED_WORDS = {
    "hello", "hi", "hey", "hiya", "howdy", "greetings", "sup", "yo",
    "yes", "no", "ok", "okay", "sure", "thanks", "thank", "bye",
    "goodbye", "start", "help", "test", "none", "null", "na", "n/a",
}

def run_name_validation(user_input: str) -> NameValidation:
    
    raw = user_input.strip()
    if re.search(r'\d', raw):
        return NameValidation(
            is_valid=False,
            reason="Name must not contain numbers. Please enter your real full name.",
            cleaned_name=""
        )
    
    if raw.lower() in _NAME_RESERVED_WORDS:
        return NameValidation(
            is_valid=False,
            reason=f"'{raw}' is a greeting, not a name. Please enter your real full name.",
            cleaned_name=""
        )

    
    try:
        extractor = get_extraction_llm(ExtractedName)
        extraction: ExtractedName = extractor.invoke([
            SystemMessage(content=(
                "You are a name extractor. Extract ONLY the person's name from the input. "
                "No extra words. 'my name is Nandhu Kumar' -> 'Nandhu Kumar'. "
                "'I am John' -> 'John'. 'Priya' -> 'Priya'."
            )),
            HumanMessage(content=f"Extract the name from: '{user_input}'")
        ])
        extracted = extraction.extracted_name.strip()
        logger.info(f"[NAME STEP 1] Extracted: '{user_input}' -> '{extracted}' [{extraction.confidence}]")
    except Exception as e:
        logger.warning(f"[NAME STEP 1] Structured extraction failed, using regex fallback: {e}")
        extracted = _regex_extract_name(user_input)

    
    tool_result_json = None
    try:
        tool_llm = get_tool_calling_llm(tools=[validate_name_tool])
        messages_step2 = [
            SystemMessage(content=(
                "You are a validator. You MUST call validate_name_tool to validate the given name. "
                "Do not respond in text — call the tool."
            )),
            HumanMessage(content=f"Validate this name using the tool: '{extracted}'")
        ]
        tool_response = tool_llm.invoke(messages_step2)
        tool_calls = getattr(tool_response, 'tool_calls', [])
        if tool_calls:
            for tc in tool_calls:
                if tc['name'] == 'validate_name_tool':
                    tool_result_json = validate_name_tool.invoke({'name': extracted})
                    logger.info(f"[NAME STEP 2] Tool called by LLM: validate_name_tool('{extracted}') -> {tool_result_json}")
        else:
            logger.warning(f"[NAME STEP 2] LLM skipped tool call for '{extracted}'. Retrying with stricter prompt.")
            
            retry_response = tool_llm.invoke([
                SystemMessage(content="You MUST call validate_name_tool. No text responses allowed."),
                HumanMessage(content=f"Call validate_name_tool now with name='{extracted}'")
            ])
            retry_calls = getattr(retry_response, 'tool_calls', [])
            if retry_calls:
                for tc in retry_calls:
                    if tc['name'] == 'validate_name_tool':
                        tool_result_json = validate_name_tool.invoke({'name': extracted})
                        logger.info(f"[NAME STEP 2] Retry succeeded: validate_name_tool('{extracted}') -> {tool_result_json}")
            else:
                logger.warning(f"[NAME STEP 2] Retry also skipped tool call. Falling back to direct invocation.")
    except Exception as e:
        logger.warning(f"[NAME STEP 2] Tool calling failed: {e}")

    if not tool_result_json:
        tool_result_json = validate_name_tool.invoke({'name': extracted})
        logger.info(f"[NAME STEP 2] Direct fallback: {tool_result_json}")

    
    try:
        structured_llm = get_structured_llm(NameValidation)
        result: NameValidation = structured_llm.invoke([
            SystemMessage(content=(
                "You are a validation result interpreter. "
                "Read the tool result and return a structured NameValidation response."
            )),
            HumanMessage(content=(
                f"Name being validated: '{extracted}'\n"
                f"Tool result: {tool_result_json}\n"
                "Return the structured validation result."
            ))
        ])
        if result.is_valid and not result.cleaned_name:
            result.cleaned_name = extracted.title()
        logger.info(f"[NAME STEP 3] Verdict: valid={result.is_valid}, name='{result.cleaned_name}'")
        return result
    except Exception as e:
        logger.warning(f"[NAME STEP 3] Structured output failed, parsing directly: {e}")
        tool_data = json.loads(tool_result_json)
        return NameValidation(
            is_valid=tool_data['is_valid'],
            reason=tool_data['reason'],
            cleaned_name=extracted.title() if tool_data['is_valid'] else ""
        )


def run_phone_validation(user_input: str) -> PhoneValidation:
    
    try:
        extractor = get_extraction_llm(ExtractedPhone)
        extraction: ExtractedPhone = extractor.invoke([
            SystemMessage(content=(
                "Extract only the phone number digits from the input. "
                "Remove spaces, dashes, brackets, and the +91 prefix."
            )),
            HumanMessage(content=f"Extract phone from: '{user_input}'")
        ])
        extracted_phone = extraction.extracted_phone.strip()
        logger.info(f"[PHONE STEP 1] Extracted: '{user_input}' -> '{extracted_phone}' [{extraction.confidence}]")
    except Exception as e:
        logger.warning(f"[PHONE STEP 1] Structured extraction failed, using regex fallback: {e}")
        extracted_phone = re.sub(r'[\s\-\(\)\+]', '', user_input.strip())
        if extracted_phone.startswith('91') and len(extracted_phone) == 12:
            extracted_phone = extracted_phone[2:]

    
    tool_result_json = None
    try:
        tool_llm = get_tool_calling_llm(tools=[validate_phone_tool])
        tool_response = tool_llm.invoke([
            SystemMessage(content=(
                "You are a validator. You MUST call validate_phone_tool to validate the given Indian phone number. "
                "Do not respond in text — call the tool."
            )),
            HumanMessage(content=f"Validate this phone number using the tool: '{extracted_phone}'")
        ])
        tool_calls = getattr(tool_response, 'tool_calls', [])
        if tool_calls:
            for tc in tool_calls:
                if tc['name'] == 'validate_phone_tool':
                    tool_result_json = validate_phone_tool.invoke({'phone': extracted_phone})
                    logger.info(f"[PHONE STEP 2] Tool called by LLM: validate_phone_tool('{extracted_phone}') -> {tool_result_json}")
        else:
            logger.warning(f"[PHONE STEP 2] LLM skipped tool call for '{extracted_phone}'. Retrying with stricter prompt.")
            retry_response = tool_llm.invoke([
                SystemMessage(content="You MUST call validate_phone_tool. No text responses allowed."),
                HumanMessage(content=f"Call validate_phone_tool now with phone='{extracted_phone}'")
            ])
            retry_calls = getattr(retry_response, 'tool_calls', [])
            if retry_calls:
                for tc in retry_calls:
                    if tc['name'] == 'validate_phone_tool':
                        tool_result_json = validate_phone_tool.invoke({'phone': extracted_phone})
                        logger.info(f"[PHONE STEP 2] Retry succeeded: validate_phone_tool('{extracted_phone}') -> {tool_result_json}")
            else:
                logger.warning(f"[PHONE STEP 2] Retry also skipped tool call. Falling back to direct invocation.")
    except Exception as e:
        logger.warning(f"[PHONE STEP 2] Tool calling failed: {e}")

    if not tool_result_json:
        tool_result_json = validate_phone_tool.invoke({'phone': extracted_phone})
        logger.info(f"[PHONE STEP 2] Direct fallback: {tool_result_json}")

    
    try:
        structured_llm = get_structured_llm(PhoneValidation)
        result: PhoneValidation = structured_llm.invoke([
            SystemMessage(content=(
                "You are a validation result interpreter. "
                "Read the tool result and return a structured PhoneValidation response."
            )),
            HumanMessage(content=(
                f"Phone being validated: '{user_input}'\n"
                f"Tool result: {tool_result_json}\n"
                "Return the structured validation result."
            ))
        ])
        logger.info(f"[PHONE STEP 3] Verdict: valid={result.is_valid}, phone='{result.cleaned_phone}'")
        return result
    except Exception as e:
        logger.warning(f"[PHONE STEP 3] Structured output failed, parsing directly: {e}")
        tool_data = json.loads(tool_result_json)
        return PhoneValidation(
            is_valid=tool_data['is_valid'],
            reason=tool_data['reason'],
            cleaned_phone=tool_data.get('cleaned_phone', '')
        )


def run_email_validation(user_input: str) -> EmailValidation:
    # ── Step 1: Extract email via structured output ─────────────────────────
    # REQ 3 FIX: Was regex-only before. Now uses with_structured_output(ExtractedEmail).
    # The LLM is instructed to preserve typos so the email tool can detect them.
    extracted_email = None
    try:
        extractor = get_extraction_llm(ExtractedEmail)
        extraction: ExtractedEmail = extractor.invoke([
            SystemMessage(content=(
                "You are an email extractor. Extract the email address from the input EXACTLY as typed. "
                "Do NOT correct any typos — if someone wrote 'gmial.com', keep 'gmial.com'. "
                "Just extract what is there, lowercased."
            )),
            HumanMessage(content=f"Extract the email from: '{user_input}'")
        ])
        extracted_email = extraction.extracted_email.strip().lower()
        logger.info(f"[EMAIL STEP 1] Extracted: '{user_input}' -> '{extracted_email}' [{extraction.confidence}]")
    except Exception as e:
        logger.warning(f"[EMAIL STEP 1] Structured extraction failed, using regex fallback: {e}")

    # Regex fallback (also used as secondary check)
    if not extracted_email:
        email_match = re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', user_input)
        extracted_email = email_match.group(0).lower() if email_match else user_input.strip().lower()
        logger.info(f"[EMAIL STEP 1] Regex fallback: '{extracted_email}'")

    # ── Step 2: LLM calls validate_email_tool (only this tool is bound) ───
    tool_result_json = None
    try:
        tool_llm = get_tool_calling_llm(tools=[validate_email_tool])
        tool_response = tool_llm.invoke([
            SystemMessage(content=(
                "You are a validator. You MUST call validate_email_tool to validate the given email address. "
                "Do not respond in text — call the tool."
            )),
            HumanMessage(content=f"Validate this email using the tool: '{extracted_email}'")
        ])
        tool_calls = getattr(tool_response, 'tool_calls', [])
        if tool_calls:
            for tc in tool_calls:
                if tc['name'] == 'validate_email_tool':
                    tool_result_json = validate_email_tool.invoke({'email': extracted_email})
                    logger.info(f"[EMAIL STEP 2] Tool called by LLM: validate_email_tool('{extracted_email}') -> {tool_result_json}")
        else:
            logger.warning(f"[EMAIL STEP 2] LLM skipped tool call for '{extracted_email}'. Retrying with stricter prompt.")
            retry_response = tool_llm.invoke([
                SystemMessage(content="You MUST call validate_email_tool. No text responses allowed."),
                HumanMessage(content=f"Call validate_email_tool now with email='{extracted_email}'")
            ])
            retry_calls = getattr(retry_response, 'tool_calls', [])
            if retry_calls:
                for tc in retry_calls:
                    if tc['name'] == 'validate_email_tool':
                        tool_result_json = validate_email_tool.invoke({'email': extracted_email})
                        logger.info(f"[EMAIL STEP 2] Retry succeeded: validate_email_tool('{extracted_email}') -> {tool_result_json}")
            else:
                logger.warning(f"[EMAIL STEP 2] Retry also skipped tool call. Falling back to direct invocation.")
    except Exception as e:
        logger.warning(f"[EMAIL STEP 2] Tool calling failed: {e}")

    if not tool_result_json:
        tool_result_json = validate_email_tool.invoke({'email': extracted_email})
        logger.info(f"[EMAIL STEP 2] Direct fallback: {tool_result_json}")

    
    try:
        structured_llm = get_structured_llm(EmailValidation)
        result: EmailValidation = structured_llm.invoke([
            SystemMessage(content=(
                "You are a validation result interpreter. "
                "Read the tool result and return a structured EmailValidation response. "
                "IMPORTANT: If the tool result contains a suggested_domain, copy it into the "
                "suggested_domain field exactly as-is. Do not alter the domain or invent one."
            )),
            HumanMessage(content=(
                f"Email being validated: '{extracted_email}'\n"
                f"Tool result: {tool_result_json}\n"
                "Return the structured validation result."
            ))
        ])
        logger.info(
            f"[EMAIL STEP 3] Verdict: valid={result.is_valid}, "
            f"suggestion='{result.suggested_domain}'"
        )
        return result
    except Exception as e:
        logger.warning(f"[EMAIL STEP 3] Structured output failed, parsing directly: {e}")
        tool_data = json.loads(tool_result_json)
        return EmailValidation(
            is_valid=tool_data['is_valid'],
            reason=tool_data['reason'],
            cleaned_email=tool_data.get('cleaned_email', ''),
            suggested_domain=tool_data.get('suggested_domain', '')
        )




def _is_refusal(user_input: str) -> bool:
    refusal_patterns = [
        r'^(no|nope|nah|na|noo|nooo)$',
        r'^(i\s+)?(don\'t|dont|do not|wont|won\'t|will not|not going to|not gonna)\s+(want|give|share|tell|provide)',
        r'^(not\s+)?(interested|willing|sharing|telling|giving)',
        r'^(i\s+)?(refuse|decline|skip\s+this|pass)',
        r'^(why|why\s+should|why\s+do)',
        r'^(private|personal|confidential|none\s+of\s+your)',
        r"^(i\s+)?(don't|dont)\s+(have|want\s+to)",
        r'^no\s+(i|way|thanks|thank)',
    ]
    s = user_input.strip().lower()
    for p in refusal_patterns:
        if re.search(p, s, re.IGNORECASE):
            return True
    return False

def _refusal_msg(field: str) -> str:
    return (
        f"If you're unwilling to share your **{field}**, you're welcome to close the conversation. "
        f"If you'd like to continue, please go ahead and provide your **{field}**."
    )

def _regex_extract_name(user_input: str) -> str:
    patterns = [
        r"(?:my name is|i am|i'm|call me|name['\\s]?s?(?:\\s+is)?)\\s+([A-Za-z\\s]+)",
        r"^([A-Za-z]+(?:\\s[A-Za-z]+)*)$"
    ]
    for p in patterns:
        m = re.search(p, user_input.strip(), re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return user_input.strip()




def dispatch_node(state: ChatState) -> ChatState:
    """
    REQ 4 FIX: Replaces the dummy 'lambda state: state' router.
    This is the real entry point — it reads state['step'] and decides
    what to do. For the greeting turn it delegates immediately;
    for all other turns it is a transparent pass-through that allows
    conditional_edges to route to the correct collector node.
    """
    return state


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

def collect_name_node(state: ChatState) -> ChatState:
    try:
        user_input = state.get("current_input", "")
        attempts = state.get("validation_attempts", 0)
        if _is_refusal(user_input):
            msg = _refusal_msg("full name")
            return {**state, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}
        result: NameValidation = run_name_validation(user_input)
        if result.is_valid:
            name = result.cleaned_name or user_input.strip().title()
            msg = (
                f"Great to meet you, **{name}**!\n\n"
                "Could you please share your **Indian mobile number**? (10 digits)"
            )
            return {
                **state,
                "name": name,
                "step": "collect_phone",
                "validation_attempts": 0,
                "last_bot_message": msg,
                "messages": state["messages"] + [AIMessage(content=msg)]
            }
        else:
            attempts += 1
            msg = f"Hmm, that doesn't look like a valid name — {result.reason}. Could you please enter your **full name**?"
            return {**state, "validation_attempts": attempts, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}
    except Exception as e:
        logger.error(f"Error in collect_name_node: {e}")
        msg = "Sorry, I had a small hiccup. Could you please share your **full name** again?"
        return {**state, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}

def collect_phone_node(state: ChatState) -> ChatState:
    try:
        user_input = state.get("current_input", "")
        name = state.get("name", "there")
        attempts = state.get("validation_attempts", 0)
        if _is_refusal(user_input):
            msg = _refusal_msg("mobile number")
            return {**state, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}
        result: PhoneValidation = run_phone_validation(user_input)
        if result.is_valid:
            phone = result.cleaned_phone or user_input.strip()
            msg = (
                f"Perfect, {name}! Got your number.\n\n"
                "Now, could you please share your **email address**?"
            )
            return {
                **state,
                "phone": phone,
                "step": "collect_email",
                "validation_attempts": 0,
                "last_bot_message": msg,
                "messages": state["messages"] + [AIMessage(content=msg)]
            }
        else:
            attempts += 1
            msg = f"That doesn't seem right, {name} — {result.reason}. Please enter a valid **10-digit Indian mobile number**."
            return {**state, "validation_attempts": attempts, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}
    except Exception as e:
        logger.error(f"Error in collect_phone_node: {e}")
        name = state.get("name", "there")
        msg = f"Sorry {name}, I had a small issue. Please share your **phone number** again."
        return {**state, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}

def collect_email_node(state: ChatState) -> ChatState:
    try:
        user_input = state.get("current_input", "")
        name = state.get("name", "there")
        attempts = state.get("validation_attempts", 0)
        if _is_refusal(user_input):
            msg = _refusal_msg("email address")
            return {**state, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}
        result: EmailValidation = run_email_validation(user_input)
        if result.is_valid:
            email = result.cleaned_email or user_input.strip().lower()
            msg = (
                f"Excellent, {name}! Email noted.\n\n"
                "Lastly, do you have any **additional message or description**? (Type 'skip' to skip)"
            )
            return {
                **state,
                "email": email,
                "step": "collect_description",
                "validation_attempts": 0,
                "last_bot_message": msg,
                "messages": state["messages"] + [AIMessage(content=msg)]
            }
        elif result.suggested_domain:
            raw_email = re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', user_input)
            if raw_email:
                local = raw_email.group(0).lower().split('@')[0]
                email = f"{local}@{result.suggested_domain}"
            else:
                email = user_input.strip().lower()
            msg = (
                f"Excellent, {name}! Email saved as **{email}** (auto-corrected domain).\n\n"
                "Lastly, do you have any **additional message or description**? (Type 'skip' to skip)"
            )
            return {
                **state,
                "email": email,
                "step": "collect_description",
                "validation_attempts": 0,
                "last_bot_message": msg,
                "messages": state["messages"] + [AIMessage(content=msg)]
            }
        else:
            attempts += 1
            msg = f"That email doesn't look right, {name} — {result.reason}. Could you re-enter your **email address**?"
            return {**state, "validation_attempts": attempts, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}
    except Exception as e:
        logger.error(f"Error in collect_email_node: {e}")
        name = state.get("name", "there")
        msg = f"Sorry {name}, something went wrong. Please share your **email** again."
        return {**state, "last_bot_message": msg, "messages": state["messages"] + [AIMessage(content=msg)]}

def collect_description_node(state: ChatState) -> ChatState:
    user_input = state.get("current_input", "").strip()
    name = state.get("name", "there")
    description = None if user_input.lower() in ['skip', '', 'no', 'none', 'n/a', 'na'] else user_input
    msg = (
        f"Thank you for providing your contact details, **{name}**!\n\n"
        f"Here's a summary of what I've collected:\n"
        f"- **Name:** {state.get('name')}\n"
        f"- **Phone:** {state.get('phone')}\n"
        f"- **Email:** {state.get('email')}\n"
        f"- **Description:** {description or '—'}\n\n"
        f"We'll be in touch soon! Have a wonderful day!"
    )
    return {
        **state,
        "description": description,
        "step": "done",
        "last_bot_message": msg,
        "messages": state["messages"] + [AIMessage(content=msg)]
    }




def route_from_dispatch(state: ChatState) -> str:
    """
    Reads state['step'] and returns the next node name.
    Called as the conditional edge function from dispatch_node.

    REQ 4 FIX: Previously this was called 'route_by_step' and was attached
    to a dummy 'router' lambda node. Now it routes directly from dispatch_node,
    which is the real set_entry_point() node.
    """
    step = state.get("step", "greeting")
    valid_nodes = {"greeting", "collect_name", "collect_phone", "collect_email", "collect_description"}
    if step in valid_nodes:
        return step
    return END



_checkpointer = MemorySaver()


def build_graph():
    graph = StateGraph(ChatState)

    
    graph.add_node("dispatch", dispatch_node)
    graph.add_node("greeting", greeting_node)
    graph.add_node("collect_name", collect_name_node)
    graph.add_node("collect_phone", collect_phone_node)
    graph.add_node("collect_email", collect_email_node)
    graph.add_node("collect_description", collect_description_node)

    
    graph.set_entry_point("dispatch")

    graph.add_conditional_edges(
        "dispatch",
        route_from_dispatch,
        {
            "greeting": "greeting",
            "collect_name": "collect_name",
            "collect_phone": "collect_phone",
            "collect_email": "collect_email",
            "collect_description": "collect_description",
            END: END,
        }
    )
    graph.add_edge("greeting", END)
    graph.add_edge("collect_name", END)
    graph.add_edge("collect_phone", END)
    graph.add_edge("collect_email", END)
    graph.add_edge("collect_description", END)

    
    return graph.compile(checkpointer=_checkpointer)


chatbot_graph = build_graph()




def process_message(session_id: str, session_state: dict, user_message: str = None) -> tuple:
    """
    Routes through the compiled LangGraph graph using MemorySaver checkpointing.

    Turn 1 (greeting): session_state has no 'step' yet — we initialise the full
    ChatState and invoke. MemorySaver writes the checkpoint.

    Turn 2+ (collect_*): MemorySaver already holds the full state for this
    thread_id. We pass ONLY the fields that change each turn (current_input,
    last_bot_message reset, and the new HumanMessage). LangGraph merges these
    with the saved checkpoint via the add_messages reducer — the step, name,
    phone, email fields are restored automatically from the checkpoint.

    Returns: (bot_response: str, lightweight_state: dict)
    """
    config = {"configurable": {"thread_id": session_id}}

    if not session_state.get("step"):
        
        state: ChatState = {
            "messages": [],
            "session_id": session_id,
            "step": "greeting",
            "name": None,
            "phone": None,
            "email": None,
            "description": None,
            "current_input": None,
            "validation_attempts": 0,
            "last_bot_message": None,
        }
    else:
       
        state: ChatState = {
            "messages": [HumanMessage(content=user_message or "")],
            "current_input": user_message,
            "last_bot_message": None,
            # session_id is not in ChatState keys that MemorySaver tracks,
            # but we keep it for safety in case a node reads it.
            "session_id": session_id,
            # Carry forward non-message fields from session_store so nodes
            # can still read them even if the checkpointer misses a merge.
            "step": session_state.get("step", "greeting"),
            "name": session_state.get("name"),
            "phone": session_state.get("phone"),
            "email": session_state.get("email"),
            "description": session_state.get("description"),
            "validation_attempts": session_state.get("validation_attempts", 0),
        }

    result = chatbot_graph.invoke(state, config=config)

    # Return bot reply + a lightweight dict (no messages list) for Flask session_store
    lightweight = {
        k: v for k, v in result.items()
        if k not in ("messages",)
    }
    return result["last_bot_message"], lightweight
