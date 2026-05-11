import re
import json
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import ChatState, ExtractedPhone, PhoneValidation
from llm import get_extraction_llm, get_tool_calling_llm, get_structured_llm, get_reply_llm
from tools import validate_phone_tool
from instructions import (
    PHONE_EXTRACTOR, PHONE_TOOL_CALLER, PHONE_TOOL_CALLER_STRICT, PHONE_VERDICT_INTERPRETER,
    PHONE_REPLY_VALID, PHONE_REPLY_INVALID, PHONE_REPLY_REFUSAL,
)

logger = logging.getLogger(__name__)


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


def _generate_reply(prompt: str, context: str) -> str | None:
    try:
        llm = get_reply_llm()
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=context)
        ])
        return response.content.strip()
    except Exception as e:
        logger.warning(f"[PHONE REPLY] LLM reply generation failed: {e}")
        return None


def run_phone_validation(user_input: str) -> PhoneValidation:
    # Step 1: Extract phone via structured output
    try:
        extractor = get_extraction_llm(ExtractedPhone)
        extraction: ExtractedPhone = extractor.invoke([
            SystemMessage(content=PHONE_EXTRACTOR),
            HumanMessage(content=f"Extract phone from: '{user_input}'")
        ])
        extracted_phone = extraction.extracted_phone.strip()
        logger.info(f"[PHONE STEP 1] Extracted: '{user_input}' -> '{extracted_phone}' [{extraction.confidence}]")
    except Exception as e:
        logger.warning(f"[PHONE STEP 1] Structured extraction failed, using regex fallback: {e}")
        extracted_phone = re.sub(r'[\s\-\(\)\+]', '', user_input.strip())
        if extracted_phone.startswith('91') and len(extracted_phone) == 12:
            extracted_phone = extracted_phone[2:]

    # Step 2: LLM calls validate_phone_tool
    tool_result_json = None
    try:
        tool_llm = get_tool_calling_llm(tools=[validate_phone_tool])
        tool_response = tool_llm.invoke([
            SystemMessage(content=PHONE_TOOL_CALLER),
            HumanMessage(content=f"Validate this phone number using the tool: '{extracted_phone}'")
        ])
        tool_calls = getattr(tool_response, 'tool_calls', [])
        if tool_calls:
            for tc in tool_calls:
                if tc['name'] == 'validate_phone_tool':
                    tool_result_json = validate_phone_tool.invoke({'phone': extracted_phone})
                    logger.info(f"[PHONE STEP 2] Tool called: validate_phone_tool('{extracted_phone}') -> {tool_result_json}")
        else:
            logger.warning("[PHONE STEP 2] LLM skipped tool call. Retrying.")
            retry_response = tool_llm.invoke([
                SystemMessage(content=PHONE_TOOL_CALLER_STRICT),
                HumanMessage(content=f"Call validate_phone_tool now with phone='{extracted_phone}'")
            ])
            retry_calls = getattr(retry_response, 'tool_calls', [])
            if retry_calls:
                for tc in retry_calls:
                    if tc['name'] == 'validate_phone_tool':
                        tool_result_json = validate_phone_tool.invoke({'phone': extracted_phone})
                        logger.info(f"[PHONE STEP 2] Retry succeeded: {tool_result_json}")
    except Exception as e:
        logger.warning(f"[PHONE STEP 2] Tool calling failed: {e}")

    if not tool_result_json:
        tool_result_json = validate_phone_tool.invoke({'phone': extracted_phone})
        logger.info(f"[PHONE STEP 2] Direct fallback: {tool_result_json}")

    # Step 3: Interpret tool result via structured output
    try:
        structured_llm = get_structured_llm(PhoneValidation)
        result: PhoneValidation = structured_llm.invoke([
            SystemMessage(content=PHONE_VERDICT_INTERPRETER),
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


def collect_phone_node(state: ChatState) -> ChatState:
    try:
        user_input = state.get("current_input", "")
        name = state.get("name", "there")
        attempts = state.get("validation_attempts", 0)

        # Refusal
        if _is_refusal(user_input):
            msg = _generate_reply(
                PHONE_REPLY_REFUSAL,
                f"User's name: {name}. User is refusing to share their phone number."
            ) or (
                f"If you're unwilling to share your mobile number, you're welcome to close the conversation. "
                f"If you'd like to continue, please provide your mobile number."
            )
            return {**state, "last_bot_message": msg,
                    "messages": state["messages"] + [AIMessage(content=msg)]}

        result: PhoneValidation = run_phone_validation(user_input)

        if result.is_valid:
            phone = result.cleaned_phone or user_input.strip()
            next_step = "collect_email" if not state.get("email") else "collect_description"

            # Step 4: LLM generates the reply
            msg = _generate_reply(
                PHONE_REPLY_VALID,
                f"User's name: {name}. Phone number received: {phone}. Next step: asking for email."
            ) or f"Perfect, {name}! Got your number. Could you please share your email address?"

            return {
                **state,
                "phone": phone,
                "step": next_step,
                "validation_attempts": 0,
                "last_bot_message": msg,
                "messages": state["messages"] + [AIMessage(content=msg)]
            }
        else:
            attempts += 1
            msg = _generate_reply(
                PHONE_REPLY_INVALID,
                f"User's name: {name}. Reason phone is invalid: {result.reason}."
            ) or f"That doesn't seem right, {name} — {result.reason}. Please enter a valid 10-digit Indian mobile number."

            return {**state, "validation_attempts": attempts, "last_bot_message": msg,
                    "messages": state["messages"] + [AIMessage(content=msg)]}

    except Exception as e:
        logger.error(f"Error in collect_phone_node: {e}")
        name = state.get("name", "there")
        msg = f"Sorry {name}, I had a small issue. Please share your phone number again."
        return {**state, "last_bot_message": msg,
                "messages": state["messages"] + [AIMessage(content=msg)]}
