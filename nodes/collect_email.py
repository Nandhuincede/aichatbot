import re
import json
import logging
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from state import ChatState, ExtractedEmail, EmailValidation
from llm import get_extraction_llm, get_tool_calling_llm, get_structured_llm, get_reply_llm
from tools import validate_email_tool
from instructions import (
    EMAIL_EXTRACTOR, EMAIL_TOOL_CALLER, EMAIL_TOOL_CALLER_STRICT, EMAIL_VERDICT_INTERPRETER,
    EMAIL_REPLY_VALID, EMAIL_REPLY_AUTOCORRECTED, EMAIL_REPLY_INVALID, EMAIL_REPLY_REFUSAL,
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
        logger.warning(f"[EMAIL REPLY] LLM reply generation failed: {e}")
        return None


def run_email_validation(user_input: str) -> EmailValidation:
    # Step 1: Extract email via structured output (preserving typos)
    extracted_email = None
    try:
        extractor = get_extraction_llm(ExtractedEmail)
        extraction: ExtractedEmail = extractor.invoke([
            SystemMessage(content=EMAIL_EXTRACTOR),
            HumanMessage(content=f"Extract the email from: '{user_input}'")
        ])
        extracted_email = extraction.extracted_email.strip().lower()
        logger.info(f"[EMAIL STEP 1] Extracted: '{user_input}' -> '{extracted_email}' [{extraction.confidence}]")
    except Exception as e:
        logger.warning(f"[EMAIL STEP 1] Structured extraction failed, using regex fallback: {e}")

    if not extracted_email:
        email_match = re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', user_input)
        extracted_email = email_match.group(0).lower() if email_match else user_input.strip().lower()
        logger.info(f"[EMAIL STEP 1] Regex fallback: '{extracted_email}'")

    # Step 2: LLM calls validate_email_tool
    tool_result_json = None
    try:
        tool_llm = get_tool_calling_llm(tools=[validate_email_tool])
        tool_response = tool_llm.invoke([
            SystemMessage(content=EMAIL_TOOL_CALLER),
            HumanMessage(content=f"Validate this email using the tool: '{extracted_email}'")
        ])
        tool_calls = getattr(tool_response, 'tool_calls', [])
        if tool_calls:
            for tc in tool_calls:
                if tc['name'] == 'validate_email_tool':
                    tool_result_json = validate_email_tool.invoke({'email': extracted_email})
                    logger.info(f"[EMAIL STEP 2] Tool called: validate_email_tool('{extracted_email}') -> {tool_result_json}")
        else:
            logger.warning("[EMAIL STEP 2] LLM skipped tool call. Retrying.")
            retry_response = tool_llm.invoke([
                SystemMessage(content=EMAIL_TOOL_CALLER_STRICT),
                HumanMessage(content=f"Call validate_email_tool now with email='{extracted_email}'")
            ])
            retry_calls = getattr(retry_response, 'tool_calls', [])
            if retry_calls:
                for tc in retry_calls:
                    if tc['name'] == 'validate_email_tool':
                        tool_result_json = validate_email_tool.invoke({'email': extracted_email})
                        logger.info(f"[EMAIL STEP 2] Retry succeeded: {tool_result_json}")
    except Exception as e:
        logger.warning(f"[EMAIL STEP 2] Tool calling failed: {e}")

    if not tool_result_json:
        tool_result_json = validate_email_tool.invoke({'email': extracted_email})
        logger.info(f"[EMAIL STEP 2] Direct fallback: {tool_result_json}")

    # Step 3: Interpret tool result via structured output
    try:
        structured_llm = get_structured_llm(EmailValidation)
        result: EmailValidation = structured_llm.invoke([
            SystemMessage(content=EMAIL_VERDICT_INTERPRETER),
            HumanMessage(content=(
                f"Email being validated: '{extracted_email}'\n"
                f"Tool result: {tool_result_json}\n"
                "Return the structured validation result."
            ))
        ])
        logger.info(f"[EMAIL STEP 3] Verdict: valid={result.is_valid}, suggestion='{result.suggested_domain}'")
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


def collect_email_node(state: ChatState) -> ChatState:
    try:
        user_input = state.get("current_input", "")
        name = state.get("name", "there")
        attempts = state.get("validation_attempts", 0)

        # Refusal
        if _is_refusal(user_input):
            msg = _generate_reply(
                EMAIL_REPLY_REFUSAL,
                f"User's name: {name}. User is refusing to share their email address."
            ) or (
                "If you're unwilling to share your email, you're welcome to close the conversation. "
                "If you'd like to continue, please provide your email address."
            )
            return {**state, "last_bot_message": msg,
                    "messages": state["messages"] + [AIMessage(content=msg)]}

        result: EmailValidation = run_email_validation(user_input)

        if result.is_valid:
            email = result.cleaned_email or user_input.strip().lower()

            # Step 4: LLM generates the reply
            msg = _generate_reply(
                EMAIL_REPLY_VALID,
                f"User's name: {name}. Email received: {email}. Next step: ask for optional description."
            ) or f"Excellent, {name}! Email noted. Do you have any additional message or description? (Type 'skip' to skip)"

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

            msg = _generate_reply(
                EMAIL_REPLY_AUTOCORRECTED,
                f"User's name: {name}. Email was auto-corrected to: {email}. Next step: ask for optional description."
            ) or f"Got it, {name}! I've saved your email as {email} (auto-corrected). Do you have any additional message? (Type 'skip' to skip)"

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
            msg = _generate_reply(
                EMAIL_REPLY_INVALID,
                f"User's name: {name}. Reason email is invalid: {result.reason}."
            ) or f"That email doesn't look right, {name} — {result.reason}. Could you re-enter your email address?"

            return {**state, "validation_attempts": attempts, "last_bot_message": msg,
                    "messages": state["messages"] + [AIMessage(content=msg)]}

    except Exception as e:
        logger.error(f"Error in collect_email_node: {e}")
        name = state.get("name", "there")
        msg = f"Sorry {name}, something went wrong. Please share your email again."
        return {**state, "last_bot_message": msg,
                "messages": state["messages"] + [AIMessage(content=msg)]}
