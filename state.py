from typing import Optional, Annotated
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


#  LangGraph State 

class ChatState(TypedDict):
    """
    Shared state passed between every LangGraph node.
    Each node reads from this and returns an updated copy.
    """
    messages: Annotated[list, add_messages]  
    session_id: str
    step: str          
    name: Optional[str]
    phone: Optional[str]
    email: Optional[str]
    description: Optional[str]
    current_input: Optional[str]      
    validation_attempts: int
    last_bot_message: Optional[str]   




class ExtractedName(BaseModel):
    """
    Used in Step 1 (with_structured_output) to extract the real name from
    any sentence.
    Example: 'my name is Nandhu Kumar' -> extracted_name='Nandhu Kumar'
    """
    extracted_name: str = Field(description="The person's name only. No extra words.")
    confidence: str = Field(description="Confidence: high, medium, or low")


class ExtractedPhone(BaseModel):
    """
    Used in Step 1 (with_structured_output) to extract phone number digits.
    Strips country codes, spaces, and symbols.
    Example: 'my number is +91 98765 43210' -> extracted_phone='9876543210'

    FIX (Req 3): Moved from inline class inside run_phone_validation() to a
    proper top-level schema alongside ExtractedName.
    """
    extracted_phone: str = Field(
        description="Only the phone digits, no spaces, dashes, brackets, or +91 prefix."
    )
    confidence: str = Field(description="Confidence: high, medium, or low")


class ExtractedEmail(BaseModel):
    """
    Used in Step 1 (with_structured_output) to extract the raw email address
    exactly as the user typed it — preserving typos so the email tool can
    detect and suggest corrections (e.g. gmial.com stays gmial.com).

    FIX (Req 3): Email extraction was previously regex-only. Now uses
    structured output so all three fields follow the same pipeline:
    extraction (structured output) -> tool call -> verdict (structured output).
    """
    extracted_email: str = Field(
        description=(
            "The raw email address exactly as typed. "
            "Do NOT correct typos — preserve gmial.com, yahooo.com, etc."
        )
    )
    confidence: str = Field(description="Confidence: high, medium, or low")




class NameValidation(BaseModel):
    """Used in Step 3 (with_structured_output) for name validation verdict."""
    is_valid: bool = Field(description="True if this is a realistic human name")
    reason: str = Field(description="Explanation of why valid or invalid")
    cleaned_name: str = Field(description="Title-cased name if valid, empty string if not")


class PhoneValidation(BaseModel):
    """Used in Step 3 (with_structured_output) for phone validation verdict."""
    is_valid: bool = Field(description="True if valid Indian mobile number")
    reason: str = Field(description="Explanation of why valid or invalid")
    cleaned_phone: str = Field(description="10-digit phone if valid, empty string if not")


class EmailValidation(BaseModel):
    """
    Used in Step 3 (with_structured_output) for email validation verdict.

    FIX (Req 3): Email Step 3 now uses with_structured_output(EmailValidation)
    via LLM — same pattern as name and phone. LLM is instructed to echo
    the tool's suggested_domain faithfully so typo detection is preserved.
    """
    is_valid: bool = Field(description="True if valid email address")
    reason: str = Field(description="Explanation of why valid or invalid")
    cleaned_email: str = Field(description="Lowercase email if valid, empty string if not")
    suggested_domain: str = Field(
        default="",
        description="Suggested correct domain if a typo was detected, empty string otherwise"
    )
