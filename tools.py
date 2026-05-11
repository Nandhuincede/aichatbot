import re
import json
from langchain_core.tools import tool

_RESERVED_WORDS = {
    "hello", "hi", "hey", "hiya", "howdy", "greetings", "sup", "yo",
    "yes", "no", "ok", "okay", "sure", "thanks", "thank", "bye",
    "goodbye", "start", "help", "test", "none", "null", "na", "n/a",
}

@tool
def validate_name_tool(name: str) -> str:
    """
    TOOL CALLING — Validates if the given string is a realistic human name.
    Checks: reserved/greeting words, length, digits, special chars, vowel ratio.
    Returns JSON: {is_valid: bool, reason: str}
    """
    result = {"is_valid": False, "reason": ""}
    s = name.strip()

    if len(s) < 2:
        result["reason"] = "Name is too short (minimum 2 characters)"
        return json.dumps(result)

    # Reject common greetings and reserved words
    if s.lower() in _RESERVED_WORDS:
        result["reason"] = f"'{s}' is a greeting or reserved word, not a name"
        return json.dumps(result)

    if re.search(r'\d', s):
        result["reason"] = "Name contains numbers which is not valid"
        return json.dumps(result)
    if not re.match(r"^[A-Za-z\s\-'\.]+$", s):
        result["reason"] = "Name contains invalid special characters"
        return json.dumps(result)

    letters = [c for c in s.lower() if c.isalpha()]
    vowels = sum(1 for c in letters if c in 'aeiou')

    if len(letters) >= 3 and vowels == 0:
        result["reason"] = "Name appears to be gibberish — no vowels found"
        return json.dumps(result)
    if len(letters) > 4 and (vowels / len(letters)) < 0.2:
        result["reason"] = "Name appears to be gibberish — too few vowels"
        return json.dumps(result)

    result["is_valid"] = True
    result["reason"] = f"'{s}' is a valid human name"
    return json.dumps(result)


# Phone Tool 

@tool
def validate_phone_tool(phone: str) -> str:
    """
    TOOL CALLING — Validates if the given string is a valid Indian mobile number.
    Rules: exactly 10 digits, starts with 6/7/8/9, strips country code if present.
    Returns JSON: {is_valid: bool, reason: str, cleaned_phone: str}
    """
    result = {"is_valid": False, "reason": "", "cleaned_phone": ""}
    cleaned = re.sub(r'[\s\-\(\)\+]', '', phone.strip())

    # Strip country code
    if cleaned.startswith('91') and len(cleaned) == 12:
        cleaned = cleaned[2:]
    if cleaned.startswith('+91'):
        cleaned = cleaned[3:]

    if not cleaned.isdigit():
        result["reason"] = "Phone number must contain only digits"
        return json.dumps(result)
    if len(cleaned) != 10:
        result["reason"] = f"Indian phone numbers must be exactly 10 digits (got {len(cleaned)})"
        return json.dumps(result)
    if cleaned[0] not in '6789':
        result["reason"] = "Indian mobile numbers must start with 6, 7, 8, or 9"
        return json.dumps(result)
    if len(set(cleaned))==1:
        result["reason"]="Phone number cannot be all the same digit"
        return json.dump(result)
    digits =[int(d) for d in cleaned]
    diffs = [digits[i+1]-digits[i] for i in range(len(digits)-1)]
    if all(d==1 for d in diffs) or all(d== -1 for d in diffs):
        result["reason"]="Phone number cannot be sequential pattern"
        return json.dump(result)
    if len(set(cleaned[0::2]))==1 and len(set(cleaned[1::2]))==1:
        result["reason"]="Phone number cannot be an alternating pattern"
        return json.dump(result)
    
    result["is_valid"] = True
    result["reason"] = f"'{cleaned}' is a valid Indian mobile number"
    result["cleaned_phone"] = cleaned
    return json.dumps(result)


# Email Tool 

# Known popular email domains (used for reference/validation)
KNOWN_EMAIL_DOMAINS = {
    "gmail.com", "googlemail.com",
    "yahoo.com", "yahoo.in", "yahoo.co.in", "yahoo.co.uk",
    "outlook.com", "hotmail.com", "hotmail.in", "live.com", "msn.com",
    "icloud.com", "me.com", "mac.com",
    "rediffmail.com", "indiatimes.com",
    "protonmail.com", "zoho.com", "aol.com", "yandex.com",
    "company.com", "business.com",
}

# Common typos → correct domain mapping
DOMAIN_TYPO_MAP = {
    # Gmail typos
    "gmial.com": "gmail.com",
    "gmal.com": "gmail.com",
    "gamil.com": "gmail.com",
    "gmaill.com": "gmail.com",
    "gmai.com": "gmail.com",
    "gmailcom": "gmail.com",
    "gmail.con": "gmail.com",
    "gmail.cm": "gmail.com",
    "gmail.cpm": "gmail.com",
    "gmail.ocm": "gmail.com",
    "gnail.com": "gmail.com",
    "gmill.com": "gmail.com",
    # Yahoo typos
    "yaho.com": "yahoo.com",
    "yahooo.com": "yahoo.com",
    "yahho.com": "yahoo.com",
    "yhaoo.com": "yahoo.com",
    "yahoo.con": "yahoo.com",
    # Outlook/Hotmail typos
    "outook.com": "outlook.com",
    "outlok.com": "outlook.com",
    "hotmial.com": "hotmail.com",
    "hotmal.com": "hotmail.com",
    "hotmaill.com": "hotmail.com",
    # iCloud typos
    "iclould.com": "icloud.com",
    "icould.com": "icloud.com",
    # Rediffmail typos
    "redifmail.com": "rediffmail.com",
    "reddifmail.com": "rediffmail.com",
}


@tool
def validate_email_tool(email: str) -> str:
    """
    TOOL CALLING — Validates if the given string is a properly formatted email address.
    Checks: format (local@domain.tld), domain structure, no malformed parts, and common domain typos.
    Returns JSON: {is_valid: bool, reason: str, cleaned_email: str, suggested_domain: str}
    """
    result = {"is_valid": False, "reason": "", "cleaned_email": "", "suggested_domain": ""}
    s = email.strip().lower()
    pattern = r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$'

    if not re.match(pattern, s):
        result["reason"] = f"'{s}' is not a valid email format — expected format: name@domain.com"
        return json.dumps(result)

    parts = s.split('@')
    local_part = parts[0]
    domain = parts[1]
    domain_parts = domain.split('.')

    if local_part.isdigit():
        result["reason"]="Email address cannot have a numbers-only username"
        return json.dumps(result)

    if any(len(p) == 0 for p in domain_parts):
        result["reason"] = "Email domain is malformed"
        return json.dumps(result)

    # Check for known typos in domain
    if domain in DOMAIN_TYPO_MAP:
        correct = DOMAIN_TYPO_MAP[domain]
        result["reason"] = f"Domain '{domain}' appears to be a typo — did you mean '{correct}'? e.g., {local_part}@{correct}"
        result["suggested_domain"] = correct
        return json.dumps(result)

    result["is_valid"] = True
    result["reason"] = f"'{s}' is a valid email address"
    result["cleaned_email"] = s
    return json.dumps(result)

ALL_TOOLS = [validate_name_tool, validate_phone_tool, validate_email_tool]



