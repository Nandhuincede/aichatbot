# AIC Chatbot

A LangGraph-powered contact collection chatbot using tool calling and structured output.

## Project Structure

```
chatbot/
├── state.py        ← ChatState (TypedDict) + all Pydantic schemas
├── tools.py        ← @tool validators (name, phone, email)
├── chatbot.py      ← LLM factories, 3-step pipeline, LangGraph nodes & graph
├── app.py          ← Flask web server & API routes
├── database.py     ← SQLite helpers (sessions, contacts, logs)
├── templates/
│   ├── index.html  ← Chat UI
│   └── logs.html   ← Admin log viewer
├── .env.example    ← Copy to .env and add your API key
└── requirements.txt
```

## Architecture

Every user message goes through a **3-step validation pipeline**:

| Step | Technique | Purpose |
|------|-----------|---------|
| 1 | `with_structured_output()` | Extract the real value (name/phone) from any sentence |
| 2 | `bind_tools()` + `@tool` | LLM calls the right validator tool (regex + rules) |
| 3 | `with_structured_output()` | LLM reads tool result → returns clean verdict |

LangGraph routes every turn through a **router node** that reads `state["step"]`
and dispatches to the correct collection node automatically.

```
router → (reads state["step"]) → greeting / collect_name / collect_phone / collect_email / collect_description → END
```

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your API key
cp .env.example .env
# Edit .env and set GROQ_API_KEY=your_key

# 3. Run
python app.py
# Open http://localhost:5000
```

## Requirements

- Python 3.10+
- Groq API key (free at https://console.groq.com)
