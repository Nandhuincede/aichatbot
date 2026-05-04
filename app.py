import os
import uuid
import logging
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('chatbot_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
CORS(app)

session_store: dict = {}

from database import (
    init_db, create_session, update_session, close_session,
    log_message, log_error, save_contact,
    get_all_sessions, get_session_detail
)
from chatbot import process_message

init_db()



if not os.environ.get("GROQ_API_KEY"):
    logger.error("GROQ_API_KEY is not set! Chatbot will not work.")
    # Log to DB as a startup error (no session_id)
    try:
        log_error(None, "STARTUP_ERROR", "GROQ_API_KEY is not set in environment", "Set GROQ_API_KEY in your .env file")
    except Exception:
        pass



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logs')
def logs_page():
    return render_template('logs.html')

@app.route('/api/session/start', methods=['POST'])
def start_session():
    session_id = None
    try:
        
        if not os.environ.get("GROQ_API_KEY"):
            session_id = str(uuid.uuid4())
            create_session(session_id)
            log_error(session_id, "MISSING_API_KEY", "GROQ_API_KEY is not set", "Add GROQ_API_KEY=your_key to .env file")
            return jsonify({"error": "API key not configured. Please set GROQ_API_KEY in .env"}), 500

        session_id = str(uuid.uuid4())
        create_session(session_id)
        session_store[session_id] = {}

        bot_response, new_state = process_message(session_id, {}, None)
        new_state["session_id"] = session_id
        session_store[session_id] = new_state

        log_message(session_id, "bot", bot_response, "greeting")
        logger.info(f"New session started: {session_id}")

        return jsonify({"session_id": session_id, "message": bot_response, "step": new_state.get("step")})

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error starting session: {e}\n{tb}")
        
        try:
            log_error(session_id, "SESSION_START_ERROR", str(e), tb)
        except Exception as db_err:
            logger.error(f"Also failed to log error to DB: {db_err}")
        return jsonify({"error": "Failed to start session. Check server logs."}), 500

@app.route('/api/session/message', methods=['POST'])
def send_message():
    data = None
    session_id = None
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        user_message = data.get("message", "").strip()

        if not session_id or session_id not in session_store:
            return jsonify({"error": "Invalid or expired session. Please refresh."}), 400
        if not user_message:
            return jsonify({"error": "Empty message"}), 400

        
        if not os.environ.get("GROQ_API_KEY"):
            log_error(session_id, "MISSING_API_KEY", "GROQ_API_KEY is not set", "Add GROQ_API_KEY to .env file")
            return jsonify({"error": "API key not configured. Please set GROQ_API_KEY in .env"}), 500

        state = session_store[session_id]
        current_step = state.get("step", "greeting")

        if current_step == "done":
            return jsonify({"message": "This conversation has ended. Please start a new one.", "step": "done"})

        log_message(session_id, "user", user_message, current_step)

        bot_response, new_state = process_message(session_id, state, user_message)
        new_state["session_id"] = session_id
        session_store[session_id] = new_state

        new_step = new_state.get("step")
        log_message(session_id, "bot", bot_response, new_step)
        update_session(session_id)

        save_contact(
            session_id,
            name=new_state.get("name"),
            phone=new_state.get("phone"),
            email=new_state.get("email"),
            description=new_state.get("description")
        )

        
        if new_step == "done":
            close_session(session_id)

        logger.info(f"[{session_id}] step={new_step} input={user_message[:40]}")

        return jsonify({
            "message": bot_response,
            "step": new_step,
            "collected": {
                "name": new_state.get("name"),
                "phone": new_state.get("phone"),
                "email": new_state.get("email"),
            }
        })

    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Error processing message: {e}\n{tb}")
        try:
            log_error(session_id, "MESSAGE_ERROR", str(e), tb)
        except Exception as db_err:
            logger.error(f"Also failed to log error to DB: {db_err}")
        return jsonify({"error": "Something went wrong. Please try again."}), 500

@app.route('/api/session/close', methods=['POST'])
def close_session_route():
    """Called when user closes the chat window mid-conversation."""
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        if session_id and session_id in session_store:
            update_session(session_id)
            logger.info(f"Session closed by user: {session_id}")
            
            del session_store[session_id]
        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"Error closing session: {e}")
        return jsonify({"ok": False}), 500

@app.route('/api/logs/sessions', methods=['GET'])
def get_sessions():
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        status = request.args.get('status', '').strip() or None
        search = request.args.get('search', '').strip() or None
        result = get_all_sessions(page=page, per_page=per_page, status=status, search=search)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/logs/session/<session_id>', methods=['GET'])
def get_session(session_id):
    try:
        detail = get_session_detail(session_id)
        return jsonify(detail)
    except Exception as e:
        logger.error(f"Error fetching session {session_id}: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)