"""
app.py
~~~~~~
Flask application factory.
Routes live in api/routes.py  |  Prompts live in instructions/prompts.py
"""

import os
import logging
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv

from database import init_db, log_error
from api import bp

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("chatbot_app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))
CORS(app)

app.register_blueprint(bp)

init_db()




if not os.environ.get("GROQ_API_KEY"):
    logger.error("GROQ_API_KEY is not set! Chatbot will not work.")
    try:
        log_error(None, "STARTUP_ERROR", "GROQ_API_KEY is not set in environment",
                  "Set GROQ_API_KEY in your .env file")
    except Exception:
        pass

if __name__ == "__main__":
    app.run(debug=True, port=5000)
