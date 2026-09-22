import os
from dotenv import load_dotenv

load_dotenv()

# Map GEMINI_API_KEY to GOOGLE_API_KEY if needed to ensure absolute compatibility
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError(
        "Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set. Please ensure you have a .env file "
        "in the root directory with your API key, or set it in your environment variables."
    )