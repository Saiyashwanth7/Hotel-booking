# main.py  (run from the project root: python main.py)
import os

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if not (os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
    print("WARNING: OPENROUTER_API_KEY not set. /ask will fall back to retrieval-only answers "
          "for questions that need the LLM.")

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=os.environ.get("RELOAD", "0") == "1",
    )
