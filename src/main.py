# main.py
import os
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if HuggingFace API token is set
if not os.environ.get("HUGGINGFACE_API_TOKEN"):
    print("WARNING: HUGGINGFACE_API_TOKEN not set. Set this environment variable to use the RAG system.")

# Run FastAPI app
if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)