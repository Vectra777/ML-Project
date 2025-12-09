# main.py
import uvicorn
from app import app #our uvicorn app

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
