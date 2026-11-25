import sys
from pathlib import Path
import uvicorn

PORT = 8000


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"Starting FX Guard Enterprise FastAPI Server at http://127.0.0.1:{port}")
    uvicorn.run("api:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
