import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from streamlit_app import main as streamlit_main


def main() -> None:
	streamlit_main()


if __name__ == "__main__":
	main()
