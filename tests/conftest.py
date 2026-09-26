import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["LAB_MOCK"] = "1"
os.environ["LAB_MOCK_DELAY"] = "0"
