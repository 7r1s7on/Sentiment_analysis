import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

def get_root_dir(sub_dir: str) -> str:
    """returns path to a project subdirectory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), sub_dir))
