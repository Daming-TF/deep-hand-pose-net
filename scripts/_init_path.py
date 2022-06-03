import os
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


project_path = os.path.dirname(os.path.dirname(__file__))
add_path(project_path)
