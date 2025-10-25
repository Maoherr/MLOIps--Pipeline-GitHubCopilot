import sys
import os

# Añadir la raíz del repo al PYTHONPATH para que `import src...` funcione
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)