# viz/config.py
from pathlib import Path
import json
from matplotlib import colormaps

BASE_DIR = Path(__file__).resolve().parent.parent

COLORS = (
    ["red", "purple", "orange", "green", "dodgerblue", "magenta", "gold",
     "darkorange", "darkolivegreen", "cyan", "lightblue", "lightgreen",
     "darkkhaki", "brown", "lime", "orangered", "blue", "mediumpurple",
     "turquoise"]
    + list(colormaps["Dark2"].colors)
    + list(colormaps["Set2"].colors)
    + list(colormaps["Pastel1"].colors)
    + ["yellow", "silver"]
)

with open(BASE_DIR / "json_files" / "cluster_colors.json", encoding="utf-8") as f:
    CLUSTER_COLOR_MAPPING = json.load(f)

with open(BASE_DIR / "json_files" / "va_positions.json", encoding="utf-8") as f:
    VA_POSITIONS = json.load(f)

HA_POSITIONS = {
    "Zonguldak": "right",
    "Adana": "right",
    "Yalova": "right",
}
