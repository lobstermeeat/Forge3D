"""
One sheet per run of a pictures-only test set: its reference pictures side by side, numbered
1-4 as --picks counts them, under the run's prompt. For choosing which picture becomes 3D.

    python ops/contact_sheets.py orainge-outputs/phase2 ops-out/private/phase2-pictures
"""

import json
import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont

SIZE = 512
GAP = 10
HEADER = 60


def sheet(folder: pathlib.Path, target: pathlib.Path) -> bool:
    state = json.loads((folder / "progress.json").read_text())
    pictures = [name for name in state["steps"].get("reference", {}).get("files", []) if (folder / name).exists()]
    if not pictures:
        return False
    width = GAP + len(pictures) * (SIZE + GAP)
    canvas = Image.new("RGB", (width, HEADER + SIZE + GAP), (24, 24, 27))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=26)
    draw.text((GAP, 16), f"{folder.name}:  {state.get('prompt')}", fill=(236, 236, 236), font=font)
    for number, name in enumerate(pictures, 1):
        x = GAP + (number - 1) * (SIZE + GAP)
        canvas.paste(Image.open(folder / name).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS), (x, HEADER))
        draw.rectangle([x, HEADER, x + 46, HEADER + 46], fill=(0, 0, 0))
        draw.text((x + 14, HEADER + 8), str(number), fill=(255, 176, 32), font=font)
    canvas.save(target / f"{folder.name}.jpg", quality=88)
    return True


def main() -> None:
    source, target = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    runs = sorted(folder for folder in source.iterdir() if (folder / "progress.json").exists())
    made = [folder.name for folder in runs if sheet(folder, target)]
    print(f"{len(made)} contact sheets in {target}")


if __name__ == "__main__":
    main()
