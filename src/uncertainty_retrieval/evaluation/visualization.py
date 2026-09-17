"""Deterministic image-based E2B failure-case visualizations."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def _load_thumbnail(path: Path, size: tuple[int, int]) -> Image.Image:
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail(size)
    except (FileNotFoundError, OSError):
        image = Image.new("RGB", size, "#eeeeee")
        ImageDraw.Draw(image).text((8, 8), "image unavailable", fill="black")
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return ImageOps.expand(canvas, border=2, fill="#444444")


def write_failure_contact_sheet(
    cases: list[dict], dataset_root: str | Path, destination: str | Path,
    title: str, maximum_cases: int = 8,
) -> None:
    """Show query, cosine Top-1 and reranked Top-1 for deterministic cases."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = cases[:maximum_cases]
    width, row_height, header = 1000, 190, 72
    sheet = Image.new("RGB", (width, header + max(1, len(rows)) * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((20, 16), title, fill="black", font=font)
    draw.text((20, 42), "Query                    Cosine Top-1              Reranked Top-1", fill="black", font=font)
    root = Path(dataset_root)
    if not rows:
        draw.text((20, header + 20), "No cases in this category.", fill="black", font=font)
    for row_index, case in enumerate(rows):
        y = header + row_index * row_height
        paths = (case["query_path"], case["baseline_paths"][0], case["reranked_paths"][0])
        for column, relative in enumerate(paths):
            image = _load_thumbnail(root / "images" / relative, (180, 140))
            sheet.paste(image, (20 + column * 250, y + 8))
        label = (
            f"q={case['query_id']} class={case['query_label']} | "
            f"base={case['baseline_labels'][0]} -> rerank={case['reranked_labels'][0]}"
        )
        draw.text((775, y + 20), label, fill="black", font=font)
        draw.text((775, y + 45), f"category={case['category']}", fill="black", font=font)
    sheet.save(destination, format="PNG", optimize=True)
