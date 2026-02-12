from __future__ import annotations

import argparse
import csv
import glob
import html
import shutil
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build polished HTML evidence packs from counterfactual CSV outputs "
            "(input image vs edited image with metadata)."
        )
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=[],
        help="CSV file path. Can be provided multiple times.",
    )
    parser.add_argument(
        "--csv-glob",
        action="append",
        default=[],
        help='Glob pattern for CSV files (example: "data/03_eval_results/place_pulse_bottom5_5each_*.csv").',
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path.cwd()),
        help="Project root used to resolve relative image paths in CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/05_evidence_pack",
        help="Output directory for evidence packs.",
    )
    return parser.parse_args()


def collect_csv_paths(args: argparse.Namespace) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []

    for csv_path in args.csv:
        path = Path(csv_path).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            paths.append(path)

    for pattern in args.csv_glob:
        for raw in sorted(glob.glob(pattern)):
            path = Path(raw).expanduser().resolve()
            if path not in seen:
                seen.add(path)
                paths.append(path)

    return paths


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_attribute_label(csv_path: Path) -> str:
    stem = csv_path.stem
    prefixes = (
        "place_pulse_bottom5_5each_",
        "place_pulse_bottom5_",
        "counterfactual_results_",
    )
    for prefix in prefixes:
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem.replace("_", " ").strip() or "counterfactual attribute"


def parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y"}


def resolve_image_path(raw: str, project_root: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def asset_copy(source: Path, assets_dir: Path, base_name: str) -> tuple[str, bool]:
    if not source.exists():
        return "", False
    suffix = source.suffix.lower() if source.suffix else ".jpg"
    dest = assets_dir / f"{base_name}{suffix}"
    shutil.copy2(source, dest)
    return f"assets/{dest.name}", True


def render_panel(
    label: str,
    image_rel_path: str,
    source_path: Path,
    exists: bool,
    alt_text: str,
) -> str:
    source_text = html.escape(str(source_path))
    if exists:
        return f"""
        <figure class="image-panel">
          <figcaption>{html.escape(label)}</figcaption>
          <img src="{html.escape(image_rel_path)}" alt="{html.escape(alt_text)}" loading="lazy" />
          <p class="image-path">{source_text}</p>
        </figure>
        """
    return f"""
    <figure class="image-panel missing">
      <figcaption>{html.escape(label)}</figcaption>
      <div class="missing-box">Image not found</div>
      <p class="image-path">{source_text}</p>
    </figure>
    """


def build_html(
    *,
    attribute: str,
    source_csv: Path,
    sections_html: str,
    num_rows: int,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Evidence Pack - {html.escape(attribute)}</title>
  <style>
    @import url("https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,500;9..144,700&display=swap");
    :root {{
      --bg: #f4f1ea;
      --ink: #1d1b17;
      --muted: #5f594f;
      --card: #fffdfa;
      --line: #d8d0c2;
      --accent: #b85d2a;
      --accent-soft: #f0d7c9;
      --good: #2e7d32;
      --bad: #b71c1c;
      --shadow: 0 14px 36px rgba(35, 28, 20, 0.08);
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Space Grotesk", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 90% 0%, #fbe7d8 0, transparent 38%),
        radial-gradient(circle at 0% 100%, #e6efe9 0, transparent 32%),
        var(--bg);
      line-height: 1.45;
    }}
    .wrap {{
      width: min(1140px, 92vw);
      margin: 0 auto;
      padding: 40px 0 64px;
    }}
    .hero {{
      background: linear-gradient(140deg, #fffdf7 0%, #f9f2ea 100%);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      padding: 28px 30px;
      margin-bottom: 24px;
    }}
    .kicker {{
      margin: 0;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }}
    h1 {{
      margin: 8px 0 10px;
      font-family: "Fraunces", serif;
      font-size: clamp(30px, 5vw, 46px);
      line-height: 1.08;
      letter-spacing: -0.02em;
    }}
    .hero-meta {{
      color: var(--muted);
      font-size: 14px;
      margin: 0;
    }}
    .pair {{
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      padding: 18px;
      margin-bottom: 18px;
      animation: fade 0.42s ease both;
    }}
    @keyframes fade {{
      from {{ opacity: 0; transform: translateY(8px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}
    .pair-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 12px;
    }}
    .pair-id {{
      margin: 0 0 4px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
    }}
    .pair-subtitle {{
      margin: 0;
      font-size: 18px;
      font-weight: 700;
      color: var(--ink);
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .chip {{
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fff;
      padding: 5px 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .chip.good {{
      border-color: #9ec7a0;
      background: #edf8ee;
      color: var(--good);
    }}
    .chip.bad {{
      border-color: #e2a4a4;
      background: #fdeeee;
      color: var(--bad);
    }}
    .pair-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .image-panel {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 14px;
      overflow: hidden;
      background: #f7f2ea;
    }}
    figcaption {{
      padding: 10px 12px 8px;
      font-weight: 700;
      color: var(--ink);
      background: linear-gradient(90deg, #fff, #fbf7ef);
      border-bottom: 1px solid var(--line);
    }}
    .image-panel img {{
      display: block;
      width: 100%;
      max-height: 410px;
      object-fit: contain;
      background: #0f0e0c;
    }}
    .image-path {{
      margin: 0;
      padding: 8px 12px 10px;
      font-size: 11px;
      color: var(--muted);
      word-break: break-all;
      background: #fffdfa;
      border-top: 1px solid var(--line);
    }}
    .missing-box {{
      min-height: 240px;
      display: grid;
      place-items: center;
      border: 2px dashed var(--line);
      margin: 12px;
      border-radius: 10px;
      color: var(--muted);
      background: #fff;
    }}
    .description {{
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      background: #fff;
    }}
    .description h3 {{
      margin: 0 0 6px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--accent);
    }}
    .description p {{
      margin: 0;
      color: var(--ink);
    }}
    .description p + h3 {{
      margin-top: 10px;
    }}
    @media (max-width: 920px) {{
      .pair-grid {{
        grid-template-columns: 1fr;
      }}
      .pair-head {{
        flex-direction: column;
      }}
      .chips {{
        justify-content: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <p class="kicker">Streetview Counterfactual Evidence Pack</p>
      <h1>{html.escape(attribute)}</h1>
      <p class="hero-meta">Source CSV: {html.escape(str(source_csv))}</p>
      <p class="hero-meta">Pairs: {num_rows} | Built: {now}</p>
    </section>
    {sections_html}
  </main>
</body>
</html>
"""


def build_pack_for_csv(csv_path: Path, project_root: Path, output_root: Path) -> Path:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows: list[dict[str, str]] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]

    attribute = normalize_attribute_label(csv_path)
    pack_dir = output_root / csv_path.stem
    assets_dir = pack_dir / "assets"
    ensure_dir(assets_dir)

    manifest_rows: list[dict[str, str]] = []
    section_blocks: list[str] = []

    for idx, row in enumerate(rows, start=1):
        input_raw = row.get("input_image_path", "").strip()
        output_raw = row.get("output_image_path", "").strip()

        input_path = resolve_image_path(input_raw, project_root) if input_raw else Path("")
        output_path = resolve_image_path(output_raw, project_root) if output_raw else Path("")

        input_rel, input_exists = asset_copy(input_path, assets_dir, f"{idx:02d}_original")
        output_rel, output_exists = asset_copy(output_path, assets_dir, f"{idx:02d}_edited")

        target_object = (row.get("planner_target_object", "") or "").strip() or "unspecified object"
        edit_plan = (row.get("planner_edit_plan", "") or "").strip() or "No edit plan recorded."
        critic_notes = (row.get("critic_notes", "") or "").strip() or "No critic notes recorded."
        realistic = parse_bool(row.get("critic_is_realistic"))
        minimal = parse_bool(row.get("critic_is_minimal_edit"))

        subtitle = f"Edited object: {target_object}"
        chips = [
            f'<span class="chip">Object: {html.escape(target_object)}</span>',
            (
                '<span class="chip good">Realistic: Yes</span>'
                if realistic
                else '<span class="chip bad">Realistic: No</span>'
            ),
            (
                '<span class="chip good">Minimal: Yes</span>'
                if minimal
                else '<span class="chip bad">Minimal: No</span>'
            ),
        ]

        left_panel = render_panel(
            label="Original",
            image_rel_path=input_rel,
            source_path=input_path,
            exists=input_exists,
            alt_text=f"{attribute} pair {idx} original",
        )
        right_panel = render_panel(
            label="Edited",
            image_rel_path=output_rel,
            source_path=output_path,
            exists=output_exists,
            alt_text=f"{attribute} pair {idx} edited",
        )

        section_blocks.append(
            f"""
            <section class="pair">
              <header class="pair-head">
                <div>
                  <p class="pair-id">Pair {idx:02d}</p>
                  <p class="pair-subtitle">{html.escape(subtitle)}</p>
                </div>
                <div class="chips">{''.join(chips)}</div>
              </header>
              <div class="pair-grid">
                {left_panel}
                {right_panel}
              </div>
              <div class="description">
                <h3>Description</h3>
                <p>{html.escape(edit_plan)}</p>
                <h3>Critic Notes</h3>
                <p>{html.escape(critic_notes)}</p>
              </div>
            </section>
            """
        )

        manifest_rows.append(
            {
                "pair_id": str(idx),
                "attribute": attribute,
                "target_object": target_object,
                "edit_description": edit_plan,
                "critic_is_realistic": str(realistic),
                "critic_is_minimal_edit": str(minimal),
                "critic_notes": critic_notes,
                "source_input_image_path": str(input_path),
                "source_output_image_path": str(output_path),
                "pack_input_image_path": input_rel,
                "pack_output_image_path": output_rel,
            }
        )

    html_doc = build_html(
        attribute=attribute,
        source_csv=csv_path,
        sections_html="\n".join(section_blocks),
        num_rows=len(rows),
    )
    ensure_dir(pack_dir)
    output_html = pack_dir / "index.html"
    output_html.write_text(html_doc)

    manifest_path = pack_dir / "evidence_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "attribute",
                "target_object",
                "edit_description",
                "critic_is_realistic",
                "critic_is_minimal_edit",
                "critic_notes",
                "source_input_image_path",
                "source_output_image_path",
                "pack_input_image_path",
                "pack_output_image_path",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return output_html


def build_index(pack_paths: list[Path], output_root: Path) -> Path:
    entries = []
    for pack_html in sorted(pack_paths):
        rel = pack_html.relative_to(output_root)
        name = rel.parts[0].replace("_", " ")
        entries.append(
            f'<li><a href="{html.escape(str(rel))}">{html.escape(name)}</a></li>'
        )

    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Evidence Packs</title>
  <style>
    body {{
      font-family: "Space Grotesk", sans-serif;
      margin: 0;
      background: #f4f1ea;
      color: #1d1b17;
    }}
    main {{
      width: min(800px, 92vw);
      margin: 40px auto;
      background: #fffdfa;
      border: 1px solid #d8d0c2;
      border-radius: 16px;
      padding: 20px 24px;
    }}
    h1 {{
      margin-top: 0;
      font-size: 28px;
    }}
    ul {{
      margin: 0;
      padding-left: 20px;
    }}
    li {{
      margin: 10px 0;
    }}
    a {{
      color: #9c4717;
      text-decoration: none;
      font-weight: 700;
    }}
    a:hover {{
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Evidence Packs</h1>
    <ul>
      {''.join(entries)}
    </ul>
  </main>
</body>
</html>
"""
    index_path = output_root / "index.html"
    index_path.write_text(index_html)
    return index_path


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser()
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    ensure_dir(output_root)

    csv_paths = collect_csv_paths(args)
    if not csv_paths:
        raise SystemExit("No CSV files provided. Use --csv or --csv-glob.")

    pack_paths: list[Path] = []
    for csv_path in csv_paths:
        pack_html = build_pack_for_csv(
            csv_path=csv_path,
            project_root=project_root,
            output_root=output_root,
        )
        print(f"Wrote evidence pack: {pack_html}")
        pack_paths.append(pack_html)

    index_path = build_index(pack_paths, output_root)
    print(f"Wrote index: {index_path}")


if __name__ == "__main__":
    main()
