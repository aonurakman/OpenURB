#!/usr/bin/env python3
"""
Generate a static leaderboard HTML page from experiment outputs under results/.

The script scans each subdirectory in the provided results directory. If a
subdirectory contains both an exp_config.json and a metrics/BenchmarkMetrics.csv
(case-insensitive) file, it is included on the leaderboard. Experiments are
grouped by exp_type, env_config, task_config, and network and rendered into tabs
with sortable tables.
"""

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence


METRIC_FILENAMES = [
    "metrics/BenchmarkMetrics.csv",
    "metrics/benchmarkMetrics.csv",
    "metrics/benchmarkmetrics.csv",
]

VERSIONED_ID_RE = re.compile(r"^(?P<base>.+)_v(?P<version>\d+)$")

REQUIRED_STRING_KEYS = [
    "page_title",
    "title_link_text",
    "title_link_url",
    "title_suffix",
    "meta_text",
    "stats_loading",
    "hero_text",
    "controls_hint",
    "download_csv_label",
    "filters_title",
    "filters_hint",
    "filter_exp_type_label",
    "filter_env_label",
    "filter_task_label",
    "filter_network_label",
    "filter_action_all",
    "filter_action_none",
    "isolate_label",
    "show_all_label",
    "deselect_label",
    "collapse_folds_label",
    "merge_folds_label",
    "filter_summary",
    "filter_empty",
    "compare_column_label",
    "compare_title",
    "compare_label_left",
    "compare_label_right",
    "compare_slider_label",
    "footer_hint",
    "footer_separator",
    "generated_label",
    "preview_title",
    "preview_alt",
    "preview_fallback",
    "logo_alt",
    "stats_pill",
    "combo_summary",
    "combo_empty",
    "type_tab_count",
    "tab_env_label",
    "tab_task_label",
    "tab_network_label",
    "tab_separator",
    "combo_key_separator",
    "unknown_env",
    "unknown_task",
    "unknown_network",
    "empty_no_experiments",
    "empty_no_group",
    "na_label",
    "folds_tooltip",
    "preview_title_template",
    "preview_alt_template",
    "preview_image_path",
    "csv_filename",
    "slug_default",
    "slug_all",
    "sort_indicator_asc",
    "sort_indicator_desc",
    "type_labels",
    "table_headers",
    "metric_descriptions",
]

REQUIRED_TABLE_HEADERS = [
    "exp_id",
    "algorithm",
    "script",
    "alg_config",
    "env_seed",
    "torch_seed",
]

REQUIRED_TYPE_LABELS = [
    "normal",
    "open",
    "cond_open",
]


def read_metrics(exp_dir: Path) -> Optional[Dict[str, str]]:
    """Return the metrics dict (first row) and header order if present."""
    metrics_path = None
    for candidate in METRIC_FILENAMES:
        path = exp_dir / candidate
        if path.exists():
            metrics_path = path
            break

    if not metrics_path:
        return None

    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None
        first_row = rows[0]
        # Preserve header order for consistent column layout.
        header_order = reader.fieldnames or list(first_row.keys())
        return {"data": first_row, "header": header_order}


def read_config(exp_dir: Path) -> Optional[Dict]:
    config_path = exp_dir / "exp_config.json"
    if not config_path.exists():
        return None
    with config_path.open() as f:
        return json.load(f)


def split_versioned_id(exp_id: str) -> tuple[str, Optional[int]]:
    match = VERSIONED_ID_RE.match(exp_id)
    if not match:
        return exp_id, None
    return match.group("base"), int(match.group("version"))


def average_metrics(experiments: Sequence[Dict], anchor_metrics: Dict[str, str]) -> Dict[str, object]:
    metric_keys = set()
    for exp in experiments:
        metric_keys.update((exp.get("metrics") or {}).keys())

    averaged: Dict[str, object] = {}
    for key in metric_keys:
        values: List[float] = []
        for exp in experiments:
            value = (exp.get("metrics") or {}).get(key)
            if value is None or value == "":
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(parsed):
                continue
            values.append(parsed)
        if values:
            averaged[key] = sum(values) / len(values)
        elif key in anchor_metrics:
            averaged[key] = anchor_metrics[key]
        else:
            averaged[key] = ""
    return averaged


def collapse_versioned_experiments(experiments: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for exp in experiments:
        base_id, _ = split_versioned_id(exp["exp_id"])
        grouped.setdefault(base_id, []).append(exp)

    collapsed: List[Dict] = []
    for base_id, group in grouped.items():
        versions = [split_versioned_id(exp["exp_id"])[1] for exp in group]
        has_versions = any(version is not None for version in versions)
        if not has_versions or len(group) == 1:
            exp = group[0]
            exp["fold_count"] = 1
            collapsed.append(exp)
            continue

        anchor = next((exp for exp in group if exp["exp_id"] == base_id), None)
        if anchor is None:
            anchor = sorted(
                group,
                key=lambda item: split_versioned_id(item["exp_id"])[1] or 0,
            )[0]

        merged = dict(anchor)
        merged["exp_id"] = base_id
        merged["metrics"] = average_metrics(group, anchor.get("metrics") or {})
        merged["metric_order"] = anchor.get("metric_order")
        merged["fold_count"] = len(group)
        if merged["fold_count"] > 1:
            env_seeds = {
                exp.get("env_seed") for exp in group if exp.get("env_seed") not in (None, "")
            }
            torch_seeds = {
                exp.get("torch_seed") for exp in group if exp.get("torch_seed") not in (None, "")
            }
            if len(env_seeds) > 1:
                merged["env_seed"] = "varies"
            elif env_seeds and merged.get("env_seed") in (None, ""):
                merged["env_seed"] = next(iter(env_seeds))
            if len(torch_seeds) > 1:
                merged["torch_seed"] = "varies"
            elif torch_seeds and merged.get("torch_seed") in (None, ""):
                merged["torch_seed"] = next(iter(torch_seeds))
        collapsed.append(merged)

    return collapsed


def validate_strings(strings: Dict, strings_path: Path) -> None:
    missing = [key for key in REQUIRED_STRING_KEYS if key not in strings]
    errors = []
    if missing:
        errors.append(f"Missing keys: {', '.join(sorted(missing))}")

    if not isinstance(strings.get("type_labels"), dict):
        errors.append("type_labels must be a JSON object")
    else:
        missing_type_labels = [key for key in REQUIRED_TYPE_LABELS if key not in strings["type_labels"]]
        if missing_type_labels:
            errors.append(f"type_labels missing: {', '.join(sorted(missing_type_labels))}")

    if not isinstance(strings.get("table_headers"), dict):
        errors.append("table_headers must be a JSON object")
    else:
        missing_headers = [key for key in REQUIRED_TABLE_HEADERS if key not in strings["table_headers"]]
        if missing_headers:
            errors.append(f"table_headers missing: {', '.join(sorted(missing_headers))}")

    if not isinstance(strings.get("metric_descriptions"), dict):
        errors.append("metric_descriptions must be a JSON object")

    if errors:
        detail = "; ".join(errors)
        raise SystemExit(f"Invalid strings file at {strings_path}: {detail}")


def load_strings(strings_path: Path) -> Dict:
    if not strings_path.exists():
        raise SystemExit(f"Strings file not found: {strings_path}")
    try:
        with strings_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Strings file is not valid JSON: {strings_path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Strings file must contain a JSON object: {strings_path}")
    validate_strings(payload, strings_path)
    return payload


def load_template(template_path: Path) -> str:
    if not template_path.exists():
        raise SystemExit(f"Template file not found: {template_path}")
    try:
        return template_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Unable to read template file: {template_path}") from exc


def collect_experiments(results_dir: Path) -> List[Dict]:
    experiments: List[Dict] = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        if exp_dir.name == "sample_results":
            continue

        config = read_config(exp_dir)
        metrics = read_metrics(exp_dir)
        if not config or not metrics:
            continue

        script_path = config.get("script") or ""
        experiments.append(
            {
                "exp_id": exp_dir.name,
                "exp_path": str(exp_dir.as_posix()),
                "exp_type": config.get("exp_type", "normal"),
                "env_config": config.get("env_config"),
                "task_config": config.get("task_config"),
                "network": config.get("network"),
                "algorithm": config.get("algorithm"),
                "script": Path(script_path).name if script_path else "",
                "alg_config": config.get("alg_config")
                or config.get("algorithm_config")
                or config.get("algorithm_configuration")
                or "",
                "env_seed": config.get("env_seed"),
                "torch_seed": config.get("torch_seed"),
                "metrics": metrics["data"],
                "metric_order": metrics["header"],
            }
        )
    return experiments


def build_html(payload: Dict, output_path: Path, template: str) -> None:
    """Write a self-contained HTML file with embedded data and styling."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, indent=2)
    generated_at = payload["generated_at"]
    strings = payload["strings"]

    replacements = {
        "__PAGE_TITLE__": strings["page_title"],
        "__TITLE_LINK_URL__": strings["title_link_url"],
        "__TITLE_LINK_TEXT__": strings["title_link_text"],
        "__TITLE_SUFFIX__": strings["title_suffix"],
        "__META_TEXT__": strings["meta_text"],
        "__STATS_LOADING__": strings["stats_loading"],
        "__HERO_TEXT__": strings["hero_text"],
        "__CONTROLS_HINT__": strings["controls_hint"],
        "__DOWNLOAD_LABEL__": strings["download_csv_label"],
        "__FILTERS_TITLE__": strings["filters_title"],
        "__FILTERS_HINT__": strings["filters_hint"],
        "__ISOLATE_LABEL__": strings["isolate_label"],
        "__SHOW_ALL_LABEL__": strings["show_all_label"],
        "__DESELECT_LABEL__": strings["deselect_label"],
        "__COLLAPSE_FOLDS_LABEL__": strings["collapse_folds_label"],
        "__MERGE_FOLDS_LABEL__": strings["merge_folds_label"],
        "__COMPARE_TITLE__": strings["compare_title"],
        "__COMPARE_SLIDER_LABEL__": strings["compare_slider_label"],
        "__FOOTER_HINT__": strings["footer_hint"],
        "__FOOTER_SEPARATOR__": strings["footer_separator"],
        "__GENERATED_LABEL__": strings["generated_label"],
        "__PREVIEW_TITLE__": strings["preview_title"],
        "__PREVIEW_ALT__": strings["preview_alt"],
        "__PREVIEW_FALLBACK__": strings["preview_fallback"],
        "__LOGO_ALT__": strings["logo_alt"],
        "__GENERATED_AT__": generated_at,
        "__DATA__": data_json,
    }

    html = template
    for token, value in replacements.items():
        html = html.replace(token, str(value))
    output_path.write_text(html, encoding="utf-8")


def main(args: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a static leaderboard page.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Base directory containing experiment result subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("leaderboard_site"),
        help="Directory where the static site will be written.",
    )
    parser.add_argument(
        "--repo-url",
        type=str,
        default="",
        help="Optional base URL for experiment links (e.g., https://github.com/org/repo/tree/main/).",
    )
    parser.add_argument(
        "--strings-path",
        type=Path,
        default=Path(__file__).resolve().parent / "leaderboard_strings.json",
        help="JSON file containing UI strings and metric descriptions.",
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        default=Path(__file__).resolve().parent / "leaderboard_template.html",
        help="HTML template file for the leaderboard page.",
    )
    parsed = parser.parse_args(args)

    strings = load_strings(parsed.strings_path)
    template = load_template(parsed.template_path)
    raw_experiments = collect_experiments(parsed.results_dir)
    experiments = collapse_versioned_experiments(raw_experiments)
    base_url = parsed.repo_url.rstrip("/") + "/" if parsed.repo_url else ""
    for exp in raw_experiments:
        exp["exp_link"] = base_url + exp["exp_path"]
    for exp in experiments:
        exp["exp_link"] = base_url + exp["exp_path"]

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "results_dir": str(parsed.results_dir),
        "experiments": experiments,
        "raw_experiments": raw_experiments,
        "strings": strings,
    }

    output_path = parsed.output_dir / "index.html"
    build_html(payload, output_path, template)
    icon_src = Path("docs/openurb.png")
    if icon_src.exists():
        shutil.copy(icon_src, parsed.output_dir / "openurb.png")
    print(f"Wrote leaderboard to {output_path}")


if __name__ == "__main__":
    main()
