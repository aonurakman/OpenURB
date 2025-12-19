#!/usr/bin/env python3
"""
Generate a static leaderboard HTML page from experiment outputs under results/.

The script scans each subdirectory in the provided results directory. If a
subdirectory contains both an exp_config.json and a metrics/BenchmarkMetrics.csv
(case-insensitive) file, it is included on the leaderboard. Experiments are
grouped by env_config, task_config, and network and rendered into tabs with
sortable tables.
"""

import argparse
import csv
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence


METRIC_FILENAMES = [
    "metrics/BenchmarkMetrics.csv",
    "metrics/benchmarkMetrics.csv",
    "metrics/benchmarkmetrics.csv",
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

        experiments.append(
            {
                "exp_id": exp_dir.name,
                "exp_path": str(exp_dir.as_posix()),
                "env_config": config.get("env_config"),
                "task_config": config.get("task_config"),
                "network": config.get("network"),
                "algorithm": config.get("algorithm"),
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


def build_html(payload: Dict, output_path: Path) -> None:
    """Write a self-contained HTML file with embedded data and styling."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, indent=2)
    generated_at = payload["generated_at"]

    template = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenURB Leaderboard</title>
  <link rel="icon" type="image/png" href="openurb.png">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: radial-gradient(circle at 20% 20%, #f5f7ff, #e7ecff 25%, #f8f9ff 60%, #fefefe);
      --panel: #ffffffee;
      --muted: #6b7280;
      --accent: #3655ff;
      --accent-2: #00c6ae;
      --border: #e5e7eb;
      --shadow: 0 20px 60px rgba(45, 57, 105, 0.15);
      --radius: 16px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Space Grotesk', 'Segoe UI', sans-serif;
      background: var(--bg);
      color: #0f172a;
      min-height: 100vh;
      padding: 32px 24px 64px;
    }
    .page {
      max-width: 1200px;
      margin: 0 auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      margin-bottom: 28px;
      flex-wrap: wrap;
    }
    .title {
      font-size: 32px;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin: 0;
    }
    .meta {
      color: var(--muted);
      font-size: 14px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      border-radius: var(--radius);
      padding: 20px;
    }
    .hero {
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 16px;
      align-items: center;
      margin-bottom: 18px;
    }
    .hero img {
      width: 64px;
      height: 64px;
      border-radius: 14px;
      border: 1px solid var(--border);
      box-shadow: 0 10px 30px rgba(0,0,0,0.05);
      background: white;
      padding: 6px;
      object-fit: contain;
    }
    .hero p {
      margin: 0;
      color: #111827;
      line-height: 1.45;
    }
    .tabs {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }
    .tab {
      border: 1px solid var(--border);
      padding: 10px 14px;
      border-radius: 12px;
      background: #f8f9ff;
      color: #1f2937;
      cursor: pointer;
      transition: all 0.15s ease;
      font-weight: 600;
    }
    .tab.active {
      background: linear-gradient(120deg, var(--accent), var(--accent-2));
      color: white;
      border-color: transparent;
      box-shadow: 0 12px 30px rgba(54, 85, 255, 0.2);
    }
    .tab:hover {
      transform: translateY(-1px);
    }
    .tab small {
      display: block;
      font-size: 12px;
      color: inherit;
      opacity: 0.9;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 8px;
      min-width: 960px;
    }
    thead th {
      text-align: left;
      font-size: 13px;
      color: #0b1224;
      border-bottom: 1px solid var(--border);
      padding: 10px 10px;
      position: sticky;
      top: 0;
      background: #fefefe;
      cursor: pointer;
      user-select: none;
    }
    tbody td {
      padding: 9px 10px;
      border-bottom: 1px solid var(--border);
      font-size: 13px;
      color: #111827;
      background: #ffffff;
    }
    tbody tr:hover td {
      background: #f6f8ff;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      background: #eef2ff;
      color: #312e81;
      font-weight: 600;
      font-size: 12px;
      border: 1px solid #d7ddff;
    }
    .empty {
      color: var(--muted);
      padding: 12px 0;
    }
    .controls {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 13px;
    }
    .sort-indicator {
      margin-left: 6px;
      font-size: 11px;
      opacity: 0.7;
    }
    a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }
    a:hover {
      text-decoration: underline;
    }
    .table-wrap {
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 12px;
    }
    @media (max-width: 720px) {
      header {
        flex-direction: column;
        align-items: flex-start;
      }
      .title {
        font-size: 26px;
      }
      thead th, tbody td {
        font-size: 12px;
        white-space: nowrap;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <header>
      <div>
        <h1 class="title">OpenURB Leaderboards</h1>
        <div class="meta">Grouped by tasks (environment configuration, task configuration, traffic network) | Generated __GENERATED_AT__</div>
      </div>
      <div class="pill" id="stats-pill">loading...</div>
    </header>

    <div class="card hero">
      <img src="openurb.png" alt="OpenURB logo">
      <p>
        OpenURB benchmarks mixed traffic where human-driven vehicles optimize individual travel time while AVs collaborate to minimize group travel time under dynamic population shifts. Leaderboards below compare algorithms across environment configurations, task dynamics, and traffic networks using benchmark KPIs.
      </p>
    </div>

    <div class="card">
      <div class="controls">
        <div>Click a column to sort | Click again to toggle direction</div>
        <div id="combo-count"></div>
      </div>
      <div class="tabs" id="tabs"></div>
      <div id="table-container"></div>
    </div>
  </div>

  <script>
    const leaderboardData = __DATA__;

    const tabsEl = document.getElementById('tabs');
    const tableContainer = document.getElementById('table-container');
    const comboCountEl = document.getElementById('combo-count');
    const statsPill = document.getElementById('stats-pill');

    const grouped = {};
    for (const exp of leaderboardData.experiments) {
      const comboKey = [
        exp.env_config || 'unknown-env',
        exp.task_config || 'unknown-task',
        exp.network || 'unknown-network',
      ].join(' | ');
      if (!grouped[comboKey]) grouped[comboKey] = [];
      grouped[comboKey].push(exp);
    }

    const comboKeys = Object.keys(grouped).sort();
    comboCountEl.textContent = comboKeys.length ? `${comboKeys.length} problem tabs` : 'No valid experiments found';
    statsPill.textContent = `${leaderboardData.experiments.length} experiments indexed`;

    let activeTab = comboKeys[0] || null;

    function renderTabs() {
      tabsEl.innerHTML = '';
      comboKeys.forEach((key) => {
        const btn = document.createElement('button');
        btn.className = 'tab' + (key === activeTab ? ' active' : '');
        const parts = key.split(' | ');
        btn.innerHTML = `<strong>Env: ${parts[0]}</strong><small>Task: ${parts[1]} | Network: ${parts[2]}</small>`;
        btn.onclick = () => {
          activeTab = key;
          renderTabs();
          renderTable();
        };
        tabsEl.appendChild(btn);
      });
    }

    function formattedValue(value) {
      if (value === null || value === undefined || value === '') return 'N/A';
      if (typeof value === 'number') return Number.isInteger(value) ? value.toString() : value.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
      const num = Number(value);
      if (!Number.isNaN(num)) {
        return Number.isInteger(num) ? num.toString() : num.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
      }
      return value;
    }

    function renderTable() {
      tableContainer.innerHTML = '';
      if (!activeTab) {
        tableContainer.innerHTML = '<div class="empty">No experiments to display.</div>';
        return;
      }
      const experiments = grouped[activeTab];
      if (!experiments || !experiments.length) {
        tableContainer.innerHTML = '<div class="empty">No experiments in this group.</div>';
        return;
      }

      const metricCols = [];
      const seen = new Set();
      experiments.forEach((exp) => {
        const ordered = exp.metric_order || Object.keys(exp.metrics || {});
        ordered.forEach((m) => {
          if (!seen.has(m)) {
            seen.add(m);
            metricCols.push(m);
          }
        });
        Object.keys(exp.metrics || {}).forEach((m) => {
          if (!seen.has(m)) {
            seen.add(m);
            metricCols.push(m);
          }
        });
      });

      const table = document.createElement('table');
      const thead = document.createElement('thead');
      const headerRow = document.createElement('tr');
      const baseHeaders = [
        { key: 'exp_id', label: 'Exp ID' },
        { key: 'algorithm', label: 'Algorithm' },
        { key: 'alg_config', label: 'Algorithm Config' },
        { key: 'env_seed', label: 'Env Seed' },
        { key: 'torch_seed', label: 'Torch Seed' },
      ];

      const sortState = { column: null, direction: 'desc' };

      function headerCell({ key, label }) {
        const th = document.createElement('th');
        th.textContent = label;
        th.dataset.key = key;
        th.onclick = () => applySort(key);
        th.appendChild(document.createElement('span')).className = 'sort-indicator';
        return th;
      }

      baseHeaders.forEach((h) => headerRow.appendChild(headerCell(h)));
      metricCols.forEach((m) => headerRow.appendChild(headerCell({ key: m, label: m })));
      thead.appendChild(headerRow);
      table.appendChild(thead);

      const tbody = document.createElement('tbody');
      experiments.forEach((exp) => {
        const tr = document.createElement('tr');
        const cells = [
          `<a href="${exp.exp_link || exp.exp_path}" target="_blank" rel="noopener">${exp.exp_id}</a>`,
          exp.algorithm || 'N/A',
          exp.alg_config || 'N/A',
          exp.env_seed ?? 'N/A',
          exp.torch_seed ?? 'N/A',
        ];
        cells.forEach((val) => {
          const td = document.createElement('td');
          td.innerHTML = val === null || val === undefined ? 'N/A' : val;
          tr.appendChild(td);
        });
        metricCols.forEach((metric) => {
          const td = document.createElement('td');
          td.textContent = formattedValue(exp.metrics?.[metric]);
          td.dataset.value = exp.metrics?.[metric] ?? '';
          tr.appendChild(td);
        });
        tbody.appendChild(tr);
      });
      table.appendChild(tbody);
      const wrap = document.createElement('div');
      wrap.className = 'table-wrap';
      wrap.appendChild(table);
      tableContainer.appendChild(wrap);

      function applySort(colKey) {
        if (sortState.column === colKey) {
          sortState.direction = sortState.direction === 'asc' ? 'desc' : 'asc';
        } else {
          sortState.column = colKey;
          sortState.direction = colKey === 'exp_id' ? 'asc' : 'desc';
        }

        const colIndex = [...headerRow.children].findIndex((th) => th.dataset.key === colKey);
        const rows = [...tbody.querySelectorAll('tr')];
        rows.sort((a, b) => {
          const av = a.children[colIndex].dataset.value ?? a.children[colIndex].textContent;
          const bv = b.children[colIndex].dataset.value ?? b.children[colIndex].textContent;
          const anum = Number(av);
          const bnum = Number(bv);
          if (!Number.isNaN(anum) && !Number.isNaN(bnum)) {
            return sortState.direction === 'asc' ? anum - bnum : bnum - anum;
          }
          return sortState.direction === 'asc'
            ? String(av).localeCompare(String(bv))
            : String(bv).localeCompare(String(av));
        });
        tbody.innerHTML = '';
        rows.forEach((r) => tbody.appendChild(r));

        // update indicators
        headerRow.querySelectorAll('.sort-indicator').forEach((el) => {
          el.textContent = '';
        });
        const th = headerRow.children[colIndex];
        const indicator = th.querySelector('.sort-indicator');
        indicator.textContent = sortState.direction === 'asc' ? '^' : 'v';
      }
    }

    renderTabs();
    renderTable();
  </script>
</body>
</html>
"""
    html = (
        template.replace("__GENERATED_AT__", generated_at)
        .replace("__DATA__", data_json)
    )
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
    parsed = parser.parse_args(args)

    experiments = collect_experiments(parsed.results_dir)
    base_url = parsed.repo_url.rstrip("/") + "/" if parsed.repo_url else ""
    for exp in experiments:
        exp["exp_link"] = base_url + exp["exp_path"]

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "results_dir": str(parsed.results_dir),
        "experiments": experiments,
    }

    output_path = parsed.output_dir / "index.html"
    build_html(payload, output_path)
    icon_src = Path("docs/openurb.png")
    if icon_src.exists():
        shutil.copy(icon_src, parsed.output_dir / "openurb.png")
    print(f"Wrote leaderboard to {output_path}")


if __name__ == "__main__":
    main()
