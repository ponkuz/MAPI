from __future__ import annotations

import pandas as pd


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy()
    display = display.map(
        lambda value: f"{value:.6g}" if isinstance(value, float) else str(value)
    )
    headers = [str(column).replace("|", "\\|") for column in display.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.itertuples(index=False, name=None):
        cells = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_markdown_report(
    title: str,
    backtest_rows: pd.DataFrame,
    ablation_rows: pd.DataFrame | None = None,
) -> str:
    lines = [
        f"# {title}",
        "",
        "MAPI is experimental. This report is evidence for or against a research hypothesis, not a profitability claim.",
        "",
        "## Event Study Results",
        "",
        _markdown_table(backtest_rows) if not backtest_rows.empty else "No backtest rows.",
    ]
    if ablation_rows is not None:
        lines.extend(
            [
                "",
                "## Ablation Results",
                "",
                _markdown_table(ablation_rows)
                if not ablation_rows.empty
                else "No ablation rows.",
            ]
        )
    lines.extend(
        [
            "",
            "## Required Review",
            "",
            "- Confirm that data is point-in-time and not revised after the fact.",
            "- Review transaction cost, spread, slippage, and signal delay assumptions.",
            "- Compare against negative controls before changing parameters.",
            "- Keep an untouched test set isolated until methodology is finalized.",
        ]
    )
    return "\n".join(lines)
