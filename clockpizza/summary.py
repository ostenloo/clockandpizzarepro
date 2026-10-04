"""Shared reporting helpers for the experiment scripts: distributions, target tables."""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


def iqr(xs: Sequence[float]) -> tuple[float, float]:
    xs = sorted(xs)
    if not xs:
        return (float("nan"), float("nan"))
    if len(xs) < 4:
        return xs[0], xs[-1]
    q = st.quantiles(xs, n=4)
    return q[0], q[2]


def fmt_dist(xs: Sequence[float], places: int = 4) -> str:
    """``median (q1-q3, n=...)``, or ``-`` when empty."""
    xs = list(xs)
    if not xs:
        return "-"
    lo, hi = iqr(xs)
    return f"{st.median(xs):.{places}f} ({lo:.{places}f}-{hi:.{places}f}, n={len(xs)})"


def finished(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in runs if r.get("finished")]


def perfect(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runs the paper would keep: finished, with 100% validation accuracy."""
    return [r for r in runs if r["val_accuracy"] == 1.0]


def circular(runs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in runs if r["circular"]]


def by_experiment(registry: Iterable[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [r for r in registry if r["config"].get("experiment") == name]


@dataclass
class Targets:
    """Collects target verdicts and prints the §3 table."""

    rows: list[tuple[str, str, str, str, str, bool]] = field(default_factory=list)

    def record(self, tid: str, quantity: str, paper: str, ours: Any, band: str,
               ok: bool) -> None:
        self.rows.append((tid, quantity, paper, str(ours), band, bool(ok)))

    @property
    def failures(self) -> list[str]:
        return [r[0] for r in self.rows if not r[5]]

    def print(self, label: str) -> int:
        if not self.rows:
            print(f"\nNo {label} targets could be evaluated.")
            return 0
        print("\n## Targets\n")
        print("| ID | quantity | paper | ours | band | verdict |")
        print("| --- | --- | --- | --- | --- | --- |")
        for tid, quantity, paper, ours, band, ok in sorted(self.rows):
            print(f"| {tid} | {quantity} | {paper} | {ours} | {band} "
                  f"| {'**pass**' if ok else '**FAIL**'} |")
        bad = self.failures
        print(f"\n**{f'All {label} targets in band' if not bad else 'Out of band: ' + ', '.join(bad)}.**")
        return 1 if bad else 0


def print_figures(made: dict[str, Any]) -> None:
    if not made:
        return
    print("\n## Figures\n")
    print("| figure | file |")
    print("| --- | --- |")
    for key, path in sorted(made.items()):
        print(f"| {key} | `{path}` |")
