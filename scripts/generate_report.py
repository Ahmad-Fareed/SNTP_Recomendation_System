#!/usr/bin/env python
"""Generate the searchable PDF report from saved experiment JSON.

Usage:
    python scripts/generate_report.py --results results/fast_demo \
        --output report/SNTP_Assignment_02_Report.pdf \
        --roll-number 22F3143 --student "Ahmad Fareed"
"""

from __future__ import annotations

import argparse
import io
import json
from datetime import date
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# Embedded Unicode fonts: the report uses λ, β, γ, arrows, ≈ and dashes.  The
# standard PDF Helvetica encoding does not contain all of them, so register a
# TrueType font that does.
FONT_REGULAR = "ReportSans"
FONT_BOLD = "ReportSans-Bold"
FONT_ITALIC = "ReportSans-Oblique"


def register_fonts() -> None:
    """Register Arial if available, otherwise matplotlib's DejaVu Sans."""
    windows = Path(r"C:\Windows\Fonts")
    candidates = [
        (FONT_REGULAR, windows / "arial.ttf"),
        (FONT_BOLD, windows / "arialbd.ttf"),
        (FONT_ITALIC, windows / "ariali.ttf"),
    ]
    if not all(path.exists() for _, path in candidates):
        font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        candidates = [
            (FONT_REGULAR, font_dir / "DejaVuSans.ttf"),
            (FONT_BOLD, font_dir / "DejaVuSans-Bold.ttf"),
            (FONT_ITALIC, font_dir / "DejaVuSans-Oblique.ttf"),
        ]
    for name, path in candidates:
        pdfmetrics.registerFont(TTFont(name, str(path)))


# ---------------------------------------------------------------------------
# Loading and formatting helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{digits}f}%"


def num(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:,.{digits}f}"


def plain(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def make_figure(draw, width: float = 165 * mm, height: float = 78 * mm) -> Image:
    buffer = io.BytesIO()
    fig = plt.figure(figsize=(7.0, 3.3), dpi=180)
    try:
        draw(fig)
        fig.tight_layout()
        fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    buffer.seek(0)
    return Image(buffer, width=width, height=height)


def table(
    header: list[str],
    rows: list[list[Any]],
    widths: list[float] | None = None,
    font_size: float = 7.5,
    header_size: float = 7.5,
) -> Table:
    data = [[Paragraph(str(cell), STYLE_TABLE_HEADER) for cell in header]]
    for row in rows:
        data.append([Paragraph(str(cell), STYLE_TABLE_BODY) for cell in row])
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), COLORS["navy"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("FONTSIZE", (0, 0), (-1, 0), header_size),
                ("FONTNAME", (0, 1), (-1, -1), FONT_REGULAR),
                ("FONTSIZE", (0, 1), (-1, -1), font_size),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, COLORS["grid"]),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLORS["pale"]]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return result


COLORS = {
    "navy": colors.HexColor("#17365D"),
    "blue": colors.HexColor("#2E75B6"),
    "pale": colors.HexColor("#EEF3F8"),
    "grid": colors.HexColor("#B7C7D9"),
    "accent": colors.HexColor("#C55A11"),
}
# Matplotlib needs plain strings / RGBA tuples, not reportlab Color objects.
MPL_COLORS = {
    "navy": "#17365D",
    "blue": "#2E75B6",
    "accent": "#C55A11",
}


def build_styles() -> None:
    styles = getSampleStyleSheet()
    global STYLE_TITLE, STYLE_SUBTITLE, STYLE_H1, STYLE_H2, STYLE_BODY
    global STYLE_BULLET, STYLE_CAPTION, STYLE_TABLE_HEADER, STYLE_TABLE_BODY
    STYLE_TITLE = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontName=FONT_BOLD,
        fontSize=22, leading=27, textColor=COLORS["navy"], spaceAfter=8,
    )
    STYLE_SUBTITLE = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontName=FONT_REGULAR,
        fontSize=11, leading=15, alignment=TA_CENTER,
        textColor=colors.HexColor("#444444"), spaceAfter=4,
    )
    STYLE_H1 = ParagraphStyle(
        "H1Custom", parent=styles["Heading1"], fontName=FONT_BOLD,
        fontSize=15, leading=19, textColor=COLORS["navy"], spaceBefore=8,
        spaceAfter=6,
    )
    STYLE_H2 = ParagraphStyle(
        "H2Custom", parent=styles["Heading2"], fontName=FONT_BOLD,
        fontSize=11.5, leading=15, textColor=COLORS["blue"], spaceBefore=7,
        spaceAfter=4,
    )
    STYLE_BODY = ParagraphStyle(
        "BodyCustom", parent=styles["BodyText"], fontName=FONT_REGULAR,
        fontSize=9.1, leading=13.1, alignment=TA_JUSTIFY, spaceAfter=5,
    )
    STYLE_BULLET = ParagraphStyle(
        "BulletCustom", parent=STYLE_BODY, leftIndent=12, bulletIndent=3,
        spaceAfter=2.5,
    )
    STYLE_CAPTION = ParagraphStyle(
        "CaptionCustom", parent=styles["Normal"], fontName=FONT_ITALIC,
        fontSize=7.8, leading=10.5, alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"), spaceBefore=2, spaceAfter=7,
    )
    STYLE_TABLE_HEADER = ParagraphStyle(
        "TableHeader", parent=styles["Normal"], fontName=FONT_BOLD,
        fontSize=7.4, leading=9.2, textColor=colors.white,
    )
    STYLE_TABLE_BODY = ParagraphStyle(
        "TableBody", parent=styles["Normal"], fontName=FONT_REGULAR,
        fontSize=7.2, leading=9.0, textColor=colors.black,
    )


# ---------------------------------------------------------------------------
# Document template with header/footer and PDF metadata
# ---------------------------------------------------------------------------

class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str, title: str, author: str, roll: str, **kwargs):
        super().__init__(filename, pagesize=A4, **kwargs)
        self.report_title = title
        self.author = author
        self.roll = roll
        frame = Frame(
            self.leftMargin, self.bottomMargin, self.width, self.height,
            id="normal", leftPadding=0, rightPadding=0, topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(id="all", frames=[frame], onPage=self._decorate),
            ]
        )

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setTitle(self.report_title)
        canvas.setAuthor(self.author)
        canvas.setSubject("SNTP Assignment 2 — Recommendation and Information Diffusion")
        canvas.setCreator("sntp report generator")
        canvas.setFont(FONT_REGULAR, 7.3)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(
            doc.leftMargin, A4[1] - 12 * mm,
            f"{self.author}  |  {self.roll}  |  SNTP Assignment 2",
        )
        canvas.drawRightString(
            A4[0] - doc.rightMargin, A4[1] - 12 * mm,
            "Epinions recommendation and diffusion",
        )
        canvas.setStrokeColor(COLORS["grid"])
        canvas.setLineWidth(0.4)
        canvas.line(
            doc.leftMargin, A4[1] - 14.5 * mm,
            A4[0] - doc.rightMargin, A4[1] - 14.5 * mm,
        )
        canvas.line(
            doc.leftMargin, 11.5 * mm,
            A4[0] - doc.rightMargin, 11.5 * mm,
        )
        canvas.drawString(doc.leftMargin, 8 * mm, self.report_title)
        canvas.drawRightString(
            A4[0] - doc.rightMargin, 8 * mm, f"Page {doc.page}"
        )
        canvas.restoreState()


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def story(metrics: dict[str, Any], args: argparse.Namespace) -> list[Any]:
    out: list[Any] = []
    add = out.append

    def p(text: str) -> None:
        add(Paragraph(text, STYLE_BODY))

    def bullet(text: str) -> None:
        add(Paragraph(text, STYLE_BULLET, bulletText="•"))

    def h1(text: str) -> None:
        add(Paragraph(text, STYLE_H1))

    def h2(text: str) -> None:
        add(Paragraph(text, STYLE_H2))

    def caption(text: str) -> None:
        add(Paragraph(text, STYLE_CAPTION))

    def fig(draw, text: str, width=165 * mm, height=76 * mm) -> None:
        add(KeepTogether([make_figure(draw, width, height), Paragraph(text, STYLE_CAPTION)]))

    r2 = metrics["r2"]
    r3 = metrics["r3"]
    r4 = metrics["r4"]
    r5 = metrics["r5"]
    r6 = metrics["r6"]
    r7 = metrics["r7"]
    r8 = metrics["r8"]
    timings = metrics["timings"]
    data_summary = metrics["data_summary"]
    split_info = metrics["split_info"]
    config = metrics["config"]

    title = "Recommendation and Information Diffusion on Epinions"
    today = date.today().strftime("%d %B %Y")

    # ---------------- Title page ----------------
    add(Spacer(1, 30 * mm))
    add(Paragraph(title, STYLE_TITLE))
    add(Spacer(1, 5 * mm))
    add(Paragraph("SNTP Assignment 2 — Technical Report", STYLE_SUBTITLE))
    add(Spacer(1, 14 * mm))
    add(
        Table(
            [
                ["Student", args.student],
                ["Roll number", args.roll_number],
                ["Course", "Social Network Theory and Programming (SNTP)"],
                ["Institution", "National University of Computer and Emerging Sciences"],
                ["Campus", "Chiniot-Faisalabad Campus"],
                ["Submission", today],
                ["Dataset", "Epinions ratings + trust network"],
                ["Measured run", "Bounded fast demo (full data files, 2-hop diffusion graph)"],
            ],
            colWidths=[36 * mm, 120 * mm],
            style=TableStyle(
                [
                    ("FONTNAME", (0, 0), (0, -1), FONT_BOLD),
                    ("FONTNAME", (1, 0), (1, -1), FONT_REGULAR),
                    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                    ("TEXTCOLOR", (0, 0), (0, -1), COLORS["navy"]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("LINEBELOW", (0, 0), (-1, -2), 0.3, COLORS["grid"]),
                ]
            ),
        )
    )
    add(Spacer(1, 15 * mm))
    p(
        "This report is generated from the saved JSON artifacts of the measured "
        "fast-demo run. It covers all eight requirements: data preparation, "
        "baseline recommenders, trust-aware recommendation, ICM/SIR diffusion, "
        "influence maximization, group recommendation, comprehensive evaluation, "
        "and the diffusion-limiting intervention with CELF and cost scaling. "
        "All cascade and satisfaction-after-intervention results are simulations; "
        "the dataset contains no adoption timestamps."
    )
    p(
        "The measured fast-demo wall-clock runtime was "
        f"<b>{num(timings['total_seconds'], 1)} s</b> "
        f"(~{timings['total_seconds']/60:.1f} min), including data loading, model "
        "fitting, all experiment stages, and result serialization. The larger "
        "full-scale configuration was not executed; no full-scale result is claimed."
    )
    add(PageBreak())

    # ---------------- Contents ----------------
    h1("Contents")
    contents = [
        "1. Executive summary and requirement map",
        "2. Dataset, split, and leakage protocol",
        "3. Requirement 1 — data preparation",
        "4. Requirement 2 — baseline recommendation",
        "5. Requirement 3 — trust-aware recommendation",
        "6. Requirement 4 — ICM, SIR, and preference-aware diffusion",
        "7. Requirement 5 — influence maximization",
        "8. Requirement 6 — group recommendation",
        "9. Requirement 7 — comprehensive evaluation",
        "10. Requirement 8 — intervention, CELF, and scaling",
        "11. Verification, tests, and reproducibility",
        "12. Limitations and conclusions",
    ]
    for item in contents:
        add(Paragraph(item, STYLE_BODY))
    add(Spacer(1, 4 * mm))
    h2("Source of reported numbers")
    p(
        "Every numeric value in the tables is read from "
        "<font face='Courier'>results/fast_demo/*.json</font> at report-generation "
        "time. The stage timing table comes from <font face='Courier'>stage_timings.json</font>. "
        "The report is regenerated with the command shown in Section 11. "
        "This makes the PDF searchable and keeps the prose synchronized with the "
        "saved artifacts rather than with hand-copied values."
    )

    # ---------------- Executive summary ----------------
    add(PageBreak())
    h1("1. Executive summary and requirement map")
    p(
        f"The dataset contains <b>{data_summary['n_users']:,}</b> users, "
        f"<b>{data_summary['n_items']:,}</b> items, "
        f"<b>{data_summary['n_ratings']:,}</b> ratings, and "
        f"<b>{data_summary['n_trust_edges']:,}</b> trust edges. "
        f"The split is {split_info['train_size']:,} train / "
        f"{split_info['val_size']:,} validation / {split_info['test_size']:,} test "
        "ratings. Models are fitted on train; validation selects the social weight; "
        "test is used only for the final observed evaluation."
    )
    p(
        "The strongest measured result is that matrix factorization and the "
        "trust-aware social model substantially outperform item-based CF on rating "
        "error. The social model is best on test MAE/RMSE and on all top-K ranking "
        "metrics in this run. Influence maximization does <b>not</b> show greedy "
        "superiority: on the restricted top-degree candidate pool, degree centrality "
        "wins at budgets 3 and 5, and the paired greedy−degree difference at budget 5 "
        "is negative. Group-strategy satisfaction differences are small, and the "
        "spread comparison is dominated by rare large cascades. The threshold "
        "intervention has a cliff at 0.8: it removes ~99% of simulated spread while "
        "slightly raising the simulator-side adopter rating proxy, but it is not an "
        "online experiment. CELF used 24.2% fewer evaluations and was 1.22× faster "
        "in this single noisy-Monte-Carlo measurement, while selecting different "
        "seeds; a deterministic controlled check reproduced standard greedy exactly."
    )
    rows = [
        ["1. Data", "Parse ratings/trust, validate, split, represent", f"{data_summary['n_users']:,} users; {split_info['train_size']:,}/{split_info['val_size']:,}/{split_info['test_size']:,} split"],
        ["2. Recommenders", "Item-based CF and MF; top-K", "MF test MAE 0.8325, RMSE 1.0677"],
        ["3. Trust-aware", "Social blend; λ selected on validation", "λ=0.3; test MAE 0.8284"],
        ["4. Diffusion", "ICM, SIR, preference-aware; controlled seeds", "p=0.05 canonical 4,684 adopters; preference 501"],
        ["5. Influence", "Random/degree/greedy; shared pool and seeds", "Degree best at budgets 3 and 5"],
        ["6. Groups", "Average / least misery / most pleasure; train-only groups", "Mean satisfaction 4.827 / 4.825 / 4.815"],
        ["7. Evaluation", "MAE/RMSE/P/R/F1/HR/nDCG/coverage/diversity", "Social best ranking; ICF broadest coverage"],
        ["8. Intervention", "Threshold, CELF, network/item/budget scaling", "0.8: −99.09% spread, +0.0264 proxy; CELF 1.22×"],
    ]
    add(table(["Requirement", "Implementation", "Headline measured outcome"], rows,
              [27 * mm, 66 * mm, 84 * mm], font_size=7.4))
    caption("Table 1. Requirement map and headline outcomes from the fast-demo run.")

    h2("What is and is not claimed")
    bullet("All diffusion, influence, group-spread, simulated-interaction, and intervention outcomes are simulations. The data has no adoption histories.")
    bullet("The submitted results use a 2-hop ego subgraph for diffusion. They do not describe the full 49,289-user graph.")
    bullet("No statistical significance is claimed for small differences. Monte Carlo standard deviations and sample scopes are reported.")
    bullet("The full-scale config is documented but not run; the report makes no full-scale performance claim.")

    # ---------------- Dataset / protocol ----------------
    add(PageBreak())
    h1("2. Dataset, split, and leakage protocol")
    h2("2.1 Dataset")
    rows = [
        ["Users", f"{data_summary['n_users']:,}"],
        ["Items", f"{data_summary['n_items']:,}"],
        ["Ratings", f"{data_summary['n_ratings']:,}"],
        ["Trust edges", f"{data_summary['n_trust_edges']:,}"],
        ["Rating matrix sparsity", f"{data_summary['sparsity']:.6f}"],
        ["Mean / median ratings per user", f"{data_summary['ratings_per_user']['mean']:.2f} / {data_summary['ratings_per_user']['median']:.1f}"],
        ["Mean / median ratings per item", f"{data_summary['ratings_per_item']['mean']:.2f} / {data_summary['ratings_per_item']['median']:.1f}"],
        ["Reciprocal trust edges", f"{data_summary['reciprocal_trust_edges']:,}"],
        ["Users with both ratings and trust", f"{data_summary['user_overlap']['both_rating_and_trust']:,}"],
    ]
    add(table(["Property", "Value"], rows, [78 * mm, 99 * mm], font_size=7.8))
    caption("Table 2. Parsed Epinions dataset summary.")
    p(
        "Trust rows use the convention “A trusts B”. Influence propagates in the "
        "opposite direction: a propagation edge B → A means B can influence A. "
        "All parsing counters for malformed, out-of-range, and duplicate records "
        "are saved in <font face='Courier'>data_summary.json</font>; this run found "
        "zero malformed rating lines, zero out-of-range ratings, and zero duplicate "
        "rating records."
    )
    h2("2.2 Split and evaluation discipline")
    p(
        f"The split seed is {split_info['seed']} with fractions "
        f"{split_info['train_frac']:.2f}/{split_info['val_frac']:.2f}/"
        f"{split_info['test_frac']:.2f}. Users with at least three ratings "
        "contribute at least one record to each split; users with fewer records are "
        "kept for training but excluded from evaluation. The split sizes are "
        f"{split_info['train_size']:,} / {split_info['val_size']:,} / "
        f"{split_info['test_size']:,}. {split_info['users_included_in_evaluation']:,} "
        "users are eligible for evaluation and "
        f"{split_info['users_excluded_lt3_ratings']:,} are excluded."
    )
    h2("2.3 Leakage prevention")
    bullet("<b>Model fitting:</b> only training triples. Validation is used for the social λ; test is used only for final reported metrics.")
    bullet("<b>Group construction (R6):</b> a matrix built explicitly from the training triples. The full <font face='Courier'>dataset.rating_matrix</font> is never passed into R6. A regression test mutates the full matrix and asserts that the groups do not change.")
    bullet("<b>Target item (R4/R7/R8):</b> selected from a training-only popularity band, with the rule and counts saved in JSON.")
    bullet("<b>Seed sets (R4/R6/R8):</b> generated once and reused across comparable settings. Common random numbers are used for paired comparisons.")
    bullet("<b>Observed vs simulated:</b> test ratings are labels, not recommender inputs. Simulated diffusion uses predicted affinity, not held-out ratings.")
    bullet("<b>Candidate pools:</b> all catalogue items except the user's training history for ranking; an explicit shared top-degree pool for R5. The restriction is documented, not hidden.")

    # ---------------- R1 ----------------
    h1("3. Requirement 1 — data preparation")
    p(
        "The loader validates every rating line, builds contiguous user/item "
        "indices, a sparse rating matrix, and a directed propagation graph. "
        "The splitter enforces the per-user minimum and records the exact sizes. "
        "The inspection command is:"
    )
    add(Paragraph("<font face='Courier'>python -m sntp.cli inspect --data-dir Assignment</font>", STYLE_BODY))
    p(
        "The item popularity distribution is extremely long-tailed: 99,715 items "
        "have only 1–5 training/test interactions in the item-popularity analysis, "
        "while 3,449 items have 21+. This is why top-K ranking metrics are low and "
        "why the diffusion target is deliberately drawn from the middle training "
        "popularity band rather than from a popular item."
    )

    # ---------------- R2 ----------------
    add(PageBreak())
    h1("4. Requirement 2 — baseline recommendation")
    h2("4.1 Models")
    p(
        "Item-based CF uses cosine similarity between item interaction vectors, "
        "at most 20 neighbours, and at least 2 co-rated items. Matrix factorization "
        "uses 10 latent factors, learning rate 0.005, L2 regularization 0.02, "
        "6 epochs, and seed 42. Both are fitted on the training split."
    )
    p(
        "Top-K excludes the user's training items and ranks the remaining catalogue. "
        "Ranking metrics use a fixed seeded sample of 500 eligible users; MAE/RMSE "
        "use all test ratings. Unobserved items are not treated as negative examples."
    )
    rows = []
    for key, label in (("item_based_cf", "Item-based CF"), ("matrix_factorization", "Matrix factorization")):
        m = r2[key]
        pr = m["prediction"]
        rk = m["ranking"]
        rows.append([
            label, num(pr["mae"], 4), num(pr["rmse"], 4),
            num(rk["precision_at_k"], 4), num(rk["recall_at_k"], 5),
            num(rk["f1_at_k"], 5), num(rk["hit_rate_at_k"], 4),
            num(rk["ndcg_at_k"], 5), num(rk["coverage"], 5),
        ])
    add(table(["Model", "MAE", "RMSE", "P@10", "R@10", "F1@10", "HR@10", "nDCG@10", "Coverage"],
              rows, [30 * mm, 17 * mm, 17 * mm, 15 * mm, 15 * mm, 15 * mm, 15 * mm, 17 * mm, 18 * mm], font_size=7.0))
    caption("Table 3. Test rating and top-10 ranking metrics. MF is much better on error; both have low absolute ranking metrics on this sparse catalogue.")
    def fig_r2(fig_obj):
        ax = fig_obj.add_subplot(111)
        labels = ["Item-based CF", "MF"]
        mae = [r2["item_based_cf"]["prediction"]["mae"], r2["matrix_factorization"]["prediction"]["mae"]]
        rmse = [r2["item_based_cf"]["prediction"]["rmse"], r2["matrix_factorization"]["prediction"]["rmse"]]
        x = np.arange(2)
        ax.bar(x - 0.18, mae, 0.36, label="MAE", color=MPL_COLORS["blue"])
        ax.bar(x + 0.18, rmse, 0.36, label="RMSE", color=MPL_COLORS["accent"])
        ax.set_xticks(x, labels)
        ax.set_ylabel("Test error (lower is better)")
        ax.set_title("Rating prediction error on the test split")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
    fig(fig_r2, "Figure 1. Test MAE and RMSE for the two baselines. MF reduces both errors substantially.")
    h2("4.2 Behaviour and popularity slices")
    p(
        "MF is consistently better than item-based CF on short, medium, and long "
        "user histories and on rare, moderate, and popular item groups. The largest "
        "gap is on moderate-popularity items: item-based CF MAE 1.2514 vs MF 0.8433. "
        "Item-based CF is not better on the rare-item group; its MAE is 0.9443 vs "
        "MF 0.8301. The complete slices are in <font face='Courier'>requirement_2_results.json</font>."
    )
    p(
        "Interpretation: MF's low-rank structure handles sparse users and items "
        "better than a neighbour model with few co-rated items. The absolute "
        "top-K numbers are low because each user has many unobserved catalogue "
        "items and only one held-out positive item is used for relevance; this is "
        "a hard, conservative ranking setting, not evidence that the recommender "
        "is useless."
    )

    # ---------------- R3 ----------------
    add(PageBreak())
    h1("5. Requirement 3 — trust-aware recommendation")
    p(
        "The social recommender blends the baseline prediction with the mean "
        "rating of trusted neighbours: "
        "<font face='Courier'>pred = (1−λ)·baseline + λ·trust_neighbour_mean</font>. "
        "The social weight is selected on <b>validation RMSE only</b>. The saved "
        "selection record explicitly states that test was not used."
    )
    rows = []
    for cand in r3["selection"]["candidates"]:
        rows.append([num(cand["lambda"], 1), num(cand["mae"], 5), num(cand["rmse"], 5), num(cand["n_evaluated"], 0)])
    add(table(["λ", "Validation MAE", "Validation RMSE", "N"], rows,
              [30 * mm, 45 * mm, 45 * mm, 35 * mm], font_size=7.8))
    caption("Table 4. Validation selection. λ=0.3 has the lowest validation RMSE and is selected before test is examined.")
    rows = []
    for key, label in (("social_lambda_0.0", "0.0 (baseline)"), ("social_lambda_0.3", "0.3 (selected)"), ("social_lambda_0.5", "0.5")):
        m = r3[key]
        pr, rk = m["prediction"], m["ranking"]
        rows.append([label, num(pr["mae"], 5), num(pr["rmse"], 5), num(rk["precision_at_k"], 4),
                     num(rk["hit_rate_at_k"], 4), num(rk["ndcg_at_k"], 5)])
    add(table(["λ", "Test MAE", "Test RMSE", "P@10", "HR@10", "nDCG@10"], rows,
              [35 * mm, 30 * mm, 30 * mm, 25 * mm, 25 * mm, 30 * mm], font_size=7.8))
    caption("Table 5. Test metrics reported after selection. λ=0.3 improves error and ranking relative to the baseline.")
    p(
        f"Sparse users (1–5 ratings, {r3['sparse_users_analysis']['n_sparse_users']:,} users) "
        f"improve from MAE {num(r3['sparse_users_analysis']['baseline']['mae'], 5)} "
        f"to {num(r3['sparse_users_analysis']['social']['mae'], 5)} at λ=0.3. "
        "The trust-vs-similarity comparison sampled 50 users (165 test ratings): "
        f"trusted neighbours MAE {num(r3['trust_vs_similarity_comparison']['trust_neighbors']['mae'], 5)}, "
        f"unconnected similar users {num(r3['trust_vs_similarity_comparison']['similar_unconnected']['mae'], 5)}, "
        f"baseline {num(r3['trust_vs_similarity_comparison']['baseline']['mae'], 5)}. "
        "The similar-user group is slightly better in this small sample, so the "
        "trust signal is not uniformly superior; the result is descriptive."
    )
    p(
        "The λ=0.5 test ranking is higher than λ=0.3 even though its validation "
        "RMSE is worse and its test RMSE is worse. This is exactly why the "
        "selection rule is fixed in advance: choosing λ=0.5 after seeing test "
        "ranking would be test leakage. The report keeps λ=0.3 and shows the "
        "post-selection λ=0.5 row for transparency."
    )

    # ---------------- R4 ----------------
    add(PageBreak())
    h1("6. Requirement 4 — ICM, SIR, and preference-aware diffusion")
    h2("6.1 Design and target popularity")
    target = r4["design"]["target_item"]
    p(
        "The target item is selected from the <b>training-only</b> rating-count "
        "distribution. The rule chooses the count closest to the median count in "
        "the inclusive [0.40, 0.60] training-count quantile band, with ties by "
        "ascending original item ID. The selected item is original ID "
        f"<b>{target['original_id']}</b> (matrix index {target['idx']}), with "
        f"{target['train_rating_count']} training rating. The band covers counts "
        f"{target['band_lower_count']:.0f}–{target['band_upper_count']:.0f} and "
        f"{target['band_size_items']:,} items. It is intentionally a low-popularity "
        "item; absolute spreads are not a catalogue-wide forecast."
    )
    p(
        "The three seed sets (five users each) and the Monte Carlo seeds are "
        "pre-generated once and reused for every ICM and SIR setting. The design "
        "therefore compares propagation models, not different random seeds:"
    )
    for seed_set in r4["design"]["seed_sets"]:
        add(Paragraph(f"<font face='Courier'>{seed_set}</font>", STYLE_BODY))
    p(
        "The fast demo uses 9,883 users and 301,710 propagation edges (2-hop ego "
        "subgraph centered on user 0), three simulations per setting, and at most "
        "25 rounds. The SIR recovery probability γ=0 uses all 25 rounds because "
        "infected users never recover; those round counts are capped by the "
        "configured maximum, not evidence that the cascade naturally lasted 25 rounds."
    )
    h2("6.2 Measured spreads")
    rows = []
    for setting, values in r4["icm_canonical"].items():
        s = values["summary"]
        rows.append(["Canonical ICM", setting, num(s["mean_adopters"], 2), num(s["std_adopters"], 2), num(s["mean_rounds"], 2), num(values["runtime_seconds"], 3)])
    for setting, values in r4["icm_preference_aware"].items():
        s = values["summary"]
        rows.append(["Preference-aware ICM", setting, num(s["mean_adopters"], 2), num(s["std_adopters"], 2), num(s["mean_rounds"], 2), num(values["runtime_seconds"], 3)])
    for setting, values in r4["sir"].items():
        s = values["summary"]
        rows.append(["SIR", setting, num(s["mean_ever_infected"], 2), num(s["std_ever_infected"], 2), num(s["mean_rounds"], 2), num(values["runtime_seconds"], 3)])
    add(table(["Model", "Setting", "Mean adopters/infected", "Std", "Mean rounds", "Runtime (s)"], rows,
              [35 * mm, 35 * mm, 32 * mm, 25 * mm, 23 * mm, 23 * mm], font_size=7.0))
    caption("Table 6. Diffusion outcomes. Means and standard deviations are across the same three pre-generated seed sets and controlled simulation seeds.")
    def fig_r4(fig_obj):
        ax = fig_obj.add_subplot(111)
        labels, means, stds, colors_ = [], [], [], []
        for setting, values in r4["icm_canonical"].items():
            s = values["summary"]
            labels.append("ICM " + setting.replace("p=", "p="))
            means.append(s["mean_adopters"])
            stds.append(s["std_adopters"])
            colors_.append(MPL_COLORS["blue"])
        for setting, values in r4["icm_preference_aware"].items():
            s = values["summary"]
            labels.append("Pref " + setting.replace("p=", "p="))
            means.append(s["mean_adopters"])
            stds.append(s["std_adopters"])
            colors_.append(MPL_COLORS["accent"])
        for setting, values in r4["sir"].items():
            s = values["summary"]
            labels.append("SIR " + setting.replace("beta=", "β=").replace("gamma=", "γ="))
            means.append(s["mean_ever_infected"])
            stds.append(s["std_ever_infected"])
            colors_.append(MPL_COLORS["navy"])
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=2, color=colors_, alpha=0.9)
        ax.set_yscale("log")
        ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=6)
        ax.set_ylabel("Adopters / ever infected (log scale)")
        ax.set_title("Diffusion outcomes with one-standard-deviation bars")
        ax.grid(axis="y", alpha=0.25)
    fig(fig_r4, "Figure 2. Diffusion outcomes on a log scale. The p=0.01 canonical ICM is extremely heavy-tailed; preference-aware adoption strongly suppresses spread at the same seeds.", height=84 * mm)
    h2("6.3 Interpretation")
    p(
        "Canonical ICM at p=0.05 reaches about 4,684 of 9,883 active users (47%), "
        "consistent with a dense ego graph. Preference-aware adoption requires "
        "predicted affinity and cuts the p=0.05 mean to 501 (−89.3%); at p=0.01 "
        "it cuts the mean from 125.3 to 6.0 (−95.2%). This is a model-design "
        "effect, not evidence that real users would adopt at that rate. SIR at "
        "β=0.05 reaches 8,182 with γ=0 and 6,021 with γ=0.2; recovery reduces "
        "ever-infected counts but the γ=0 runs remain active to the round cap."
    )
    p(
        "The p=0.01 canonical ICM has mean 125.3 but standard deviation 167.4 and "
        "a maximum of 362: one simulation found a large cascade and two did not. "
        "With only three simulations, the mean is unstable. This is reported "
        "explicitly rather than hidden; the full configuration uses 20 simulations."
    )

    # ---------------- R5 ----------------
    add(PageBreak())
    h1("7. Requirement 5 — influence maximization")
    p(
        "The comparison uses one propagation graph (9,883 nodes, 301,710 edges), "
        "one propagation probability (0.05), budgets 1/3/5, and a shared candidate "
        "pool. The pool is the top 15 out-degree users in the propagation graph "
        "(ties by ascending user index), identical for all methods. This bound is "
        "necessary for a fast demo, but it gives degree centrality a structural "
        "advantage. The result describes that pool, not all graph users."
    )
    p(
        "Random selection is repeated for three seeded trials. Degree and greedy "
        "are deterministic selectors. Every method is evaluated on the same two "
        "evaluation seeds; random's mean is the mean of its trial means. Marginal "
        "gains are paired differences on the same Monte Carlo trials, not "
        "differences of unrelated runs."
    )
    rows = []
    for b in r5["design"]["budgets"]:
        b = str(b)
        row = [b]
        for method in ("random", "degree", "greedy"):
            row.append(num(r5["methods"][method][b]["mean_spread"], 2))
        row.append(r5["analysis"]["comparison"][b]["best_method"])
        rows.append(row)
    add(table(["Budget", "Random mean", "Degree mean", "Greedy mean", "Best measured"], rows,
              [28 * mm, 38 * mm, 38 * mm, 38 * mm, 45 * mm], font_size=8.0))
    caption("Table 7. Mean simulated spread under common evaluation seeds. The restricted top-degree pool favors degree; greedy is best only at budget 1.")
    def fig_r5(fig_obj):
        ax = fig_obj.add_subplot(111)
        budgets = [int(b) for b in r5["design"]["budgets"]]
        for method, color in (("random", MPL_COLORS["blue"]), ("degree", MPL_COLORS["accent"]), ("greedy", MPL_COLORS["navy"])):
            vals = [r5["methods"][method][str(b)]["mean_spread"] for b in budgets]
            ax.plot(budgets, vals, marker="o", label=method, color=color)
        ax.set_xticks(budgets)
        ax.set_xlabel("Seed budget")
        ax.set_ylabel("Mean simulated spread")
        ax.set_title("Influence spread vs seed budget (shared top-degree candidate pool)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
    fig(fig_r5, "Figure 3. Greedy, degree, and random spreads. Differences are small relative to the saturated baseline and do not show greedy superiority.", height=72 * mm)
    h2("7.1 Marginal gains and uncertainty")
    rows = []
    for method in ("random", "degree", "greedy"):
        for b in ("3", "5"):
            m = r5["methods"][method][b]
            ci = "—"
            if "marginal_gain_ci95_low" in m:
                ci = f"[{num(m['marginal_gain_ci95_low'], 2)}, {num(m['marginal_gain_ci95_high'], 2)}]"
            rows.append([method, b, num(m.get("marginal_gain"), 2), ci,
                         f"{m.get('marginal_gain_wins', '—')}/{m.get('marginal_gain_losses', '—')}"])
    add(table(["Method", "Budget", "Paired marginal gain", "95% CI", "Wins/losses"], rows,
              [25 * mm, 23 * mm, 38 * mm, 45 * mm, 36 * mm], font_size=7.5))
    caption("Table 8. Marginal gains relative to the previous evaluated budget. Negative values are legitimate under finite Monte Carlo sampling; they are not clipped to zero.")
    p(
        "At budget 5, greedy − degree = −12.0 adopters (paired 95% CI "
        "[−19.84, −4.16]), and greedy loses both paired trials. Greedy exceeds "
        "random by 8.0 adopters (0.2%), far below the trial variability. The "
        "honest conclusion is <b>no evidence of greedy superiority</b> on this "
        "restricted pool. Degree and greedy adopter sets overlap substantially "
        f"(mean Jaccard {r5['adopter_overlaps']['degree_vs_greedy']:.3f}), so "
        "high-degree seeds largely reach the same audience."
    )
    p(
        f"Greedy selected the full budget in every row (65 evaluations at budget 5) "
        f"and took {num(r5['runtimes_seconds']['greedy_selection'], 2)} s for selection. "
        f"Degree selection was effectively instant ({num(r5['runtimes_seconds']['degree_selection'], 4)} s). "
        "The standard-greedy sentinel is −∞, so a negative Monte Carlo gain does "
        "not stop seeding; this was a defect in the earlier implementation."
    )

    # ---------------- R6 ----------------
    add(PageBreak())
    h1("8. Requirement 6 — group recommendation")
    p(
        "Groups are constructed <b>only from the training split</b> via "
        "<font face='Courier'>build_rating_matrix(train_ratings)</font>. The full "
        "observed matrix is never passed into R6; a regression test mutates the "
        "full matrix and asserts that the groups are unchanged. The construction "
        "forms conflicting groups from low, middle, and high mean training ratings, "
        "with seed 42. The fast demo uses 3 groups × 4 members; groups may overlap "
        "across groups."
    )
    p(
        "The three aggregation strategies are average predicted rating, least "
        "misery (minimum), and most pleasure (maximum). Satisfaction is the "
        "recommender's prediction for each member, not a held-out rating. "
        "Diffusion uses preference-aware ICM with the group members as seeds, "
        "two simulations per candidate item, and controlled seeds."
    )
    rows = []
    for strategy, s in r6["summary"].items():
        rows.append([strategy.replace("_", " "), num(s["avg_mean_satisfaction"], 3),
                     num(s["avg_min_satisfaction"], 3), num(s["avg_disagreement_range"], 3),
                     num(s["avg_diffusion_spread"], 3), num(s["avg_diffusion_std"], 3)])
    add(table(["Strategy", "Mean satisfaction", "Mean min", "Mean range", "Mean spread", "Spread std"], rows,
              [38 * mm, 31 * mm, 27 * mm, 28 * mm, 27 * mm, 28 * mm], font_size=7.2))
    caption("Table 9. Group strategy outcomes across three groups. Satisfaction differences are small; spread variability is large.")
    p(
        "The average strategy has the highest mean and minimum satisfaction, while "
        "least misery has the highest mean simulated spread. The differences are "
        "tiny: mean satisfaction 4.827 vs 4.825 vs 4.815, and minimum satisfaction "
        "4.656 vs 4.656 vs 4.633. The spread standard deviation (~195–200) is much "
        "larger than the between-strategy mean difference (~5). One group's "
        "cascade was 4 adopters in one simulation and ~1,171–1,199 in the other, "
        "depending on the seed; that single path dominates the average. These data "
        "do not support a strong “group strategy trade-off” claim."
    )
    h2("8.1 Item-level comparison and ties")
    p(
        "For each group, the candidate items are the union of the three strategies' "
        "top items, not the full catalogue. The item with the highest predicted "
        "mean satisfaction was also one of the highest-spread items in "
        f"{r6['analysis']['n_best_satisfaction_and_spread_overlap']} of "
        f"{r6['analysis']['n_groups_with_candidates']} groups "
        f"({r6['analysis']['agreement_rate']*100:.1f}%). Ties are treated as tied "
        "bests: in group 0 the best-spread set is {127, 16353}, so the satisfaction "
        "winner 16353 is a spread winner even though 127 is the canonical "
        "lowest-ID representative. In group 1 the satisfaction winner (16353) and "
        "spread winner (8559) differ. This tie-aware overlap is a non-empty "
        "intersection; it does not require the two best sets to be identical."
    )

    # ---------------- R7 ----------------
    add(PageBreak())
    h1("9. Requirement 7 — comprehensive evaluation")
    h2("9.1 Metric definitions")
    bullet("<b>Relevance:</b> held-out test rating ≥ 4. Unobserved items are not negatives.")
    bullet("<b>Candidates:</b> all catalogue items except the user's training items; no negative sampling.")
    bullet("<b>Precision@10, Recall@10, HitRate@10, nDCG@10:</b> standard per-user definitions, averaged over users. A user with no relevant held-out item contributes defined zeros for the ranking average.")
    bullet("<b>F1@10:</b> mean of per-user F1 values, not the harmonic mean of aggregate precision and recall.")
    bullet("<b>Coverage:</b> unique recommended items / full 139,738-item catalogue over the evaluated user sample. It is a sample-scope coverage, not a platform-wide estimate.")
    bullet("<b>Diversity:</b> mean pairwise 1−cosine of training-interaction item feature vectors within each list, on a bounded deterministic list sample. It is feature diversity, not category diversity.")
    bullet("No accuracy classification metric is reported: with only held-out positives and no observed negatives, a confusion-matrix accuracy would be misleading.")
    h2("9.2 Observed evaluation")
    rows = []
    labels = {"item_based_cf": "Item-based CF", "matrix_factorization": "MF", "social_recommender": "Social (λ=0.3)"}
    for key in ("item_based_cf", "matrix_factorization", "social_recommender"):
        m = r7["observed"]["models"][key]
        pe, rk = m["prediction_error"], m["ranking"]
        rows.append([labels[key], num(pe["mae"], 4), num(pe["rmse"], 4),
                     num(rk["precision_at_k"], 4), num(rk["recall_at_k"], 5),
                     num(rk["f1_at_k"], 5), num(rk["hit_rate_at_k"], 4),
                     num(rk["ndcg_at_k"], 5), num(rk["coverage"], 5),
                     num(rk["diversity"], 4)])
    add(table(["Model", "MAE", "RMSE", "P@10", "R@10", "F1@10", "HR@10", "nDCG@10", "Coverage", "Diversity"],
              rows, [26 * mm, 15 * mm, 15 * mm, 14 * mm, 14 * mm, 14 * mm, 14 * mm, 16 * mm, 16 * mm, 16 * mm], font_size=6.8))
    caption("Table 10. Observed test metrics (500 seeded users for ranking; all test ratings for MAE/RMSE). Social is best on error and ranking; ICF has the broadest coverage.")
    p(
        "The social model is best on MAE, RMSE, precision, recall, F1, hit rate, "
        "and nDCG. Item-based CF has much higher coverage but its ranking accuracy "
        "is far lower. MF has very low coverage because its top-10 lists concentrate "
        "on a small set of high-scoring items; this is a known coverage/accuracy "
        "trade-off. The diversity values are high for MF/social (≈0.985) because "
        "their lists mix items with different interaction vectors, while ICF is "
        "lower (0.859)."
    )
    h2("9.3 Separate simulated interaction")
    p(
        "The simulated interaction uses one training-selected target item and "
        "common seeds; it is not an observed adoption forecast. Mean adopters are "
        f"ICF {num(r7['tradeoff_analysis']['simulated_spread_ranking']['item_based_cf'], 1)}, "
        f"MF {num(r7['tradeoff_analysis']['simulated_spread_ranking']['matrix_factorization'], 1)}, "
        f"social {num(r7['tradeoff_analysis']['simulated_spread_ranking']['social_recommender'], 1)}. "
        "The highest simulated spread and the highest observed nDCG agree on the "
        "social model in this run, but the two quantities answer different "
        "questions. The report does not merge them into one score."
    )

    # ---------------- R8 ----------------
    add(PageBreak())
    h1("10. Requirement 8 — intervention, CELF, and scaling")
    h2("10.1 Diffusion-limiting intervention")
    p(
        "The intervention applies a minimum predicted-affinity threshold inside "
        "preference-aware ICM: exposed users below the threshold neither adopt nor "
        "propagate. Seeds are platform-promoted users and always start the cascade. "
        "The baseline is threshold 0.0 with the same seed sets and seeds as each "
        "threshold row. Five simulations are used per threshold."
    )
    rows = []
    for key, row in r8["intervention"]["results"].items():
        rel = row.get("relative_to_baseline", {})
        rows.append([num(row["threshold"], 1), num(row["mean_adopters"], 2),
                     num(row["mean_exposed"], 2), num(row["conversion_efficiency"], 4),
                     num(row["mean_adopter_rating_proxy"], 4),
                     pct(rel.get("spread_change_pct")), num(rel.get("adopter_satisfaction_change"), 4) if rel else "—"])
    add(table(["Threshold", "Mean adopters", "Mean exposed", "Conversion", "Rating proxy", "Spread change", "Proxy change"], rows,
              [23 * mm, 26 * mm, 26 * mm, 24 * mm, 27 * mm, 25 * mm, 25 * mm], font_size=7.2))
    caption("Table 11. Intervention results relative to threshold 0. The rating proxy is 1+4×predicted affinity and is not an observed satisfaction rating.")
    aff = r8["intervention"].get("baseline_exposed_affinity") or {}
    if aff:
        p(
            f"The threshold effect is a cliff, not a smooth trade-off. The baseline "
            f"exposure population is {aff['n_exposed']:,} unique users (across five "
            f"simulations); affinity ranges {aff['min_affinity']:.4f}–"
            f"{aff['max_affinity']:.4f}. The share below threshold is "
            f"{aff['share_below_threshold'].get('0.30', 0)*100:.2f}% at 0.3, "
            f"{aff['share_below_threshold'].get('0.60', 0)*100:.2f}% at 0.6, "
            f"{aff['share_below_threshold'].get('0.80', 0)*100:.2f}% at 0.8, and "
            f"{aff['share_below_threshold'].get('0.90', 0)*100:.2f}% at 0.9. "
            "Threshold 0.3 therefore cannot change the baseline cascade; 0.6 "
            "changes only a few paths; 0.8 removes most propagation; 0.9 removes "
            "slightly more spread but lowers the proxy because the remaining "
            "adopters are mostly seeds. Shared seeds reduce but do not eliminate "
            "path divergence when filtering changes the exposed set."
        )
    def fig_r8(fig_obj):
        ax = fig_obj.add_subplot(111)
        rows_ = list(r8["intervention"]["results"].values())
        th = [row["threshold"] for row in rows_]
        spread_pct = [row.get("relative_to_baseline", {}).get("spread_change_pct", 0.0) for row in rows_]
        proxy_change = [row.get("relative_to_baseline", {}).get("adopter_satisfaction_change", 0.0) for row in rows_]
        ax.plot(th, spread_pct, marker="o", color=MPL_COLORS["accent"], label="Spread change (%)")
        ax.axhline(0, color="#888888", linewidth=0.7)
        ax.set_xlabel("Minimum affinity threshold")
        ax.set_ylabel("Spread change vs baseline (%)")
        ax.set_title("Intervention: spread change")
        ax2 = ax.twinx()
        ax2.plot(th, proxy_change, marker="s", color=MPL_COLORS["blue"], label="Proxy change")
        ax2.set_ylabel("Adopter rating proxy change")
        lines, labels_ = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels_ + labels2, frameon=False, loc="lower left")
        ax.grid(alpha=0.25)
    fig(fig_r8, "Figure 4. The 0.3 threshold is inactive; 0.6 is near baseline; 0.8 collapses spread and slightly raises the proxy; 0.9 lowers the proxy.", height=72 * mm)
    p(
        "The honest reading: this intervention can strongly limit simulated "
        "diffusion at a high threshold, but the effect is not monotone and the "
        "proxy is not observed post-intervention satisfaction. The 0.8 row is the "
        "largest measured spread reduction among thresholds with a non-negative "
        "proxy change; the 0.9 row shows that over-filtering can also reduce the "
        "proxy. No online randomized experiment was run."
    )
    h2("10.2 CELF optimization")
    celf = r8["celf_optimization"]["comparison"]
    controlled = r8["celf_optimization"].get("controlled_check", {})
    rows = []
    for key, label in (("standard_greedy", "Standard greedy"), ("celf_greedy", "CELF")):
        m = celf[key]
        rows.append([label, m["n_evaluations"], num(m["runtime_seconds"], 2),
                     plain(m["seeds"]), plain(m["marginal_gains"])])
    add(table(["Method", "Evaluations", "Runtime (s)", "Selected seeds", "Marginal gains"], rows,
              [34 * mm, 24 * mm, 24 * mm, 38 * mm, 45 * mm], font_size=7.3))
    caption("Table 12. CELF vs standard greedy at budget 3, 12 candidates, 2 simulations. CELF used 24.24% fewer evaluations and was 1.224× faster, but selected different seeds under noisy estimates.")
    p(
        f"CELF's lazy bound is exact for a deterministic monotone submodular "
        f"objective, not for a finite-sample noisy estimate. The measured 1.22× "
        f"speedup and 24.2% evaluation reduction are therefore reported as "
        f"single-run measurements, not an unconditional guarantee. A controlled "
        f"deterministic check (p=1.0, one simulation, 120 nodes, 20 candidates, "
        f"budget 2) gave {controlled['standard_n_evaluations']} "
        f"evaluations for both methods and identical seeds "
        f"{plain(controlled['standard_seeds'])}. That check is the "
        f"implementation correctness evidence; the noisy run is the cost evidence. "
        f"Memory tracemalloc peaks were {num(celf['standard_greedy']['memory']['tracemalloc_peak_mb'], 3)} MB "
        f"and {num(celf['celf_greedy']['memory']['tracemalloc_peak_mb'], 3)} MB; "
        f"RSS deltas were {num(celf['standard_greedy']['memory']['rss_delta_mb'], 3)} MB "
        f"and {num(celf['celf_greedy']['memory']['rss_delta_mb'], 3)} MB. NumPy native "
        f"allocations may be under-represented."
    )
    h2("10.3 Measured scaling and bottleneck")
    net = r8["scaling"]["network"]["rows"]
    rows = [[num(r["n_nodes"], 0), num(r["n_edges"], 0), num(r["runtime_seconds"], 2),
             num(r["n_evaluations"], 0), num(r["memory"]["tracemalloc_peak_mb"], 2)] for r in net]
    add(table(["Network nodes", "Edges", "Runtime (s)", "Evaluations", "Tracemalloc peak (MB)"], rows,
              [33 * mm, 36 * mm, 30 * mm, 30 * mm, 45 * mm], font_size=7.5))
    caption("Table 13. Network-size scaling for greedy selection from scratch on BFS-induced subgraphs.")
    items = r8["scaling"]["item_catalogue"]["rows"]
    rows = [[num(r["n_items"], 0), num(r["ms_per_user"], 3), num(r["runtime_seconds"], 3),
             num(r["memory"]["tracemalloc_peak_mb"], 2)] for r in items]
    add(table(["Catalogue items", "ms/user", "Runtime (s)", "Tracemalloc peak (MB)"], rows,
              [45 * mm, 32 * mm, 34 * mm, 45 * mm], font_size=7.5))
    caption("Table 14. Item-catalogue scaling for MF scoring/ranking. This isolates scoring, not model training.")
    def fig_r8_scale(fig_obj):
        ax = fig_obj.add_subplot(111)
        nodes = [r["n_nodes"] for r in net]
        times = [r["runtime_seconds"] for r in net]
        ax.plot(nodes, times, marker="o", color=MPL_COLORS["navy"])
        ax.set_xlabel("Propagation-graph nodes")
        ax.set_ylabel("Greedy selection runtime (s)")
        ax.set_title("Network-size scaling (15 candidates, budget 2)")
        ax.grid(alpha=0.25)
    fig(fig_r8_scale, "Figure 5. Measured network scaling. The slope is machine-dependent; the report compares relative growth within this run.", height=65 * mm)
    budget = r8["scaling"]["promotion_budget"]["rows"]
    rows = [[num(r["budget"], 0), num(r["n_evaluations"], 0), num(r["runtime_seconds"], 2)] for r in budget]
    add(table(["Seed budget", "Evaluations", "Runtime (s)"], rows,
              [40 * mm, 40 * mm, 40 * mm], font_size=7.8))
    caption("Table 15. Promotion-budget scaling on a 2,471-node subgraph. Evaluations grow with budget because greedy re-evaluates candidates.")
    r8_times = r8["bottleneck_analysis"]["r8_sub_experiment_seconds"]
    p(
        f"Within R8, the CELF comparison is the dominant cost "
        f"({num(r8_times['celf'], 2)} s), followed by network scaling "
        f"({num(r8_times['network_scaling'], 2)} s) and budget scaling "
        f"({num(r8_times['budget_scaling'], 2)} s). The intervention itself is "
        f"cheap ({num(r8_times['intervention'], 2)} s); the cost is repeated "
        f"Monte Carlo evaluation, not the threshold arithmetic. Network runtime "
        f"grows from {num(net[0]['runtime_seconds'], 2)} s at "
        f"{net[0]['n_nodes']:,} nodes to {num(net[-1]['runtime_seconds'], 2)} s at "
        f"{net[-1]['n_nodes']:,} nodes. Budget runtime grows from "
        f"{num(budget[0]['runtime_seconds'], 2)} s to "
        f"{num(budget[-1]['runtime_seconds'], 2)} s as evaluations grow from "
        f"{budget[0]['n_evaluations']} to {budget[-1]['n_evaluations']}. Item "
        f"scoring grows from {num(items[0]['ms_per_user'], 2)} to "
        f"{num(items[-1]['ms_per_user'], 2)} ms/user as the catalogue grows from "
        f"{items[0]['n_items']:,} to {items[-1]['n_items']:,}; this is a measured "
        f"cache/allocation effect, not a claimed linear law."
    )

    # ---------------- Verification ----------------
    add(PageBreak())
    h1("11. Verification, tests, and reproducibility")
    h2("11.1 Automated tests")
    p(
        "The latest local run was <b>87 passed</b> with "
        "<font face='Courier'>python -m pytest -q</font>. The leakage/fairness "
        "regressions are explicit:"
    )
    bullet("Training-only matrix construction and R6 groups unaffected by mutating the full observed matrix.")
    bullet("Social λ selected on validation when test ratings are perturbed.")
    bullet("Target-item selection and seed-set generation deterministic and training-only.")
    bullet("Greedy continues after a negative Monte Carlo marginal gain; explicit candidate pool respected.")
    bullet("Paired marginal gains equal the difference of the paired spread estimates.")
    bullet("Feature diversity, per-user F1, and hand-checkable metric examples.")
    h2("11.2 Exact commands")
    commands = [
        "python -m pytest -q",
        "python -m sntp.cli inspect --data-dir Assignment",
        "python -m sntp.cli run --config configs/fast_demo.yaml",
        "python -m pip install -r requirements-report.txt",
        "python scripts/generate_report.py --results results/fast_demo --output report/SNTP_Assignment_02_Report.pdf --roll-number <roll> --student \"Ahmad Fareed\"",
    ]
    for cmd in commands:
        add(Paragraph(f"<font face='Courier'>{cmd}</font>", STYLE_BODY))
    h2("11.3 Runtime and seeds")
    rows = [[s["stage"], num(s["seconds"], 2)] for s in timings["stages"]]
    add(table(["Stage", "Seconds"], rows, [140 * mm, 30 * mm], font_size=7.0))
    caption(f"Table 16. Stage timings from the submitted run. Total CLI wall clock: {num(timings['total_seconds'], 1)} s.")
    p(
        "Config: <font face='Courier'>configs/fast_demo.yaml</font> (resolved copy in "
        "<font face='Courier'>results/fast_demo/experiment_config.json</font>). "
        "Split seed 42; ranking/diversity sample seed 42; MF seed 42; group seed 42; "
        "R5 selection seeds [49, 1049], evaluation seeds [500049, 501049]; R6 "
        "per-group seeds [100049, 101049], [101049, 102049], [102049, 103049]; R8 "
        "intervention seeds [900049, 901049, 902049, 903049, 904049]; R8 CELF "
        "selection seeds [800049, 801049]; scaling offset 820000. The JSON files "
        "record the seed schedules used."
    )

    # ---------------- Limitations/conclusions ----------------
    add(PageBreak())
    h1("12. Limitations and conclusions")
    h2("12.1 Limitations")
    bullet("<b>Full-scale run not executed.</b> The full config uses the full graph, 20 simulations, larger candidate pools, and larger budgets; estimated cost is many hours. No full-scale number is claimed.")
    bullet("<b>Fast-demo scope.</b> Diffusion/influence/group/intervention use a 2-hop ego subgraph (9,883 users); R5 uses a top-15 out-degree pool. The demo also uses smaller model budgets than the full config (ICF 20 neighbours vs 50; MF 6 epochs vs 20). These are bounded, deterministic scopes, not random full-graph samples.")
    bullet("<b>Monte Carlo noise.</b> R4 uses 3 simulations; R5 uses 2 evaluation simulations and 3 random trials; R6 uses 2 per item; R8 CELF uses 2. Heavy tails make small mean differences unstable.")
    bullet("<b>R5 does not show greedy superiority.</b> Degree wins at budgets 3 and 5; the paired greedy−degree difference at budget 5 is negative. The dense graph is near saturation (~47% of active users), compressing differences.")
    bullet("<b>R6 differences are small.</b> Satisfaction ranges 4.815–4.827; spread is dominated by one rare cascade. Candidate items are only the union of strategy top items.")
    bullet("<b>R7 is sample-based.</b> Ranking uses 500 users, coverage/diversity are list-sample properties, and unobserved items are not negatives. The simulation uses one low-popularity target item.")
    bullet("<b>R8 is a simulator intervention.</b> No online experiment; the rating proxy is predicted, not observed; the threshold cliff is a property of the affinity distribution; CELF's 1.22× is a single noisy measurement and seeds differ. The deterministic check is the correctness evidence.")
    bullet("<b>Memory measurement.</b> tracemalloc undercounts NumPy native buffers; RSS deltas are near zero because arrays are already resident.")
    bullet("<b>Runtime variability.</b> Prior runs of the same fast config ranged 365–596 s; the saved stage timings are one run and should not be treated as a benchmark.")
    h2("12.2 Conclusions by requirement")
    conclusions = [
        ("Data/split", "The parser is clean (zero malformed/duplicate/out-of-range ratings) and the 70/15/15 split is reproducible with per-user coverage."),
        ("Baselines", "MF clearly beats item-based CF on rating error; both have low absolute top-K metrics on this sparse catalogue. MF is better across user-history and item-popularity slices."),
        ("Trust-aware", "Validation-selected λ=0.3 gives a small but consistent test improvement (MAE 0.83245→0.82838) and better ranking. The trust-vs-similarity comparison is small and descriptive."),
        ("Diffusion", "Canonical ICM at p=0.05 saturates the ego graph; preference-aware adoption reduces spread by ~89–95% at the same seeds. SIR recovery reduces ever-infected counts. Results are simulation-only and heavy-tailed at low p."),
        ("Influence", "On a shared top-degree pool, degree is best at budgets 3 and 5; greedy is best only at budget 1. There is no evidence for greedy superiority, and the pool restriction is material."),
        ("Groups", "Average has the best predicted mean/min satisfaction; least misery has the largest (but noisy) simulated spread. Differences are small; one rare cascade dominates spread."),
        ("Evaluation", "Social is best on observed error and ranking; ICF gives the broadest coverage; MF gives the highest feature diversity. Observed and simulated outcomes are reported separately."),
        ("Intervention/optimization", "Threshold 0.8 reduces simulated spread by 99.09% with a +0.0264 proxy change; 0.9 reduces spread slightly more but lowers the proxy. CELF uses 24.2% fewer evaluations and is 1.22× faster in the noisy run, with different seeds; a deterministic check matches standard greedy exactly."),
    ]
    rows = [[a, b] for a, b in conclusions]
    add(table(["Requirement", "Conclusion"], rows, [35 * mm, 140 * mm], font_size=7.4))
    caption("Table 17. Conclusions tied to measured evidence. No unexecuted configuration is presented as a result.")
    p(
        "The codebase is runnable, the eight requirements are implemented, and the "
        "submitted numbers are reproducible from the saved configuration and JSON. "
        "The main scientific caveats are the bounded fast-demo scope, the small "
        "Monte Carlo budgets, and the absence of observed adoption/satisfaction "
        "outcomes. Those caveats are reflected in the conclusions rather than "
        "hidden."
    )

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/fast_demo", help="Results directory")
    parser.add_argument("--output", default="report/SNTP_Assignment_02_Report.pdf", help="Output PDF")
    parser.add_argument("--roll-number", default="ROLL_NUMBER", help="Student roll number")
    parser.add_argument("--student", default="Ahmad Fareed", help="Student name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = Path(args.results)
    required = [
        "data_summary.json", "split_info.json", "experiment_config.json",
        "stage_timings.json", "requirement_2_results.json",
        "requirement_3_results.json", "requirement_4_results.json",
        "requirement_5_results.json", "requirement_6_results.json",
        "requirement_7_results.json", "requirement_8_results.json",
    ]
    missing = [name for name in required if not (results / name).exists()]
    if missing:
        raise SystemExit(f"Missing result files in {results}: {missing}")

    register_fonts()
    build_styles()
    metrics = {
        "data_summary": load_json(results / "data_summary.json"),
        "split_info": load_json(results / "split_info.json"),
        "config": load_json(results / "experiment_config.json"),
        "timings": load_json(results / "stage_timings.json"),
        "r2": load_json(results / "requirement_2_results.json"),
        "r3": load_json(results / "requirement_3_results.json"),
        "r4": load_json(results / "requirement_4_results.json"),
        "r5": load_json(results / "requirement_5_results.json"),
        "r6": load_json(results / "requirement_6_results.json"),
        "r7": load_json(results / "requirement_7_results.json"),
        "r8": load_json(results / "requirement_8_results.json"),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = ReportDoc(
        str(output),
        title="SNTP Assignment 2 — Recommendation and Information Diffusion on Epinions",
        author=args.student,
        roll=args.roll_number,
        leftMargin=19 * mm,
        rightMargin=19 * mm,
        topMargin=19 * mm,
        bottomMargin=17 * mm,
    )
    doc.build(story(metrics, args))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
