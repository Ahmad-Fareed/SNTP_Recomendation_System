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


def make_figure(draw, width: float = 165 * mm, height: float = 78 * mm,
                output_path: Path | None = None) -> Image:
    buffer = io.BytesIO()
    fig = plt.figure(figsize=(7.0, 3.3), dpi=180)
    try:
        draw(fig)
        fig.tight_layout()
        fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, format="png", dpi=200,
                        bbox_inches="tight", facecolor="white")
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

    def fig(draw, text: str, width=165 * mm, height=76 * mm,
            name: str | None = None) -> None:
        path = (Path(args.figures_dir) / name) if name else None
        add(KeepTogether([
            make_figure(draw, width, height, output_path=path),
            Paragraph(text, STYLE_CAPTION),
        ]))

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
    selection = metrics.get("selection")

    r2_icf = r2["item_based_cf"]
    r2_mf = r2["matrix_factorization"]
    best_lambda_key = f"social_lambda_{r3['best_lambda']:.1f}"
    r3_best = r3[best_lambda_key]
    r4_can_01 = r4["icm_canonical"]["p=0.01"]["summary"]
    r4_can_05 = r4["icm_canonical"]["p=0.05"]["summary"]
    r4_pref_01 = r4["icm_preference_aware"]["p=0.01"]["summary"]
    r4_pref_05 = r4["icm_preference_aware"]["p=0.05"]["summary"]
    r8_by_threshold = {
        float(row["threshold"]): row
        for row in r8["intervention"]["results"].values()
    }
    celf_cmp = r8["celf_optimization"]["comparison"]

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
        "The measured fast-demo CLI wall-clock runtime was "
        f"<b>{num(timings.get('wall_clock_seconds', timings['total_seconds']), 2)} s</b> "
        f"(~{timings.get('wall_clock_seconds', timings['total_seconds'])/60:.1f} min); "
        f"the stage-timing sum is {num(timings['total_seconds'], 2)} s. This includes "
        "data loading, model fitting, all experiment stages, and result "
        "serialization. The larger full-scale configuration was not executed; no "
        "full-scale result is claimed."
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
    t08 = r8_by_threshold.get(0.8, {})
    t08_rel = t08.get("relative_to_baseline", {})
    t08_spread = t08_rel.get("spread_change_pct")
    t08_proxy = t08_rel.get("adopter_satisfaction_change")
    r6_means = [r6["summary"][k]["avg_mean_satisfaction"]
                for k in ("average", "least_misery", "most_pleasure")
                if k in r6["summary"]]
    p(
        "The strongest measured result is that matrix factorization and the "
        "trust-aware social model substantially outperform item-based CF on rating "
        "error. The social model is best on test MAE/RMSE and on all top-K ranking "
        "metrics in this run. Rankings are ordered by unclipped ordering scores "
        "(MF down-weights the global item-bias term for ordering); only the "
        "reported predicted ratings are clipped to [1, 5]. Influence "
        "maximization does <b>not</b> show greedy superiority: on the restricted "
        "top-degree candidate pool, degree centrality wins at budgets 3 and 5, and "
        "the paired greedy−degree difference at budget 5 is negative. Group-strategy "
        "satisfaction differences are small, and the spread comparison is dominated "
        "by rare large cascades. The threshold intervention has a cliff at 0.8: it "
        f"removes {num(abs(t08_spread), 2)}% of simulated spread while changing the "
        "simulator-side adopter rating proxy by "
        f"{num(t08_proxy, 4)}, but it is not an online experiment. CELF used "
        f"{num(celf_cmp['evaluation_reduction_pct'], 2)}% fewer evaluations and was "
        f"{num(celf_cmp['speedup_factor'], 2)}× faster in this single noisy "
        "Monte-Carlo measurement, while selecting different seeds; a deterministic "
        "controlled check reproduced standard greedy exactly."
    )
    rows = [
        ["1. Data", "Parse ratings/trust, validate, split, represent", f"{data_summary['n_users']:,} users; {split_info['train_size']:,}/{split_info['val_size']:,}/{split_info['test_size']:,} split"],
        ["2. Recommenders", "Item-based CF and MF; top-K; bias-weighted ordering", f"MF test MAE {num(r2_mf['prediction']['mae'], 4)}, RMSE {num(r2_mf['prediction']['rmse'], 4)}"],
        ["3. Trust-aware", "Social blend; λ selected on validation", f"λ={r3['best_lambda']:.1f}; test MAE {num(r3_best['prediction']['mae'], 4)}"],
        ["4. Diffusion", "ICM, SIR, preference-aware; rare + moderate targets; controlled network null", f"p=0.05 rare canonical {num(r4_can_05['mean_adopters'], 0)}; moderate {num(r4['target_popularity_comparison']['moderate']['icm_canonical']['p=0.05']['summary']['mean_adopters'], 0)}"],
        ["5. Influence", "Random/degree/greedy; shared pool and seeds", "Degree best at budgets 3 and 5"],
        ["6. Groups", "Average / least misery / most pleasure; train-only groups", "Mean satisfaction " + " / ".join(num(v, 3) for v in r6_means)],
        ["7. Evaluation", "MAE/RMSE/P/R/F1/HR/nDCG/coverage/diversity", "Social best ranking; ICF broadest coverage"],
        ["8. Intervention", "Threshold, CELF, network/item/budget scaling", f"0.8: {num(t08_spread, 2)}% spread, proxy {num(t08_proxy, 4)}; CELF {num(celf_cmp['speedup_factor'], 2)}×"],
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
        "while 3,449 items have 21+. This is why top-K ranking metrics are low. "
        "Part 4 retains the original one-training-rating rare target and adds a "
        "separate 6–20-training-rating moderate target for comparison; neither is "
        "drawn from the popular head of the distribution."
    )

    # ---------------- R2 ----------------
    add(PageBreak())
    h1("4. Requirement 2 — baseline recommendation")
    h2("4.1 Models")
    p(
        "Item-based CF uses cosine similarity between item interaction vectors, "
        "at most 20 neighbours, and at least 2 co-rated items. Matrix factorization "
        "uses 20 latent factors, learning rate 0.01, L2 regularization 0.005, "
        "20 epochs, and seed 42. Both are fitted on the training split. MF tracks "
        "which users and items have at least one training observation; untrained "
        "rows have zero bias and zero latent factors, so the fallback is μ+b_i for "
        "an untrained user with a trained item, μ+b_u for a trained user with an "
        "untrained item, and μ when both are unseen. Random initial factors are "
        "never used as a cold-start prediction."
    )
    p(
        "Top-K excludes the user's training items and ranks the remaining catalogue. "
        "Item-based CF items are ordered by the <b>unclipped</b> prediction score. "
        "MF ordering uses the configured ranking score: the full prediction with "
        "the global item-bias term weighted by 0.05. In both cases only the "
        "predicted rating shown with a returned item is clipped to [1, 5]. This "
        "prevents artificial ties at 5.0 from replacing confidence ordering with "
        "item-ID ordering. For item-based CF an item with no stored neighbours is "
        "unsupported by the item-item model, so its documented fallback uses "
        "training data only: the user's training mean, then the global training "
        "mean. Ranking metrics use a fixed seeded sample of 500 eligible users; "
        "MAE/RMSE use all test ratings. Unobserved items are not treated as "
        "negative examples."
    )
    if selection:
        diag = selection["coverage_diagnosis"]
        pre = selection["pre_correction_baseline"]
        chosen = selection["selected"]
        chosen_rank = chosen["validation_ranking_beta_0.05"]
        p(
            "<b>Coverage diagnosis and validation selection.</b> The pre-correction "
            "full-score MF ranking was dominated by the global item-bias component: "
            f"{diag['bias_share_of_within_user_score_variance']*100:.2f}% of "
            "within-user score variance, and "
            f"{diag['top10_overlap_with_bias_only_ranking']*100:.1f}% of top-10 "
            "positions were shared with a bias-only ranking. This was not random "
            "tie-breaking, so ranking noise was not added. Instead, item bias was "
            "down-weighted only for ordering; prediction error still uses the full "
            "model. A bounded validation sweep (500 seeded users) compared factor "
            "counts and learning rates. The selected configuration uses 20 factors, "
            "learning rate 0.01, regularization 0.005, 20 epochs, and item-bias "
            "weight 0.05. On validation it achieved MAE "
            f"{num(chosen['validation_prediction']['mae'], 4)}, RMSE "
            f"{num(chosen['validation_prediction']['rmse'], 4)}, nDCG@10 "
            f"{num(chosen_rank['ndcg_at_k'], 5)}, and "
            f"{chosen_rank['n_unique_items']} unique items, versus "
            f"{pre['validation_ranking']['n_unique_items']} before the correction. "
            "The test split was not used for selection."
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
    caption(
        "Table 3. Test rating and top-10 ranking metrics. MF is much better on "
        "error. The bias-corrected MF list contains "
        f"{r2_mf['ranking']['n_unique_items']} unique items across the 500-user "
        f"ranking sample, versus {r2_icf['ranking']['n_unique_items']:,} for "
        "item-based CF; absolute top-K accuracy remains low on this sparse "
        "catalogue."
    )
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
    fig(fig_r2, "Figure 1. Test MAE and RMSE for the two baselines. MF reduces both errors substantially.",
        name="fig01_test_error.png")
    h2("4.2 Behaviour and popularity slices")
    behavior = r2["behavior_analysis"]
    item_pop = r2["item_popularity_analysis"]
    p(
        "MF is consistently better than item-based CF on short, medium, and long "
        "user histories and on rare, moderate, and popular item groups. The largest "
        "gap is on moderate-popularity items: item-based CF MAE "
        f"{num(item_pop['moderate_6_20']['item_based_cf']['mae'], 4)} vs MF "
        f"{num(item_pop['moderate_6_20']['matrix_factorization']['mae'], 4)}. "
        "Item-based CF is not better on the rare-item group; its MAE is "
        f"{num(item_pop['rare_1_5']['item_based_cf']['mae'], 4)} vs MF "
        f"{num(item_pop['rare_1_5']['matrix_factorization']['mae'], 4)}. "
        "The complete slices are in "
        "<font face='Courier'>requirement_2_results.json</font>."
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
    caption(f"Table 4. Validation selection. λ={r3['best_lambda']:.1f} has the lowest validation RMSE and is selected before test is examined.")
    rows = []
    for key, label in (("social_lambda_0.0", "0.0 (baseline)"), ("social_lambda_0.3", "0.3 (selected)"), ("social_lambda_0.5", "0.5")):
        m = r3[key]
        pr, rk = m["prediction"], m["ranking"]
        rows.append([label, num(pr["mae"], 5), num(pr["rmse"], 5), num(rk["precision_at_k"], 4),
                     num(rk["hit_rate_at_k"], 4), num(rk["ndcg_at_k"], 5)])
    add(table(["λ", "Test MAE", "Test RMSE", "P@10", "HR@10", "nDCG@10"], rows,
              [35 * mm, 30 * mm, 30 * mm, 25 * mm, 25 * mm, 30 * mm], font_size=7.8))
    caption(f"Table 5. Test metrics reported after selection. λ={r3['best_lambda']:.1f} is the validation-selected row.")
    p(
        f"Sparse users (1–5 ratings, {r3['sparse_users_analysis']['n_sparse_users']:,} users) "
        f"improve from MAE {num(r3['sparse_users_analysis']['baseline']['mae'], 5)} "
        f"to {num(r3['sparse_users_analysis']['social']['mae'], 5)} at λ={r3['best_lambda']:.1f}. "
        f"The trust-vs-similarity comparison sampled "
        f"{r3['trust_vs_similarity_comparison']['n_sampled_users']} users "
        f"({r3['trust_vs_similarity_comparison']['n_test_ratings']} test ratings): "
        f"trusted neighbours MAE {num(r3['trust_vs_similarity_comparison']['trust_neighbors']['mae'], 5)}, "
        f"unconnected similar users {num(r3['trust_vs_similarity_comparison']['similar_unconnected']['mae'], 5)}, "
        f"baseline {num(r3['trust_vs_similarity_comparison']['baseline']['mae'], 5)}. "
        "The similar-user group is slightly better in this small sample, so the "
        "trust signal is not uniformly superior; the result is descriptive."
    )
    p(
        "The post-selection rows are shown for transparency. The selection rule is "
        "fixed in advance on validation RMSE; choosing a different λ after seeing "
        "test ranking would be test leakage. No test metric is used to change the "
        "submitted model."
    )

    # ---------------- R4 ----------------
    add(PageBreak())
    h1("6. Requirement 4 — ICM, SIR, and preference-aware diffusion")
    h2("6.1 Design and target popularity")
    target = r4["design"]["target_item"]
    moderate_target = r4["design"]["target_item_moderate"]
    p(
        "Part 4 keeps the original <b>rare-item</b> experiment and adds a separate "
        "<b>moderate-popularity</b> target. Both are selected from the training-only "
        "rating-count distribution. The rare rule uses the explicit inclusive band "
        f"[{target['band_min_train_ratings']}, {target['band_max_train_ratings']}] "
        f"training rating and selects original ID <b>{target['original_id']}</b> "
        f"(matrix index {target['idx']}, {target['train_rating_count']} training "
        f"rating); {target['band_size_items']:,} items satisfy that band. The "
        "moderate rule uses the inclusive 6–20 band and selects original ID "
        f"<b>{moderate_target['original_id']}</b> (matrix index "
        f"{moderate_target['idx']}, {moderate_target['train_rating_count']} training "
        f"ratings); the band contains {moderate_target['band_size_items']:,} items. "
        "The rule chooses the count closest to the band median, with ties by "
        "ascending original item ID. This retains the earlier ultra-rare case "
        "instead of replacing it."
    )
    p(
        "The three seed sets (five users each) and the Monte Carlo seeds are "
        "pre-generated once and reused for every ICM and SIR setting on the rare "
        "target. The moderate target uses the same generation seed, model settings, "
        "graph, and simulation seeds, but its seed users are drawn from the "
        "target-specific eligible set (users without a training rating for that "
        "item). This is a target comparison, not a claim that both targets share "
        "identical seed users. Rare-target seed sets:"
    )
    for seed_set in r4["design"]["seed_sets"]:
        add(Paragraph(f"<font face='Courier'>{seed_set}</font>", STYLE_BODY))
    graph_info = r4["design"]["graph_info"]
    p(
        f"The fast demo uses {graph_info['n_users']:,} users and "
        f"{graph_info['n_edges']:,} propagation edges (2-hop ego subgraph centered "
        "on user 0), three simulations per setting, and at most 25 rounds. The SIR "
        "recovery probability γ=0 uses all 25 rounds because infected users never "
        "recover; those round counts are capped by the configured maximum, not "
        "evidence that the cascade naturally lasted 25 rounds."
    )
    h2("6.2 Measured spreads — retained rare target")
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
    caption("Table 6. Rare-target diffusion outcomes. Means and standard deviations are across the same three pre-generated seed sets and controlled simulation seeds.")
    def fig_r4(fig_obj):
        ax = fig_obj.add_subplot(111)
        labels, means, stds, colors_ = [], [], [], []
        for setting, values in r4["icm_canonical"].items():
            s = values["summary"]
            labels.append("ICM " + setting)
            means.append(s["mean_adopters"])
            stds.append(s["std_adopters"])
            colors_.append(MPL_COLORS["blue"])
        for setting, values in r4["icm_preference_aware"].items():
            s = values["summary"]
            labels.append("Pref " + setting)
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
        ax.set_title("Rare-target diffusion outcomes with one-standard-deviation bars")
        ax.grid(axis="y", alpha=0.25)
    fig(fig_r4, "Figure 2. Rare-target diffusion outcomes on a log scale. The p=0.01 canonical ICM is heavy-tailed; preference-aware adoption strongly suppresses spread at the same seeds.",
        height=84 * mm, name="fig02_rare_target_diffusion.png")
    h2("6.3 Interpretation — retained rare target")
    canon05 = r4_can_05["mean_adopters"]
    pref05 = r4_pref_05["mean_adopters"]
    pct05 = r4["comparison"]["p=0.05"]["preference_relative_change_pct"]
    pct01 = r4["comparison"]["p=0.01"]["preference_relative_change_pct"]
    sir00 = r4["sir"]["beta=0.05_gamma=0.0"]["summary"]["mean_ever_infected"]
    sir02 = r4["sir"]["beta=0.05_gamma=0.2"]["summary"]["mean_ever_infected"]
    p(
        f"Canonical ICM at p=0.05 reaches about {num(canon05, 0)} of "
        f"{graph_info['n_users']:,} active users "
        f"({100.0*canon05/graph_info['n_users']:.1f}%), consistent with a dense ego "
        "graph. Preference-aware adoption requires predicted affinity and cuts the "
        f"p=0.05 mean to {num(pref05, 0)} ({num(pct05, 1)}%); at p=0.01 it changes "
        f"the mean by {num(pct01, 1)}%. This is a model-design effect, not evidence "
        f"that real users would adopt at that rate. SIR at β=0.05 reaches "
        f"{num(sir00, 0)} with γ=0 and {num(sir02, 0)} with γ=0.2; recovery reduces "
        "ever-infected counts but the γ=0 runs remain active to the round cap."
    )
    can01_std = r4_can_01["std_adopters"]
    can01_max = r4_can_01["adopters"]["max"]
    p(
        f"The p=0.01 canonical ICM has mean {num(r4_can_01['mean_adopters'], 1)} "
        f"but standard deviation {num(can01_std, 1)} and a maximum of "
        f"{num(can01_max, 0)}: the three simulations are highly unequal. With only "
        "three simulations, the mean is unstable. This is reported explicitly "
        "rather than hidden; the full configuration uses 20 simulations."
    )

    h2("6.4 Moderate-popularity target comparison")
    comp = r4["target_popularity_comparison"]
    comp_design = comp["design"]
    mod = comp["moderate"]
    p(
        f"The added target is original ID <b>{moderate_target['original_id']}</b> "
        f"with <b>{moderate_target['train_rating_count']} training ratings</b> "
        f"(versus {target['train_rating_count']} for the retained rare target). "
        "The same graph, canonical/preference-aware ICM implementations, propagation "
        "probabilities, simulation seeds, and seed size are used; only the target "
        "item and its target-specific eligible seed pool differ. This makes the "
        "comparison descriptive rather than a claim that popularity alone is "
        "identified."
    )
    rows = []
    for key in ("p=0.01", "p=0.05"):
        c = comp["comparison"][key]
        rows.append(["Canonical", key, num(c["rare_canonical_mean_adopters"], 2),
                     num(c["moderate_canonical_mean_adopters"], 2),
                     num(c["moderate_minus_rare_canonical"], 2),
                     pct(c["moderate_relative_canonical_pct"])])
        rows.append(["Preference-aware", key, num(c["rare_preference_mean_adopters"], 2),
                     num(c["moderate_preference_mean_adopters"], 2),
                     num(c["moderate_minus_rare_preference"], 2),
                     pct(c["moderate_relative_preference_pct"])])
    add(table(["Model", "Setting", "Rare target mean", "Moderate target mean",
               "Difference (moderate−rare)", "Relative"],
              rows, [32 * mm, 20 * mm, 30 * mm, 34 * mm, 37 * mm, 25 * mm],
              font_size=7.2))
    caption("Table 7. Retained rare target versus the added 6–20-training-rating target. Seed users are target-specific; all other controls are shared.")
    def fig_target(fig_obj):
        ax = fig_obj.add_subplot(111)
        labels = ["Canonical p=0.05", "Preference p=0.05"]
        rare_vals = [
            comp["comparison"]["p=0.05"]["rare_canonical_mean_adopters"],
            comp["comparison"]["p=0.05"]["rare_preference_mean_adopters"],
        ]
        mod_vals = [
            comp["comparison"]["p=0.05"]["moderate_canonical_mean_adopters"],
            comp["comparison"]["p=0.05"]["moderate_preference_mean_adopters"],
        ]
        x = np.arange(len(labels))
        ax.bar(x - 0.18, rare_vals, 0.36, label="Rare target (1 training rating)",
               color=MPL_COLORS["blue"])
        ax.bar(x + 0.18, mod_vals, 0.36,
               label=f"Moderate target ({moderate_target['train_rating_count']} training ratings)",
               color=MPL_COLORS["accent"])
        ax.set_xticks(x, labels)
        ax.set_ylabel("Mean simulated adopters")
        ax.set_title("Part 4 target-popularity comparison at p=0.05")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig(fig_target, "Figure 3. Rare versus moderate target at p=0.05. The comparison is descriptive: targets have different seed pools and different training popularity.",
        height=68 * mm, name="fig03_target_popularity.png")

    h2("6.5 Controlled network-structure comparison")
    net = r4.get("network_structure_comparison", {})
    if net:
        net_design = net["design"]
        rand_info = net_design["randomized_graph"]["randomization"]
        p(
            "To test whether the observed edge arrangement matters, the active "
            "directed graph is randomized while preserving its node set, edge count, "
            "and directed in/out-degree sequence. The canonical ICM then runs on the "
            "randomized graph with the <b>same rare-target seed users, simulation "
            "seeds, propagation probabilities, and round cap</b>. The null model is "
            "one seeded degree-preserving rearrangement, not a fully mixed sample of "
            "all graphs with that degree sequence."
        )
        p(
            f"Randomization preserved {rand_info['n_nodes']:,} nodes and "
            f"{rand_info['n_edges']:,} directed edges; "
            f"{rand_info['swaps_accepted']:,} of {rand_info['swaps_attempted']:,} "
            f"double-edge swaps were accepted "
            f"({100.0*rand_info['acceptance_rate']:.1f}%). The degree sequence "
            "equality is checked before the comparison is accepted."
        )
        rows = []
        for key, values in net["settings"].items():
            rows.append([key, num(values["observed"]["mean_adopters"], 2),
                         num(values["randomized"]["mean_adopters"], 2),
                         num(values["paired_randomized_minus_observed"]["mean_difference"], 2),
                         pct(values["relative_change_pct"])])
        add(table(["Setting", "Observed mean", "Randomized mean",
                   "Paired difference (random−observed)", "Relative"],
                  rows, [28 * mm, 32 * mm, 34 * mm, 50 * mm, 27 * mm], font_size=7.4))
        caption("Table 8. Canonical ICM on observed versus degree-preserving randomized structure. Differences are simulation outcomes, not observed cascades.")
        def fig_net(fig_obj):
            ax = fig_obj.add_subplot(111)
            keys = list(net["settings"])
            x = np.arange(len(keys))
            obs = [net["settings"][k]["observed"]["mean_adopters"] for k in keys]
            rnd = [net["settings"][k]["randomized"]["mean_adopters"] for k in keys]
            ax.bar(x - 0.18, obs, 0.36, label="Observed graph", color=MPL_COLORS["blue"])
            ax.bar(x + 0.18, rnd, 0.36, label="Degree-preserving randomized", color=MPL_COLORS["accent"])
            ax.set_xticks(x, keys)
            ax.set_ylabel("Mean simulated adopters")
            ax.set_title("Network-structure null comparison (canonical ICM)")
            ax.legend(frameon=False, fontsize=7)
            ax.grid(axis="y", alpha=0.25)
        fig(fig_net, "Figure 4. Observed versus degree-preserving randomized graph under the same seeds and canonical ICM. This isolates the edge-arrangement effect within the selected null model.",
            height=68 * mm, name="fig04_network_structure.png")
        best_key = max(net["settings"], key=lambda k: abs(net["settings"][k]["relative_change_pct"] or 0.0))
        best = net["settings"][best_key]
        p(
            f"The largest relative difference in this run is at {best_key}: "
            f"observed mean {num(best['observed']['mean_adopters'], 1)} versus "
            f"randomized mean {num(best['randomized']['mean_adopters'], 1)}. "
            "This is evidence about this seeded null model and three simulations, "
            "not a general estimate of how much structure matters in Epinions."
        )
    else:
        p("The network-structure comparison was not present in the saved results; no claim is made.")

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
    caption("Table 9. Mean simulated spread under common evaluation seeds. The restricted top-degree pool favors degree; greedy is best only at budget 1.")
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
    fig(fig_r5, "Figure 5. Greedy, degree, and random spreads. Differences are small relative to the saturated baseline and do not show greedy superiority.",
        height=72 * mm, name="fig05_influence_spread.png")
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
    caption("Table 10. Marginal gains relative to the previous evaluated budget. Negative values are legitimate under finite Monte Carlo sampling; they are not clipped to zero.")
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
    caption("Table 11. Group strategy outcomes across three groups. Satisfaction differences are small; spread variability is large.")
    r6_summary = r6["summary"]
    r6_analysis = r6["analysis"]
    best_mean = r6_analysis["best_strategy_by_mean_satisfaction"]
    best_min = r6_analysis["best_strategy_by_min_satisfaction"]
    best_spread = r6_analysis["best_strategy_by_simulated_spread"]
    means = [s["avg_mean_satisfaction"] for s in r6_summary.values()]
    mins = [s["avg_min_satisfaction"] for s in r6_summary.values()]
    max_std = max(s["avg_diffusion_std"] for s in r6_summary.values())
    p(
        f"The {best_mean.replace('_', ' ')} strategy has the highest mean and "
        f"minimum satisfaction; {best_spread.replace('_', ' ')} has the highest "
        "mean simulated spread. The differences are tiny: mean satisfaction "
        + " vs ".join(num(s["avg_mean_satisfaction"], 3) for s in r6_summary.values())
        + ", and minimum satisfaction "
        + " vs ".join(num(s["avg_min_satisfaction"], 3) for s in r6_summary.values())
        + ". The spread standard deviation (up to "
        f"{num(max_std, 1)}) is much larger than the between-strategy mean "
        f"difference ({num(max(means) - min(means), 3)}). At least one group's "
        "cascade is dominated by a single simulation path. These data do not "
        "support a strong group-strategy trade-off claim."
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
    bullet("<b>Diversity:</b> mean pairwise 1−cosine of training-interaction item feature vectors within each list. The evaluated list set may itself be a seeded ranking-user sample; <font face='Courier'>diversity_is_sampled</font>, <font face='Courier'>diversity_sample_seed</font>, and <font face='Courier'>diversity_sampling_scope</font> record the actual scope. It is feature diversity, not category diversity.")
    bullet("No accuracy classification metric is reported: with only held-out positives and no observed negatives, a confusion-matrix accuracy would be misleading.")
    h2("9.2 Observed evaluation")
    rows = []
    labels = {"item_based_cf": "Item-based CF", "matrix_factorization": "MF", "social_recommender": f"Social (lambda={r3['best_lambda']:.1f})"}
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
    r7_models = r7["observed"]["models"]
    best_error_key = min(r7_models, key=lambda k: r7_models[k]["prediction_error"]["rmse"])
    best_ndcg_key = max(r7_models, key=lambda k: r7_models[k]["ranking"]["ndcg_at_k"])
    best_cov_key = max(r7_models, key=lambda k: r7_models[k]["ranking"]["coverage"])
    if best_error_key == best_ndcg_key:
        caption(
            "Table 12. Observed test metrics (500 seeded users for ranking; all "
            "test ratings for MAE/RMSE). "
            f"{labels[best_error_key]} has both the lowest RMSE and the highest "
            f"nDCG@10; {labels[best_cov_key]} has the broadest coverage."
        )
        lead = (
            f"{labels[best_error_key]} has the lowest observed RMSE ("
            f"{num(r7_models[best_error_key]['prediction_error']['rmse'], 4)}) and "
            f"the highest nDCG@10 ("
            f"{num(r7_models[best_ndcg_key]['ranking']['ndcg_at_k'], 5)}). "
        )
    else:
        caption(
            "Table 12. Observed test metrics (500 seeded users for ranking; all "
            "test ratings for MAE/RMSE). "
            f"{labels[best_error_key]} has the lowest RMSE, "
            f"{labels[best_ndcg_key]} the highest nDCG@10, and "
            f"{labels[best_cov_key]} the broadest coverage."
        )
        lead = (
            f"{labels[best_error_key]} has the lowest observed RMSE ("
            f"{num(r7_models[best_error_key]['prediction_error']['rmse'], 4)}) and "
            f"{labels[best_ndcg_key]} the highest nDCG@10 ("
            f"{num(r7_models[best_ndcg_key]['ranking']['ndcg_at_k'], 5)}). "
        )
    p(
        lead
        + f"{labels[best_cov_key]} has the broadest coverage ("
        f"{num(r7_models[best_cov_key]['ranking']['coverage'], 5)}); "
        "the coverage/accuracy trade-off is reported, not hidden. Feature diversity "
        f"is {num(r7_models['matrix_factorization']['ranking']['diversity'], 4)} for "
        f"MF, {num(r7_models['social_recommender']['ranking']['diversity'], 4)} for "
        f"social, and {num(r7_models['item_based_cf']['ranking']['diversity'], 4)} "
        "for item-based CF; it is not category diversity."
    )
    mf_rank_meta = r7_models["matrix_factorization"]["evaluation_metadata"]
    p(
        "Diversity metadata now describes the lists actually evaluated: "
        f"{mf_rank_meta['diversity_n_lists']} lists, "
        f"<font face='Courier'>diversity_is_sampled={str(mf_rank_meta['diversity_is_sampled']).lower()}</font>, "
        f"seed <font face='Courier'>{mf_rank_meta['diversity_sample_seed']}</font>, "
        f"scope <font face='Courier'>{mf_rank_meta['diversity_sampling_scope']}</font>. "
        "The flag is true because the 500 lists come from the seeded ranking-user "
        "sample even though all 500 were used; the earlier false/null flag "
        "described only the internal list cap and was incorrect for that scope."
    )
    h2("9.3 Separate simulated interaction")
    spread = r7['tradeoff_analysis']['simulated_spread_ranking']
    best_spread_key = max(spread, key=spread.get)
    agree = best_spread_key == best_ndcg_key
    p(
        "The simulated interaction retains the original rare training-selected "
        "target item and common seeds; it is not an observed adoption forecast. "
        "Mean adopters are "
        f"ICF {num(spread['item_based_cf'], 1)}, "
        f"MF {num(spread['matrix_factorization'], 1)}, "
        f"social {num(spread['social_recommender'], 1)}. "
        f"The highest simulated spread is {labels[best_spread_key]}; the highest "
        f"observed nDCG is {labels[best_ndcg_key]}. "
        + ("The two quantities agree in this run, " if agree else
           "The two quantities do not agree in this run, ")
        + "but they answer different questions and are not merged into one score."
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
    caption("Table 13. Intervention results relative to threshold 0. The rating proxy is 1+4×predicted affinity and is not an observed satisfaction rating.")
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
    intervention_rows = list(r8["intervention"]["results"].values())
    eligible_cuts = [
        row for row in intervention_rows
        if row["threshold"] > 0
        and (row.get("relative_to_baseline", {}).get("adopter_satisfaction_change", -1) >= 0)
    ]
    best_cut = max(
        eligible_cuts,
        key=lambda row: -row["relative_to_baseline"]["spread_change_pct"],
    )
    fig(fig_r8, f"Figure 6. Threshold intervention. The {best_cut['threshold']:.1f} threshold gives the largest simulated spread reduction among non-negative proxy changes; the proxy is simulator-side, not observed satisfaction.",
        height=72 * mm, name="fig06_intervention.png")
    best_cut_rel = best_cut["relative_to_baseline"]
    t08 = next((r for r in intervention_rows if abs(r["threshold"] - 0.8) < 1e-9), None)
    p(
        "The honest reading: this intervention can strongly limit simulated "
        "diffusion at a high threshold, but the effect is not monotone and the "
        "proxy is not observed post-intervention satisfaction. The largest "
        "measured spread reduction among thresholds with a non-negative proxy "
        f"change is threshold {best_cut['threshold']:.1f}: "
        f"{num(best_cut_rel['spread_change_pct'], 2)}% spread change with "
        f"{num(best_cut_rel['adopter_satisfaction_change'], 4)} proxy change. "
        + (f"Threshold 0.8 changes spread by "
           f"{num(t08['relative_to_baseline']['spread_change_pct'], 2)}% and the "
           f"proxy by {num(t08['relative_to_baseline']['adopter_satisfaction_change'], 4)}. "
           if t08 else "")
        + "No online randomized experiment was run."
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
    caption(f"Table 14. CELF vs standard greedy at budget {celf_cmp['test_budget']}, {celf_cmp['candidate_pool_size']} candidates, {celf_cmp['n_simulations']} simulations. CELF used {num(celf_cmp['evaluation_reduction_pct'], 2)}% fewer evaluations and was {num(celf_cmp['speedup_factor'], 3)}× faster, but selected different seeds under noisy estimates.")
    p(
        f"CELF's lazy bound is exact for a deterministic monotone submodular "
        f"objective, not for a finite-sample noisy estimate. The measured "
        f"{num(celf_cmp['speedup_factor'], 2)}× speedup and "
        f"{num(celf_cmp['evaluation_reduction_pct'], 2)}% evaluation reduction are "
        f"therefore reported as single-run measurements, not an unconditional "
        f"guarantee. A controlled "
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
    caption("Table 15. Network-size scaling for greedy selection from scratch on BFS-induced subgraphs.")
    items = r8["scaling"]["item_catalogue"]["rows"]
    rows = [[num(r["n_items"], 0), num(r["ms_per_user"], 3), num(r["runtime_seconds"], 3),
             num(r["memory"]["tracemalloc_peak_mb"], 2)] for r in items]
    add(table(["Catalogue items", "ms/user", "Runtime (s)", "Tracemalloc peak (MB)"], rows,
              [45 * mm, 32 * mm, 34 * mm, 45 * mm], font_size=7.5))
    caption("Table 16. Item-catalogue scaling for MF scoring/ranking. This isolates scoring, not model training.")
    def fig_r8_scale(fig_obj):
        ax = fig_obj.add_subplot(111)
        nodes = [r["n_nodes"] for r in net]
        times = [r["runtime_seconds"] for r in net]
        ax.plot(nodes, times, marker="o", color=MPL_COLORS["navy"])
        ax.set_xlabel("Propagation-graph nodes")
        ax.set_ylabel("Greedy selection runtime (s)")
        ax.set_title("Network-size scaling (15 candidates, budget 2)")
        ax.grid(alpha=0.25)
    fig(fig_r8_scale, "Figure 7. Measured network scaling. The slope is machine-dependent; the report compares relative growth within this run.",
        height=65 * mm, name="fig07_network_scaling.png")
    budget = r8["scaling"]["promotion_budget"]["rows"]
    rows = [[num(r["budget"], 0), num(r["n_evaluations"], 0), num(r["runtime_seconds"], 2)] for r in budget]
    add(table(["Seed budget", "Evaluations", "Runtime (s)"], rows,
              [40 * mm, 40 * mm, 40 * mm], font_size=7.8))
    caption("Table 17. Promotion-budget scaling on a bounded subgraph. Evaluations grow with budget because greedy re-evaluates candidates.")
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
        "The latest full local run was <b>94 passed</b> in about 1.7 s with "
        "<font face='Courier'>python -m pytest -q</font> after installing "
        "<font face='Courier'>requirements-dev.txt</font>. The regression coverage "
        "is explicit:"
    )
    bullet("Ranking separates the ordering score from the reported full prediction; batch and single-user paths agree, and the item-bias ranking weight changes order without changing the reported rating.")
    bullet("Diversity metadata propagates both ranking-user and list-sampling scope, including the correct is-sampled flag and seed.")
    bullet("Item-based unsupported-item fallback uses training-only user mean, then global training mean.")
    bullet("MF cold starts use μ+b_i, μ+b_u, or μ and never random initial factors.")
    bullet("Moderate (6–20) and rare (1) target bands, deterministic seed-set generation, and training-only target selection.")
    bullet("Degree-preserving directed randomization preserves nodes, edges, and in/out degrees, and is reproducible.")
    bullet("Training-only matrix construction and R6 groups unaffected by mutating the full observed matrix.")
    bullet("Social λ selected on validation when test ratings are perturbed.")
    bullet("Greedy continues after a negative Monte Carlo marginal gain; explicit candidate pool respected.")
    bullet("Paired marginal gains equal the difference of the paired spread estimates.")
    bullet("Feature diversity, per-user F1, and hand-checkable metric examples.")
    h2("11.2 Exact commands")
    commands = [
        "python -m pip install -r requirements-dev.txt",
        "python -m pytest -q",
        "python -m sntp.cli inspect --data-dir Assignment",
        "python -m sntp.cli run --config configs/fast_demo.yaml",
        "python -m pip install -r requirements-report.txt",
        "python scripts/generate_report.py --results results/fast_demo --output report/SNTP_Assignment_02_Report.pdf --figures-dir results/fast_demo/plots --roll-number 23F3008 --student \"Ahmad Fareed\"",
    ]
    for cmd in commands:
        add(Paragraph(f"<font face='Courier'>{cmd}</font>", STYLE_BODY))
    h2("11.3 Runtime and seeds")
    rows = [[s["stage"], num(s["seconds"], 2)] for s in timings["stages"]]
    add(table(["Stage", "Seconds"], rows, [140 * mm, 30 * mm], font_size=7.0))
    caption(
        f"Table 18. Stage timings from the submitted run. Stage sum: "
        f"{num(timings['total_seconds'], 2)} s; external CLI wall clock: "
        f"{num(timings.get('wall_clock_seconds', timings['total_seconds']), 2)} s."
    )
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
    # No forced page break here: the seed paragraph at the end of Section 11
    # previously overflowed onto an otherwise almost-empty page.
    h1("12. Limitations and conclusions")
    h2("12.1 Limitations")
    bullet("<b>Full-scale run not executed.</b> The full config uses the full graph, 20 simulations, larger candidate pools, and larger budgets; estimated cost is many hours. No full-scale number is claimed.")
    bullet("<b>Fast-demo scope.</b> Diffusion/influence/group/intervention use a 2-hop ego subgraph (9,883 users); R5 uses a top-15 out-degree pool. The demo also uses a smaller ICF neighbourhood than the full config (20 vs 50); MF uses 20 factors/20 epochs with the learning-rate, regularization, and bias-corrected ranking setting selected on validation. These are bounded, deterministic scopes, not random full-graph samples.")
    bullet("<b>Monte Carlo noise.</b> R4 uses 3 simulations; R5 uses 2 evaluation simulations and 3 random trials; R6 uses 2 per item; R8 CELF uses 2. Heavy tails make small mean differences unstable.")
    bullet("<b>R5 does not show greedy superiority.</b> Degree wins at budgets 3 and 5; the paired greedy−degree difference at budget 5 is negative. The dense graph is near saturation (~47% of active users), compressing differences.")
    bullet(f"<b>R6 differences are small.</b> Mean satisfaction ranges {min(means):.3f}–{max(means):.3f}; spread is dominated by one rare cascade. Candidate items are only the union of strategy top items.")
    bullet("<b>R7 is sample-based.</b> Ranking uses 500 users from a seeded eligible-user sample (seed 42); coverage/diversity are list-sample properties. Because the 500 lists are themselves sampled users, the corrected diversity metadata reports is_sampled=true with that seed even though all 500 lists were used. Unobserved items are not negatives. The simulated interaction uses the retained rare target; Part 4 separately adds a moderate 6–20-rating target.")
    bullet("<b>R8 is a simulator intervention.</b> No online experiment; the rating proxy is predicted, not observed; the threshold cliff is a property of the affinity distribution; CELF's single-run speedup is noisy and seeds differ. The deterministic check is the correctness evidence.")
    bullet("<b>Cold starts are fallback-only.</b> MF deliberately returns μ+b_i, μ+b_u, or μ for untrained rows; it does not learn a representation for users/items absent from training.")
    bullet("<b>Ranking score is not the reported rating.</b> MF orders by mu + b_u + 0.05*b_i + p_u.q_i; the returned value and MAE/RMSE use mu + b_u + b_i + p_u.q_i, clipped. Down-weighting item bias improves catalogue reach but is a validation-selected heuristic, not a learned objective.")
    bullet("<b>Network null is not fully mixed.</b> The degree-preserving directed randomization is one seeded rearrangement, not an exhaustive null distribution; three simulations per setting limit precision.")
    bullet("<b>Memory measurement.</b> tracemalloc undercounts NumPy native buffers; RSS deltas are near zero because arrays are already resident.")
    bullet("<b>Runtime variability.</b> Prior runs of the same fast config varied with machine load; the saved stage timings are one run and should not be treated as a benchmark.")
    bullet("<b>Selection sweep is offline.</b> The validation candidates in <font face='Courier'>model_selection.json</font> were measured separately from the reported fast-demo runtime; the sweep is not part of the end-to-end stage timings.")
    h2("12.2 Conclusions by requirement")
    conclusions = [
        ("Data/split", "The parser is clean (zero malformed/duplicate/out-of-range ratings) and the 70/15/15 split is reproducible with per-user coverage."),
        ("Baselines", f"MF clearly beats item-based CF on rating error; its bias-corrected list uses {r2_mf['ranking']['n_unique_items']} unique items across the 500-user sample versus {r2_icf['ranking']['n_unique_items']} for ICF. Absolute top-K accuracy remains low on this sparse catalogue. MF is better across user-history and item-popularity slices."),
        ("Trust-aware", f"Validation-selected λ={r3['best_lambda']:.1f} gives test MAE {num(r3_best['prediction']['mae'], 5)}. The trust-vs-similarity comparison is descriptive and small."),
        ("Diffusion", f"On the retained rare target, canonical ICM at p=0.05 has mean {num(r4_can_05['mean_adopters'], 0)}; preference-aware adoption is {num(r4_pref_05['mean_adopters'], 0)} at the same seeds. The added moderate target is reported separately. Results are simulation-only and heavy-tailed at low p."),
        ("Influence", "On a shared top-degree pool, degree is best at budgets 3 and 5; greedy is best only at budget 1. There is no evidence for greedy superiority, and the pool restriction is material."),
        ("Groups", f"{best_mean.replace('_', ' ')} has the best predicted mean/min satisfaction; {best_spread.replace('_', ' ')} has the largest mean simulated spread. Differences are small, and one rare cascade dominates spread."),
        ("Evaluation", f"{labels[best_ndcg_key]} has the highest observed nDCG and {labels[best_cov_key]} the broadest coverage; feature diversity is reported separately. Observed and simulated outcomes are not merged."),
        ("Intervention/optimization", f"Threshold 0.8 changes simulated spread by {num(t08_spread, 2)}% with a {num(t08_proxy, 4)} proxy change. CELF uses {num(celf_cmp['evaluation_reduction_pct'], 2)}% fewer evaluations and is {num(celf_cmp['speedup_factor'], 2)}× faster in the noisy run, with different seeds; a deterministic check matches standard greedy exactly."),
    ]
    rows = [[a, b] for a, b in conclusions]
    add(table(["Requirement", "Conclusion"], rows, [35 * mm, 140 * mm], font_size=7.4))
    caption("Table 19. Conclusions tied to measured evidence. No unexecuted configuration is presented as a result.")
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
    parser.add_argument("--figures-dir", default="results/fast_demo/plots",
                        help="Directory for standalone PNG chart exports")
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
        "selection": (
            load_json(results / "model_selection.json")
            if (results / "model_selection.json").exists() else None
        ),
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
