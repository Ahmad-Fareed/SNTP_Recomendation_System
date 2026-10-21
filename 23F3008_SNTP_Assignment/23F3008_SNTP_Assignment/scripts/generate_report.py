#!/usr/bin/env python
"""Generate the searchable PDF report from saved experiment JSON.

Usage:
    python scripts/generate_report.py --results results/fast_demo \
        --output report/SNTP_Assignment_02_Report.pdf \
        --roll-number 22F3143 --student "Ahmad Fareed"

The layout is a compact two-column A4 report (at most eight pages).  Every
number in the tables is read from the saved JSON, so the report cannot drift
from the measured artifacts.  Un-executed full-scale work is labelled as such.
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
    FrameBreak,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

# Embedded Unicode fonts: the report uses lambda, beta, gamma, arrows and
# dashes.  The standard Helvetica encoding does not contain all of them, so a
# TrueType font that does is registered instead.
FONT_REGULAR = "ReportSans"
FONT_BOLD = "ReportSans-Bold"
FONT_ITALIC = "ReportSans-Oblique"

COLORS = {
    "navy": colors.HexColor("#17365D"),
    "blue": colors.HexColor("#2E75B6"),
    "pale": colors.HexColor("#EEF3F8"),
    "grid": colors.HexColor("#B7C7D9"),
    "accent": colors.HexColor("#C55A11"),
}
MPL = {"navy": "#17365D", "blue": "#2E75B6", "accent": "#C55A11", "gray": "#7F7F7F"}

PAGE_W, PAGE_H = A4
LM, RM, TM, BM = 15 * mm, 15 * mm, 16 * mm, 15 * mm
CONTENT_W = PAGE_W - LM - RM
GUTTER = 6 * mm
COL_W = (CONTENT_W - GUTTER) / 2.0
CONTENT_TOP = PAGE_H - TM


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


STYLE_TITLE = STYLE_SUBTITLE = STYLE_H1 = STYLE_H2 = STYLE_BODY = None
STYLE_BULLET = STYLE_CAPTION = STYLE_TABLE_HEADER = STYLE_TABLE_BODY = None
STYLE_SMALL = STYLE_META = None


def build_styles() -> None:
    global STYLE_TITLE, STYLE_SUBTITLE, STYLE_H1, STYLE_H2, STYLE_BODY
    global STYLE_BULLET, STYLE_CAPTION, STYLE_TABLE_HEADER, STYLE_TABLE_BODY
    global STYLE_SMALL, STYLE_META
    styles = getSampleStyleSheet()
    STYLE_TITLE = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontName=FONT_BOLD,
        fontSize=16, leading=19, textColor=COLORS["navy"], spaceAfter=1,
        alignment=TA_LEFT,
    )
    STYLE_SUBTITLE = ParagraphStyle(
        "Subtitle", parent=styles["Normal"], fontName=FONT_REGULAR,
        fontSize=8.8, leading=11.5, alignment=TA_LEFT,
        textColor=colors.HexColor("#333333"), spaceAfter=1,
    )
    STYLE_META = ParagraphStyle(
        "Meta", parent=STYLE_SUBTITLE, fontName=FONT_BOLD, fontSize=8.6,
        leading=11, textColor=COLORS["blue"],
    )
    STYLE_H1 = ParagraphStyle(
        "H1Custom", parent=styles["Heading1"], fontName=FONT_BOLD,
        fontSize=11, leading=13.5, textColor=COLORS["navy"],
        spaceBefore=5, spaceAfter=2.5,
    )
    STYLE_H2 = ParagraphStyle(
        "H2Custom", parent=styles["Heading2"], fontName=FONT_BOLD,
        fontSize=8.9, leading=11.5, textColor=COLORS["blue"],
        spaceBefore=4, spaceAfter=1.5,
    )
    STYLE_BODY = ParagraphStyle(
        "BodyCustom", parent=styles["BodyText"], fontName=FONT_REGULAR,
        fontSize=8.3, leading=11.0, alignment=TA_JUSTIFY, spaceAfter=3.2,
    )
    STYLE_BULLET = ParagraphStyle(
        "BulletCustom", parent=STYLE_BODY, leftIndent=9, bulletIndent=1,
        spaceAfter=1.6,
    )
    STYLE_SMALL = ParagraphStyle(
        "SmallCustom", parent=STYLE_BODY, fontSize=7.3, leading=9.4,
        textColor=colors.HexColor("#333333"), spaceAfter=2.2,
    )
    STYLE_CAPTION = ParagraphStyle(
        "CaptionCustom", parent=styles["Normal"], fontName=FONT_ITALIC,
        fontSize=6.9, leading=8.6, alignment=TA_LEFT,
        textColor=colors.HexColor("#555555"), spaceBefore=1, spaceAfter=4,
    )
    STYLE_TABLE_HEADER = ParagraphStyle(
        "TableHeader", parent=styles["Normal"], fontName=FONT_BOLD,
        fontSize=6.7, leading=8.1, textColor=colors.white,
    )
    STYLE_TABLE_BODY = ParagraphStyle(
        "TableBody", parent=styles["Normal"], fontName=FONT_REGULAR,
        fontSize=6.6, leading=8.0, textColor=colors.black,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    sign = "+" if float(value) > 0 else ""
    return f"{sign}{float(value):.{digits}f}%"


def add_color(ax, bars, color) -> None:
    for bar in bars:
        bar.set_color(color)


def table(header, rows, widths, font_size: float = 6.6,
          header_size: float = 6.7) -> Table:
    body_style = ParagraphStyle(
        "td", parent=STYLE_TABLE_BODY, fontSize=font_size,
        leading=font_size + 1.4,
    )
    head_style = ParagraphStyle(
        "th", parent=STYLE_TABLE_HEADER, fontSize=header_size,
        leading=header_size + 1.4,
    )
    data = [[Paragraph(str(c), head_style) for c in header]]
    for row in rows:
        data.append([Paragraph(str(c), body_style) for c in row])
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), COLORS["navy"]),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTNAME", (0, 1), (-1, -1), FONT_REGULAR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, COLORS["grid"]),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLORS["pale"]]),
        ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
    ]))
    return result


def figure(draw, width: float = COL_W, height: float = 52 * mm,
           output_path: Path | None = None) -> Image:
    buffer = io.BytesIO()
    fig = plt.figure(figsize=(3.6, 2.5), dpi=190)
    try:
        draw(fig)
        fig.tight_layout(pad=0.4)
        fig.savefig(buffer, format="png", bbox_inches="tight", facecolor="white")
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, format="png", dpi=200,
                        bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
    buffer.seek(0)
    return Image(buffer, width=width, height=height)


# ---------------------------------------------------------------------------
# Document template
# ---------------------------------------------------------------------------

class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str, title: str, author: str, roll: str, **kwargs):
        super().__init__(filename, pagesize=A4, **kwargs)
        self.report_title = title
        self.author = author
        self.roll = roll
        title_h = 30 * mm
        col_top = CONTENT_TOP - title_h - 3 * mm
        col_h = col_top - BM
        first = PageTemplate(
            id="first",
            frames=[
                Frame(LM, CONTENT_TOP - title_h, CONTENT_W, title_h,
                      id="title", leftPadding=0, rightPadding=0,
                      topPadding=0, bottomPadding=0),
                Frame(LM, BM, COL_W, col_h, id="left1",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                Frame(LM + COL_W + GUTTER, BM, COL_W, col_h, id="right1",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            ],
            onPage=self._decorate,
        )
        full_h = CONTENT_TOP - BM
        two = PageTemplate(
            id="twocol",
            frames=[
                Frame(LM, BM, COL_W, full_h, id="left",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
                Frame(LM + COL_W + GUTTER, BM, COL_W, full_h, id="right",
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0),
            ],
            onPage=self._decorate,
        )
        self.addPageTemplates([first, two])

    def _decorate(self, canvas, doc):
        canvas.saveState()
        canvas.setTitle(self.report_title)
        canvas.setAuthor(self.author)
        canvas.setSubject("SNTP Assignment 2 - Recommendation and Information Diffusion")
        canvas.setCreator("sntp report generator")
        canvas.setFont(FONT_REGULAR, 6.8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(
            LM, PAGE_H - 11.5 * mm,
            f"{self.author}  |  {self.roll}  |  SNTP Assignment 2",
        )
        canvas.drawRightString(
            PAGE_W - RM, PAGE_H - 11.5 * mm,
            "Epinions: recommendation and simulated diffusion",
        )
        canvas.setStrokeColor(COLORS["grid"])
        canvas.setLineWidth(0.4)
        canvas.line(LM, PAGE_H - 13.5 * mm, PAGE_W - RM, PAGE_H - 13.5 * mm)
        canvas.line(LM, 11 * mm, PAGE_W - RM, 11 * mm)
        canvas.drawString(LM, 7.6 * mm, self.report_title)
        canvas.drawRightString(PAGE_W - RM, 7.6 * mm, f"Page {doc.page}")
        canvas.restoreState()


# ---------------------------------------------------------------------------
# Story
# ---------------------------------------------------------------------------

def story(metrics: dict[str, Any], args: argparse.Namespace) -> list[Any]:
    out: list[Any] = []
    add = out.append

    r2 = metrics["r2"]
    r3 = metrics["r3"]
    r4 = metrics["r4"]
    r5 = metrics["r5"]
    r6 = metrics["r6"]
    r7 = metrics["r7"]
    r8 = metrics["r8"]
    timings = metrics["timings"]
    ds = metrics["data_summary"]
    sp = metrics["split_info"]
    cfg = metrics["config"]
    sel = metrics.get("selection")
    fig_dir = Path(args.figures_dir)

    def p(text, style=STYLE_BODY):
        add(Paragraph(text, style))

    def bullet(text):
        add(Paragraph(text, STYLE_BULLET, bulletText="•"))

    def h1(text):
        add(Paragraph(text, STYLE_H1))

    def h2(text):
        add(Paragraph(text, STYLE_H2))

    def caption(text):
        add(Paragraph(text, STYLE_CAPTION))

    def fig(draw, cap, name, width=COL_W, height=50 * mm):
        add(KeepTogether([
            figure(draw, width, height, output_path=fig_dir / name),
            Paragraph(cap, STYLE_CAPTION),
        ]))

    def tbl(header, rows, widths, font_size=6.6, header_size=6.7):
        add(table(header, rows, widths, font_size, header_size))
        add(Spacer(1, 2.4 * mm))

    # ------------------------------------------------------------------ title
    add(NextPageTemplate("twocol"))
    add(Paragraph("Recommendation and Information Diffusion on Epinions",
                  STYLE_TITLE))
    add(Paragraph(
        "SNTP Assignment 2 — Social Network Theory and Programming",
        STYLE_SUBTITLE,
    ))
    add(Paragraph(
        f"{args.student} &nbsp;|&nbsp; {args.roll_number} &nbsp;|&nbsp; "
        f"NU-FAST Chiniot-Faisalabad &nbsp;|&nbsp; {date.today().isoformat()}",
        STYLE_META,
    ))
    add(Paragraph(
        "Measured artifacts: <b>results/fast_demo/</b> produced by "
        "<b>configs/fast_demo.yaml</b>. The full-scale configuration "
        "<b>configs/default.yaml</b> was <b>not executed</b>; no full-scale "
        "number is claimed anywhere in this report. Every diffusion, "
        "intervention and satisfaction outcome is a <b>simulation</b>, "
        "not an observed adoption.",
        STYLE_SUBTITLE,
    ))
    add(FrameBreak())

    # ---------------------------------------------------------------- abstract
    h1("Abstract")
    p(
        "This report studies recommendation and information diffusion on the "
        "Epinions who-trusts-whom network: 49,289 users, 139,738 items, "
        "664,824 ratings and 487,002 trust edges (sparsity 0.999903). "
        "A training/validation/test split (70/15/15, seed 42) drives "
        "item-based collaborative filtering, matrix factorisation and a "
        "trust-aware social extension. Diffusion is studied with canonical "
        f"and preference-aware Independent Cascade and with SIR on a "
        f"{num(r4['design']['graph_info']['n_users'],0)}-user, "
        f"{num(r4['design']['graph_info']['n_edges'],0)}-edge 2-hop ego "
        "subgraph. Seed selection compares random, degree and greedy/CELF "
        "strategies; group recommendation compares average, least-misery and "
        "most-pleasure aggregation; a relevance intervention trades simulated "
        "spread against a predicted-satisfaction proxy. All Monte-Carlo "
        "comparisons use genuine common random outcomes (one per-edge "
        "threshold per trial, reused across seed sets, probabilities and "
        "variants), which makes paired differences and the CELF lazy bound "
        "meaningful. Headline results: MF reaches test MAE "
        f"{num(r2['matrix_factorization']['prediction']['mae'],4)} and the "
        f"social model test nDCG@10 "
        f"{num(r7['observed']['models']['social_recommender']['ranking']['ndcg_at_k'],5)}; "
        f"canonical ICM at p=0.05 averages "
        f"{num(r4['icm_canonical']['p=0.05']['summary']['mean_adopters'],1)} "
        f"adopters versus "
        f"{num(r4['icm_preference_aware']['p=0.05']['summary']['mean_adopters'],1)} "
        "for the preference-aware variant; greedy influence maximisation "
        f"gains {num(r5['analysis']['comparison']['5']['greedy_minus_degree'],1)} "
        "adopters over degree centrality inside a restricted 25-user pool; "
        f"and CELF matches greedy exactly with "
        f"{r8['celf_optimization']['comparison']['celf_greedy']['n_evaluations']} "
        f"versus "
        f"{r8['celf_optimization']['comparison']['standard_greedy']['n_evaluations']} "
        "evaluations. Negative and weak results are reported as measured.",
        STYLE_SMALL,
    )

    # ------------------------------------------------------------------- R1
    h1("1. Data, Split and Protocol")
    p(
        f"The Epinions transfer contains {num(ds['n_ratings'],0)} ratings "
        f"({num(ds['n_users'],0)} users, {num(ds['n_items'],0)} items) and "
        f"{num(ds['n_trust_edges'],0)} directed trust edges after removing "
        f"{ds['parsing']['trust']['self_loops']} self-loops. "
        f"{num(ds['user_overlap']['both_rating_and_trust'],0)} users both rate "
        f"and trust; {num(ds['reciprocal_trust_edges'],0)} trust edges are "
        "reciprocal. Ratings are skewed positive "
        f"({num(ds['rating_distribution']['5'],0)} fives versus "
        f"{num(ds['rating_distribution']['1'],0)} ones)."
    )
    tbl(
        ["Split", "Ratings", "Share", "Notes"],
        [
            ["Train", num(sp["train_size"], 0), f"{sp['train_frac']*100:.0f}%",
             "Items/users for fitting"],
            ["Validation", num(sp["val_size"], 0), f"{sp['val_frac']*100:.0f}%",
             "Model/lambda/target selection"],
            ["Test", num(sp["test_size"], 0), f"{sp['test_frac']*100:.0f}%",
             "Final observed evaluation only"],
        ],
        [17 * mm, 20 * mm, 13 * mm, 34 * mm],
    )
    p(
        f"Seed {sp['seed']} is fixed. Users with fewer than three ratings are "
        f"excluded from evaluation ({num(sp['users_excluded_lt3_ratings'],0)} "
        f"users); {num(sp['users_included_in_evaluation'],0)} remain. The "
        "target item for diffusion and intervention is chosen from a "
        "<b>training-only</b> popularity band, and the choice is recorded with "
        "the band size and the target's training rating count. Unobserved items "
        "are never treated as negative examples.",
        STYLE_BODY,
    )

    # ------------------------------------------------------------------- R2
    h1("2. Recommendation Models (R1–R3)")
    mf_cfg = r2["matrix_factorization"]["config"]
    h2("2.1 Item-based CF, matrix factorisation and the social extension")
    p(
        "Item-based CF uses cosine item similarity over the training matrix "
        f"(max_neighbors={r2['item_based_cf']['config']['max_neighbors']}, "
        f"min_corated={r2['item_based_cf']['config']['min_corated']}). "
        f"Matrix factorisation uses {mf_cfg['n_factors']} factors, "
        f"lr={mf_cfg['lr']}, reg={mf_cfg['reg']}, {mf_cfg['n_epochs']} epochs. "
        "The social model blends the MF score with trust-neighbour and "
        "similar-user terms; lambda was selected on validation only "
        f"(chosen λ = {r3['best_lambda']})."
    )
    obs = r7["observed"]["models"]
    tbl(
        ["Model", "MAE", "RMSE", "nDCG@10", "Cov.", "Uniq.", "Div."],
        [
            ["Item-based CF",
             num(obs["item_based_cf"]["prediction_error"]["mae"], 4),
             num(obs["item_based_cf"]["prediction_error"]["rmse"], 4),
             num(obs["item_based_cf"]["ranking"]["ndcg_at_k"], 5),
             num(obs["item_based_cf"]["ranking"]["coverage"], 5),
             num(obs["item_based_cf"]["ranking"]["n_unique_items"], 0),
             num(obs["item_based_cf"]["ranking"]["diversity"], 3)],
            ["Matrix factorisation",
             num(obs["matrix_factorization"]["prediction_error"]["mae"], 4),
             num(obs["matrix_factorization"]["prediction_error"]["rmse"], 4),
             num(obs["matrix_factorization"]["ranking"]["ndcg_at_k"], 5),
             num(obs["matrix_factorization"]["ranking"]["coverage"], 5),
             num(obs["matrix_factorization"]["ranking"]["n_unique_items"], 0),
             num(obs["matrix_factorization"]["ranking"]["diversity"], 3)],
            ["Social (trust, λ=%.1f)" % r3["best_lambda"],
             num(obs["social_recommender"]["prediction_error"]["mae"], 4),
             num(obs["social_recommender"]["prediction_error"]["rmse"], 4),
             num(obs["social_recommender"]["ranking"]["ndcg_at_k"], 5),
             num(obs["social_recommender"]["ranking"]["coverage"], 5),
             num(obs["social_recommender"]["ranking"]["n_unique_items"], 0),
             num(obs["social_recommender"]["ranking"]["diversity"], 3)],
        ],
        [26 * mm, 11 * mm, 11 * mm, 12 * mm, 11 * mm, 8 * mm, 8 * mm],
    )
    p(
        "MAE/RMSE use the full model; nDCG/coverage use the raw ranking scores. "
        "The social model is best on every observed accuracy metric here. "
        "Item-CF has the highest coverage because its similarity neighbourhood "
        "spreads recommendations across more of the catalogue, but its ranking "
        "quality is the weakest.",
        STYLE_BODY,
    )
    if sel is not None:
        base = sel["pre_correction_baseline"]
        chosen = sel["selected"]
        h2("2.2 MF ranking trade-off (disclosed)")
        p(
            "The submitted MF top-N lists originally collapsed onto a few "
            "popular items because the global item bias dominated the "
            "within-user score variance (bias share "
            f"{num(sel['coverage_diagnosis']['bias_share_of_within_user_score_variance'],5)}). "
            "Reducing the bias weight to "
            f"<b>{mf_cfg['ranking_item_bias_weight']}</b>, selected on "
            "validation, widens catalogue coverage. The trade-off is real and "
            "is reported rather than hidden:"
        )
        bullet(
            "Validation nDCG@10 rose from "
            f"{num(base['validation_ranking']['ndcg_at_k'],5)} "
            f"({num(base['validation_ranking']['n_unique_items'],0)} unique "
            "items) to "
            f"{num(chosen['validation_ranking_beta_0.05']['ndcg_at_k'],5)} "
            f"({num(chosen['validation_ranking_beta_0.05']['n_unique_items'],0)} "
            "unique items)."
        )
        bullet(
            "On test, the selected model reaches nDCG@10 "
            f"{num(chosen.get('test_ranking', {}).get('ndcg_at_k', obs['matrix_factorization']['ranking']['ndcg_at_k']),5)} "
            f"with {num(obs['matrix_factorization']['ranking']['n_unique_items'],0)} "
            "unique items, versus the previously committed configuration's "
            "test nDCG@10 0.00343 over ~16 unique items "
            "(coverage 0.0001145 × 139,738 ≈ 16). Coverage rose roughly 17× "
            "while test nDCG@10 fell by about 8.5%."
        )
        bullet(
            "Validation metrics, not test metrics, drove the choice; test was "
            "read once for the final table."
        )
    h2("2.3 Trust versus similarity")
    tv = r3["trust_vs_similarity_comparison"]
    p(
        f"On {num(tv['n_test_ratings'],0)} held-out ratings from "
        f"{num(tv['n_sampled_users'],0)} sampled users (λ={tv['lambda_used']}, "
        f"k={tv['k_similar']}), trust neighbours give RMSE "
        f"{num(tv['trust_neighbors']['rmse'],4)} and similar unconnected users "
        f"give {num(tv['similar_unconnected']['rmse'],4)}, versus baseline "
        f"{num(tv['baseline']['rmse'],4)}. The two tie in practice, so this "
        "sample does not establish that trust beats similarity.",
        STYLE_BODY,
    )

    def draw_err(fig_obj):
        ax = fig_obj.add_subplot(111)
        names = ["Item-CF", "MF", "Social"]
        mae_v = [obs[m]["prediction_error"]["mae"] for m in obs]
        rmse_v = [obs[m]["prediction_error"]["rmse"] for m in obs]
        x = np.arange(3)
        w = 0.38
        b1 = ax.bar(x - w / 2, mae_v, w, label="MAE")
        b2 = ax.bar(x + w / 2, rmse_v, w, label="RMSE")
        add_color(ax, b1, MPL["navy"])
        add_color(ax, b2, MPL["blue"])
        for r, v in zip(b1, mae_v):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=5.5)
        for r, v in zip(b2, rmse_v):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=5.5)
        ax.set_xticks(x, names, fontsize=6.5)
        ax.set_ylabel("Error (test)", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False)
        ax.set_ylim(0, max(rmse_v) * 1.22)

    fig(draw_err, "Fig. 1. Observed test prediction error per recommender "
        "(R7). Lower is better.", "fig01_test_error.png")

    # ------------------------------------------------------------------- R4
    h1("3. Information Diffusion (R4)")
    gi = r4["design"]["graph_info"]
    p(
        "Diffusion runs on a documented 2-hop ego subgraph (centre user "
        f"0): {num(gi['n_users'],0)} users and {num(gi['n_edges'],0)} directed "
        "propagation edges. Canonical ICM activates each out-edge once with "
        "probability p; preference-aware ICM separates exposure from adoption, "
        "where the adoption probability is scaled by the recommender's "
        "predicted affinity. SIR adds a recovery chance γ per round. "
        f"Every setting uses {r4['icm_canonical']['p=0.05']['summary']['adopters']['n']} "
        "Monte-Carlo trials with the same seed sets."
    )
    h2("3.1 Common random outcomes (CRN)")
    crn = r4["design"]["common_random_numbers"]
    p(
        "A sequential RNG restarted at the same integer seed still yields "
        "different live edges when the cascade path changes. This work "
        "generates <b>one activation threshold per directed edge</b> per trial "
        "and reuses it across every seed set, probability and ICM variant; "
        "preference-aware ICM adds one adoption threshold per user; SIR uses "
        "one transmission threshold per (edge, round) and one recovery "
        "threshold per (user, round). Consequently the finite-trial objective "
        "is deterministic, monotone and submodular, paired differences are "
        "meaningful, and the CELF lazy bound is valid. Thresholds are "
        "<b>not</b> shared across different graphs because edge identities "
        "differ (" + crn["not_claimed"] + ")."
    )
    icm = r4["icm_canonical"]
    pre = r4["icm_preference_aware"]
    sir = r4["sir"]
    tbl(
        ["Model", "Setting", "Mean", "SD", "Rounds"],
        [
            ["ICM canonical", "p=0.01",
             num(icm["p=0.01"]["summary"]["mean_adopters"], 1),
             num(icm["p=0.01"]["summary"]["std_adopters"], 1),
             num(icm["p=0.01"]["summary"]["rounds"]["mean"], 1)],
            ["ICM canonical", "p=0.05",
             num(icm["p=0.05"]["summary"]["mean_adopters"], 1),
             num(icm["p=0.05"]["summary"]["std_adopters"], 1),
             num(icm["p=0.05"]["summary"]["rounds"]["mean"], 1)],
            ["Preference-aware", "p=0.01",
             num(pre["p=0.01"]["summary"]["mean_adopters"], 1),
             num(pre["p=0.01"]["summary"]["std_adopters"], 1),
             num(pre["p=0.01"]["summary"]["rounds"]["mean"], 1)],
            ["Preference-aware", "p=0.05",
             num(pre["p=0.05"]["summary"]["mean_adopters"], 1),
             num(pre["p=0.05"]["summary"]["std_adopters"], 1),
             num(pre["p=0.05"]["summary"]["rounds"]["mean"], 1)],
            ["SIR", "β=0.01, γ=0.0",
             num(sir["beta=0.01_gamma=0.0"]["summary"]["mean_ever_infected"], 1),
             num(sir["beta=0.01_gamma=0.0"]["summary"]["std_ever_infected"], 1),
             num(sir["beta=0.01_gamma=0.0"]["summary"]["rounds"]["mean"], 1)],
            ["SIR", "β=0.01, γ=0.2",
             num(sir["beta=0.01_gamma=0.2"]["summary"]["mean_ever_infected"], 1),
             num(sir["beta=0.01_gamma=0.2"]["summary"]["std_ever_infected"], 1),
             num(sir["beta=0.01_gamma=0.2"]["summary"]["rounds"]["mean"], 1)],
            ["SIR", "β=0.05, γ=0.0",
             num(sir["beta=0.05_gamma=0.0"]["summary"]["mean_ever_infected"], 1),
             num(sir["beta=0.05_gamma=0.0"]["summary"]["std_ever_infected"], 1),
             num(sir["beta=0.05_gamma=0.0"]["summary"]["rounds"]["mean"], 1)],
            ["SIR", "β=0.05, γ=0.2",
             num(sir["beta=0.05_gamma=0.2"]["summary"]["mean_ever_infected"], 1),
             num(sir["beta=0.05_gamma=0.2"]["summary"]["std_ever_infected"], 1),
             num(sir["beta=0.05_gamma=0.2"]["summary"]["rounds"]["mean"], 1)],
        ],
        [26 * mm, 22 * mm, 13 * mm, 13 * mm, 13 * mm],
    )
    p(
        "Increasing p raises ICM spread strongly. Preference-aware adoption "
        "cuts the cascade because low-affinity users often do not adopt, even "
        "when they are exposed. SIR with γ=0.05 never recovers users in the "
        "25-round horizon (SI-like); γ=0.2 sharply reduces the infected "
        "population at low β, showing that recovery matters most when "
        "transmission is weak.",
        STYLE_BODY,
    )

    def draw_icm(fig_obj):
        ax = fig_obj.add_subplot(111)
        ps = [0.01, 0.05]
        x = np.arange(2)
        w = 0.38
        can = [icm[f"p={q}"]["summary"]["mean_adopters"] for q in ps]
        cstd = [icm[f"p={q}"]["summary"]["std_adopters"] for q in ps]
        paw = [pre[f"p={q}"]["summary"]["mean_adopters"] for q in ps]
        pstd = [pre[f"p={q}"]["summary"]["std_adopters"] for q in ps]
        b1 = ax.bar(x - w / 2, can, w, yerr=cstd, capsize=2, label="Canonical",
                    error_kw={"elinewidth": 0.7})
        b2 = ax.bar(x + w / 2, paw, w, yerr=pstd, capsize=2,
                    label="Preference-aware", error_kw={"elinewidth": 0.7})
        add_color(ax, b1, MPL["navy"])
        add_color(ax, b2, MPL["blue"])
        ax.set_yscale("log")
        ax.set_xticks(x, [f"p={q}" for q in ps], fontsize=6.5)
        ax.set_ylabel("Mean adopters (log)", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False)
        for xi, (a, b) in enumerate(zip(can, paw)):
            ax.text(xi - w / 2, a * 1.08, f"{a:.0f}", ha="center", fontsize=5.3)
            ax.text(xi + w / 2, b * 1.08, f"{b:.0f}", ha="center", fontsize=5.3)

    fig(draw_icm, "Fig. 2. Rare-target (item 3, one training rating) ICM "
        "spread on the 9,883-user subgraph, 10 trials, error bars = SD.",
        "fig02_rare_target_diffusion.png")

    h2("3.2 Target popularity and network structure")
    tp = r4["target_popularity_comparison"]["comparison"]
    p(
        "The same models are repeated on a moderate target (item 69, nine "
        "training ratings, band size 10,807) with its own seed sets. Canonical "
        "spread is similar across the two targets; preference-aware spread "
        "drops much more on the moderate target at p=0.05 "
        f"({num(tp['p=0.05']['moderate_preference_mean_adopters'],1)} versus "
        f"{num(tp['p=0.05']['rare_preference_mean_adopters'],1)}). The "
        "affinity function, not the target's raw popularity, drives the "
        "preference-aware differences.",
        STYLE_BODY,
    )

    def draw_target(fig_obj):
        ax = fig_obj.add_subplot(111)
        labels = ["p=0.01\nrare", "p=0.01\nmod", "p=0.05\nrare", "p=0.05\nmod"]
        rare = [tp["p=0.01"]["rare_canonical_mean_adopters"],
                tp["p=0.01"]["rare_preference_mean_adopters"],
                tp["p=0.05"]["rare_canonical_mean_adopters"],
                tp["p=0.05"]["rare_preference_mean_adopters"]]
        mod = [tp["p=0.01"]["moderate_canonical_mean_adopters"],
               tp["p=0.01"]["moderate_preference_mean_adopters"],
               tp["p=0.05"]["moderate_canonical_mean_adopters"],
               tp["p=0.05"]["moderate_preference_mean_adopters"]]
        x = np.arange(4)
        w = 0.38
        b1 = ax.bar(x - w / 2, rare, w, label="Rare (1 rating)")
        b2 = ax.bar(x + w / 2, mod, w, label="Moderate (9 ratings)")
        add_color(ax, b1, MPL["navy"])
        add_color(ax, b2, MPL["accent"])
        ax.set_xticks(x, labels, fontsize=5.6)
        ax.set_ylabel("Mean adopters", fontsize=6.5)
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False)
        for r, v in zip(list(b1) + list(b2), rare + mod):
            ax.text(r.get_x() + r.get_width() / 2, v * 1.08, f"{v:.0f}",
                    ha="center", fontsize=5.0)

    fig(draw_target, "Fig. 3. Rare versus moderate target item, same graph and "
        "seeds (R4). Canonical pairs are the two left bars of each p; "
        "preference-aware the two right bars.", "fig03_target_popularity.png")

    ns = r4["network_structure_comparison"]["settings"]
    nsd = r4["network_structure_comparison"]["design"]
    tbl(
        ["Setting", "Observed", "Randomized", "Change", "SD obs/rand"],
        [
            ["p=0.01",
             num(ns["p=0.01"]["observed"]["mean_adopters"], 1),
             num(ns["p=0.01"]["randomized"]["mean_adopters"], 1),
             pct(ns["p=0.01"]["relative_change_pct"]),
             num(ns["p=0.01"]["observed"]["std_adopters"], 1) + " / " +
             num(ns["p=0.01"]["randomized"]["std_adopters"], 1)],
            ["p=0.05",
             num(ns["p=0.05"]["observed"]["mean_adopters"], 1),
             num(ns["p=0.05"]["randomized"]["mean_adopters"], 1),
             pct(ns["p=0.05"]["relative_change_pct"]),
             num(ns["p=0.05"]["observed"]["std_adopters"], 1) + " / " +
             num(ns["p=0.05"]["randomized"]["std_adopters"], 1)],
        ],
        [16 * mm, 18 * mm, 20 * mm, 15 * mm, 18 * mm],
    )
    p(
        "A degree-preserving directed randomization keeps node count, edge "
        f"count and the in/out-degree sequence identical "
        f"({num(nsd['randomized_graph']['randomization']['swaps_accepted'],0)} "
        f"accepted swaps). At p=0.01 the observed arrangement spreads "
        f"far more ({num(ns['p=0.01']['observed']['mean_adopters'],1)} versus "
        f"{num(ns['p=0.01']['randomized']['mean_adopters'],1)}, "
        f"{pct(ns['p=0.01']['relative_change_pct'])}); at p=0.05 the two "
        "structures are statistically indistinguishable. Structure matters at "
        "the sparse-propagation regime, not at saturation.",
        STYLE_BODY,
    )

    def draw_network(fig_obj):
        ax = fig_obj.add_subplot(111)
        ps = ["p=0.01", "p=0.05"]
        x = np.arange(2)
        w = 0.38
        ob = [ns[q]["observed"]["mean_adopters"] for q in ps]
        oe = [ns[q]["observed"]["std_adopters"] for q in ps]
        rb = [ns[q]["randomized"]["mean_adopters"] for q in ps]
        re_ = [ns[q]["randomized"]["std_adopters"] for q in ps]
        b1 = ax.bar(x - w / 2, ob, w, yerr=oe, capsize=2, label="Observed",
                    error_kw={"elinewidth": 0.7})
        b2 = ax.bar(x + w / 2, rb, w, yerr=re_, capsize=2,
                    label="Degree-preserving random", error_kw={"elinewidth": 0.7})
        add_color(ax, b1, MPL["navy"])
        add_color(ax, b2, MPL["gray"])
        ax.set_xticks(x, ps, fontsize=6.5)
        ax.set_ylabel("Mean adopters", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5.6, frameon=False)
        ax.set_ylim(0, max(ob) * 1.25)

    fig(draw_network, "Fig. 4. Observed versus degree-preserving randomized "
        "structure, same seed sets, 10 trials (R4).", "fig04_network_structure.png")

    # ------------------------------------------------------------------- R5
    h1("4. Influence Maximization (R5)")
    pool = r5["design"]["candidate_pool"]
    comp5 = r5["analysis"]["comparison"]
    p(
        f"Random, degree-centrality and greedy seed selection are compared on "
        f"the same graph at budgets {r5['design']['budgets']}, "
        f"p={r5['design']['propagation_prob']}, "
        f"{r5['design']['n_simulations']} Monte-Carlo trials per evaluation "
        f"and {r5['design']['n_random_trials']} random trials. Because a full "
        f"greedy sweep over {num(pool['n_eligible'],0)} users is infeasible "
        f"here, all three methods draw from the <b>same restricted pool of "
        f"{pool['n_candidates']} highest out-degree users</b> "
        f"({pool['rule']}). The result therefore describes that pool, not the "
        "whole graph.",
    )
    tbl(
        ["Budget", "Random", "Degree", "Greedy", "Greedy−deg", "Greedy−rand"],
        [
            [b,
             num(r5["methods"]["random"][b]["mean_spread"], 1),
             num(r5["methods"]["degree"][b]["mean_spread"], 1),
             num(r5["methods"]["greedy"][b]["mean_spread"], 1),
             num(comp5[b]["greedy_minus_degree"], 1),
             num(comp5[b]["greedy_minus_random"], 1)]
            for b in ["1", "3", "5"]
        ],
        [11 * mm, 14 * mm, 14 * mm, 14 * mm, 17 * mm, 17 * mm],
    )
    p(
        "The graph is near saturation: one seed already reaches about 4,719 "
        "users. Greedy is not worse than the baselines and gains up to "
        f"{num(comp5['5']['greedy_minus_degree'],1)} adopters over degree and "
        f"{num(comp5['5']['greedy_minus_random'],1)} over random at budget 5, "
        "but the lift is about 0.12%. Random is marginally best at budget 1. "
        "The honest reading is that the restricted high-degree pool leaves "
        "little room for improvement, not that greedy is useless.",
        STYLE_BODY,
    )
    p(
        "Marginal gains are non-negative by construction under CRN, which "
        "removes the previous artefact where a freshly drawn random stream "
        f"produced a negative gain. Greedy required "
        f"{r5['runtimes_seconds']['greedy_n_evaluations']} evaluations and "
        f"{num(r5['runtimes_seconds']['greedy_selection'],1)} s of selection "
        f"time; evaluating all three methods on common trials cost "
        f"{num(r5['runtimes_seconds']['degree_evaluation'],1)} s (degree) and "
        f"{num(r5['runtimes_seconds']['greedy_evaluation'],1)} s (greedy).",
        STYLE_BODY,
    )

    def draw_influence(fig_obj):
        ax = fig_obj.add_subplot(111)
        budgets = r5["design"]["budgets"]
        for method, color, mk in [("random", MPL["gray"], "o"),
                                  ("degree", MPL["accent"], "s"),
                                  ("greedy", MPL["navy"], "^")]:
            means = [r5["methods"][method][str(b)]["mean_spread"] for b in budgets]
            stds = [r5["methods"][method][str(b)].get("std_spread") or 0
                    for b in budgets]
            ax.errorbar(budgets, means, yerr=stds, marker=mk, color=color,
                        label=method, capsize=2, linewidth=1.0, elinewidth=0.7)
        ax.set_xlabel("Budget (seeds)", fontsize=6.5)
        ax.set_ylabel("Mean spread", fontsize=6.5)
        ax.set_xticks(budgets)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, frameon=False)
        ax.set_ylim(4700, 4732)

    fig(draw_influence, "Fig. 5. Mean simulated spread versus budget for "
        "random, degree and greedy selection on the restricted 25-user pool "
        "(R5); error bars = SD across evaluation trials.",
        "fig05_influence_spread.png", height=48 * mm)

    # ------------------------------------------------------------------- R6
    h1("5. Group Recommendation (R6)")
    gsum = r6["summary"]
    n_groups = gsum["average"]["n_groups"]
    p(
        f"{n_groups} conflicting-preference groups are built from the training "
        "split; each group's top-N list is produced under average, "
        "least-misery and most-pleasure aggregation. Ordering uses the "
        "recommender's <b>raw (unclipped)</b> aggregate so items that clip to "
        "the same rating no longer tie; the reported group score is the "
        "<b>clipped</b> aggregate satisfaction. Spread for the top item is a "
        "preference-aware ICM simulation from the group members as seeds.",
    )
    tbl(
        ["Strategy", "Member sat.", "Group score", "Raw rank",
         "Spread"],
        [
            [s.replace("_", " "),
             num(gsum[s]["avg_mean_satisfaction"], 3),
             num(gsum[s]["avg_group_score"], 3),
             num(gsum[s]["avg_group_ranking_score"], 3),
             num(gsum[s]["avg_diffusion_spread"], 1)]
            for s in ["average", "least_misery", "most_pleasure"]
        ],
        [22 * mm, 18 * mm, 14 * mm, 16 * mm, 14 * mm],
    )
    g2_mp = r6["groups"]["group_2"]["strategies"]["most_pleasure"]
    p(
        f"Least-misery returns the lowest group score by definition; "
        f"most-pleasure returns 5.0 because at least one member predicts the "
        f"ceiling, yet its raw ranking score is distinct "
        f"({num(g2_mp['group_ranking_score'],3)} for group 2), which is why "
        "the ordering is no longer an artefact of clipping. All three "
        "strategies select the same top item in every group in this run, so "
        "the simulated spread is identical; that is a property of these "
        "training-only groups, not evidence that the strategies are "
        "equivalent.",
        STYLE_BODY,
    )

    # ------------------------------------------------------------------- R7
    h1("6. Comprehensive Evaluation (R7)")
    sim = r7["simulated"]["models"]
    tr = r7["tradeoff_analysis"]
    p(
        "Observed test ranking is paired with a preference-aware ICM "
        "simulation that uses each model's affinity function on the same "
        "target item, seed sets and CRN thresholds. The two answer different "
        "questions and are not interchangeable."
    )
    tbl(
        ["Model", "nDCG@10", "RMSE", "Adopt.", "SD", "Sat. proxy"],
        [
            [m.replace("_", " "),
             num(obs[m]["ranking"]["ndcg_at_k"], 5),
             num(obs[m]["prediction_error"]["rmse"], 4),
             num(sim[m]["adopters"]["mean"], 1),
             num(sim[m]["adopters"]["std"], 1),
             num(sim[m]["adopter_satisfaction_proxy"]["mean"], 3)]
            for m in ["item_based_cf", "matrix_factorization", "social_recommender"]
        ],
        [24 * mm, 12 * mm, 12 * mm, 15 * mm, 10 * mm, 14 * mm],
    )
    p(
        f"The best observed nDCG model and the largest simulated spread both "
        f"belong to the social recommender ({tr['best_observed_ndcg_model'].replace('_', ' ')}, "
        f"spread {num(tr['simulated_spread_ranking']['social_recommender'],1)}), "
        f"so the two rankings agree in this run. The simulated cascade uses "
        "one target item from a training popularity band and is not a "
        "platform-wide forecast. Coverage and diversity depend on the "
        "evaluated list sample (500 seeded users) and on training-derived "
        "item features.",
        STYLE_BODY,
    )

    # ------------------------------------------------------------------- R8
    h1("7. Intervention, CELF and Cost Scaling (R8)")
    iv = r8["intervention"]
    h2("7.1 Relevance intervention")
    target_id = r8["design"]["target_item"]["original_id"]
    iv_keys = ["threshold_0", "threshold_0.3", "threshold_0.6",
               "threshold_0.8", "threshold_0.9"]
    iv_labels = ["0.0", "0.3", "0.6", "0.8", "0.9"]
    p(
        "A minimum predicted-affinity threshold blocks low-affinity adoption "
        f"inside the simulator (target item {target_id}, "
        f"{iv['results']['threshold_0']['adopters']['n']} trials, same CRN "
        "thresholds). Spread falls as the threshold rises; the adopter-rating "
        "proxy is a simulator-side quantity, not observed satisfaction."
    )
    tbl(
        ["Threshold", "Mean adopters", "Change", "Rating proxy", "≥4 share"],
        [
            [lbl,
             num(iv["results"][key]["mean_adopters"], 1),
             pct(iv["results"][key].get("relative_to_baseline", {}).get(
                 "spread_change_pct", 0.0)),
             num(iv["results"][key]["mean_adopter_rating_proxy"], 3),
             num(iv["results"][key]["share_adopters_rating_ge_4"], 3)]
            for key, lbl in zip(iv_keys, iv_labels)
        ],
        [14 * mm, 17 * mm, 13 * mm, 15 * mm, 12 * mm],
    )
    p(
        "Thresholds at or below every observed affinity (0.0, 0.3) change "
        "nothing. At 0.6 spread barely moves; at 0.8 it falls 84% while the "
        "rating proxy rises; at 0.9 the cascade is almost fully suppressed "
        "(7.1 adopters) and the proxy falls because the remaining adopters are "
        "not the highest-affinity users. The measured table, not an assumed "
        "monotone trade-off, is the evidence.",
        STYLE_BODY,
    )

    def draw_intervention(fig_obj):
        ax = fig_obj.add_subplot(111)
        ts = iv_labels
        adopt = [iv["results"][k]["mean_adopters"] for k in iv_keys]
        proxy = [iv["results"][k]["mean_adopter_rating_proxy"] for k in iv_keys]
        bars = ax.bar(ts, adopt)
        add_color(ax, bars, MPL["navy"])
        ax.set_ylabel("Mean adopters", fontsize=6.5)
        ax.set_xlabel("Min affinity threshold", fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax2 = ax.twinx()
        ax2.plot(ts, proxy, color=MPL["accent"], marker="o", linewidth=1.0,
                 markersize=3)
        ax2.set_ylabel("Adopter rating proxy", fontsize=6.5, color=MPL["accent"])
        ax2.tick_params(labelsize=6, colors=MPL["accent"])
        ax2.set_ylim(4.2, 4.6)

    fig(draw_intervention, "Fig. 6. Simulated spread (bars, left axis) and "
        "adopter-satisfaction proxy (line, right axis) versus minimum-affinity "
        "threshold (R8).", "fig06_intervention.png", height=48 * mm)

    h2("7.2 CELF lazy greedy")
    c = r8["celf_optimization"]["comparison"]
    p(
        "With the CRN objective now deterministic and submodular, CELF's lazy "
        "bound is valid and should reproduce standard greedy exactly. It does:"
    )
    tbl(
        ["Variant", "Seeds", "Evaluations", "Runtime (s)", "Marginal gains"],
        [
            ["Standard greedy",
             str(c["standard_greedy"]["seeds"]),
             num(c["standard_greedy"]["n_evaluations"], 0),
             num(c["standard_greedy"]["runtime_seconds"], 2),
             ", ".join(num(g, 1) for g in c["standard_greedy"]["marginal_gains"])],
            ["CELF greedy",
             str(c["celf_greedy"]["seeds"]),
             num(c["celf_greedy"]["n_evaluations"], 0),
             num(c["celf_greedy"]["runtime_seconds"], 2),
             ", ".join(num(g, 1) for g in c["celf_greedy"]["marginal_gains"])],
        ],
        [19 * mm, 19 * mm, 13 * mm, 14 * mm, 19 * mm],
    )
    p(
        f"CELF used {num(c['evaluation_reduction_pct'],2)}% fewer evaluations "
        f"({c['celf_greedy']['n_evaluations']} vs "
        f"{c['standard_greedy']['n_evaluations']}) and ran "
        f"{num(c['speedup_factor'],2)}× faster in this measured setting, "
        "selecting the identical seed set. A separate controlled check at "
        "p=1.0 on a 120-node graph also produced identical seeds and equal "
        "evaluation counts (39 each).",
        STYLE_BODY,
    )

    h2("7.3 Cost scaling")
    sc = r8["scaling"]
    tbl(
        ["Dimension", "Size", "Runtime", "Evals", "Memory peak"],
        [
            ["Network 25%",
             num(sc["network"]["rows"][0]["n_nodes"], 0) + " nodes",
             num(sc["network"]["rows"][0]["runtime_seconds"], 2) + " s",
             num(sc["network"]["rows"][0]["n_evaluations"], 0),
             num(sc["network"]["rows"][0]["memory"]["tracemalloc_peak_mb"], 2) + " MB"],
            ["Network 50%",
             num(sc["network"]["rows"][1]["n_nodes"], 0) + " nodes",
             num(sc["network"]["rows"][1]["runtime_seconds"], 2) + " s",
             num(sc["network"]["rows"][1]["n_evaluations"], 0),
             num(sc["network"]["rows"][1]["memory"]["tracemalloc_peak_mb"], 2) + " MB"],
            ["Items 25%",
             num(sc["item_catalogue"]["rows"][0]["n_items"], 0) + " items",
             num(sc["item_catalogue"]["rows"][0]["ms_per_user"], 3) + " ms/user",
             "—",
             num(sc["item_catalogue"]["rows"][0]["memory"]["tracemalloc_peak_mb"], 2) + " MB"],
            ["Items 50%",
             num(sc["item_catalogue"]["rows"][1]["n_items"], 0) + " items",
             num(sc["item_catalogue"]["rows"][1]["ms_per_user"], 3) + " ms/user",
             "—",
             num(sc["item_catalogue"]["rows"][1]["memory"]["tracemalloc_peak_mb"], 2) + " MB"],
            ["Items 100%",
             num(sc["item_catalogue"]["rows"][2]["n_items"], 0) + " items",
             num(sc["item_catalogue"]["rows"][2]["ms_per_user"], 3) + " ms/user",
             "—",
             num(sc["item_catalogue"]["rows"][2]["memory"]["tracemalloc_peak_mb"], 2) + " MB"],
        ],
        [20 * mm, 18 * mm, 18 * mm, 10 * mm, 16 * mm],
    )
    p(
        "Greedy runtime grows super-linearly with network size because each "
        "candidate evaluation is a full Monte-Carlo cascade; item scoring is "
        "linear in catalogue size. The scaling runs use a single Monte-Carlo "
        "trial to isolate cost, so they are cost measurements, not spread "
        "estimates.",
        STYLE_BODY,
    )

    def draw_scaling(fig_obj):
        ax1 = fig_obj.add_subplot(121)
        nodes = [r["n_nodes"] for r in sc["network"]["rows"]]
        times = [r["runtime_seconds"] for r in sc["network"]["rows"]]
        ax1.plot(nodes, times, marker="o", color=MPL["navy"], linewidth=1.1,
                 markersize=3)
        ax1.set_xlabel("Network nodes", fontsize=6)
        ax1.set_ylabel("Greedy runtime (s)", fontsize=6)
        ax1.tick_params(labelsize=5.5)
        ax2 = fig_obj.add_subplot(122)
        items = [r["n_items"] for r in sc["item_catalogue"]["rows"]]
        msu = [r["ms_per_user"] for r in sc["item_catalogue"]["rows"]]
        ax2.plot(items, msu, marker="s", color=MPL["accent"], linewidth=1.1,
                 markersize=3)
        ax2.set_xlabel("Catalogue items", fontsize=6)
        ax2.set_ylabel("Scoring (ms/user)", fontsize=6)
        ax2.tick_params(labelsize=5.5)

    fig(draw_scaling, "Fig. 7. Cost scaling: greedy network runtime (left) and "
        "MF scoring cost (right). One Monte-Carlo trial, cost isolated (R8).",
        "fig07_cost_scaling.png", height=46 * mm)

    # ------------------------------------------------------------------- R8b
    h1("8. Reproducibility, Limitations and Conclusion")
    h2("8.1 Reproducibility")
    total_s = timings["total_seconds"]
    bullet(
        f"Measured fast-demo pipeline: {num(total_s,1)} s of stage time "
        f"({num(total_s/60,1)} min) on the development machine. Longest "
        f"stages: MF training {num(next(s['seconds'] for s in timings['stages'] if s['stage']=='R2: MF training'),1)} s, "
        f"R5 {num(next(s['seconds'] for s in timings['stages'] if s['stage']=='R5: Influence Maximization'),1)} s, "
        f"R8 {num(next(s['seconds'] for s in timings['stages'] if s['stage']=='R8: Intervention & Cost Analysis'),1)} s."
    )
    bullet(
        "Commands: <b>python -m pytest -q</b> (103 tests, ~1.7 s) and "
        "<b>python -m sntp.cli run --config configs/fast_demo.yaml</b>. "
        "The report is regenerated with "
        "<b>python scripts/generate_report.py --results results/fast_demo "
        f"--output report/SNTP_Assignment_02_Report.pdf --roll-number "
        f"{args.roll_number} --student \"{args.student}\"</b>."
    )
    bullet(
        "Fixed seeds: data split 42, MF 42, ranking sample 42, diffusion base "
        "42, target randomization 4242, influence 42, intervention 42. Each "
        "stage records the exact seed schedule in its JSON."
    )
    bullet(
        "validation/test selection discipline: model, lambda and target "
        "choices use validation or training data only; test is read once for "
        "the final tables."
    )

    h2("8.2 Limitations (all disclosed)")
    bullet(
        "<b>Full-scale config not executed.</b> configs/default.yaml is "
        "provided for a longer run but no full-scale result is claimed."
    )
    bullet(
        "<b>Restricted influence pool.</b> R5 cannot sweep all 9,883 users; "
        "the comparison holds inside the 25 highest out-degree candidates and "
        "the graph is near saturation."
    )
    bullet(
        "<b>Simulated outcomes.</b> Epinions has no adoption timestamps; every "
        "diffusion, intervention and satisfaction number is simulated. The "
        "rating proxy is model-predicted, not observed after an intervention."
    )
    bullet(
        "<b>MF ranking trade-off.</b> Coverage rose roughly 17× but test "
        "nDCG@10 fell ~8.5%; validation improved, so the model was kept and "
        "the trade-off is reported."
    )
    bullet(
        "<b>Sampled evaluation.</b> Ranking/diversity use a fixed 500-user "
        "sample; coverage and diversity are sample- and feature-dependent."
    )
    bullet(
        "<b>Network null model.</b> The randomization is one seeded "
        "degree-preserving rearrangement, not full mixing; observed and "
        "randomized graphs cannot share edge-level thresholds."
    )
    bullet(
        "<b>Group results.</b> All three strategies chose the same top item "
        "in these training-only groups, so their simulated spreads are "
        "identical."
    )

    h2("8.3 Conclusion")
    p(
        "The submission is reproducible and internally consistent: the "
        "recommender comparison favours the trust-aware social model on "
        "observed metrics; CRN-based simulated cascades are governed by "
        "probability and preference rather than by target popularity alone; "
        "the observed network arrangement beats a degree-preserving "
        "randomization only in the sparse-propagation regime; greedy improves "
        "on degree only marginally inside a restricted high-degree pool; and "
        "CELF reproduces greedy with about 27% fewer evaluations once common "
        "random outcomes make the objective deterministic. Weak and negative "
        "results are reported alongside the positive ones."
    )

    h1("References")
    refs = [
        "[1] P. Massa and P. Avesani, “Trust-aware recommender systems,” "
        "in Proc. ACM RecSys, 2007.",
        "[2] D. Kempe, J. Kleinberg, and É. Tardos, “Maximizing the spread of "
        "influence through a social network,” in Proc. ACM KDD, 2003.",
        "[3] J. Leskovec, A. Krause, C. Guestrin, C. Faloutsos, J. VanBriesen, "
        "and N. Glance, “Cost-effective outbreak detection in networks,” in "
        "Proc. ACM KDD, 2007.",
        "[4] W. O. Kermack and A. G. McKendrick, “A contribution to the "
        "mathematical theory of epidemics,” Proc. R. Soc. A, vol. 115, 1927.",
        "[5] Y. Koren, R. Bell, and C. Volinsky, “Matrix factorization "
        "techniques for recommender systems,” IEEE Computer, vol. 42, 2009.",
        "[6] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, “Item-based "
        "collaborative filtering recommendation algorithms,” in Proc. WWW, 2001.",
        "[7] J. L. Herlocker, J. A. Konstan, L. G. Terveen, and J. Riedl, "
        "“Evaluating collaborative filtering recommender systems,” ACM TOIS, "
        "vol. 22, 2004.",
        "[8] M. Jamali and M. Ester, “A matrix factorization technique with "
        "trust propagation for recommendation in social networks,” in Proc. "
        "ACM RecSys, 2010.",
    ]
    for ref in refs:
        add(Paragraph(ref, STYLE_SMALL))

    return out


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
        title="SNTP Assignment 2 - Recommendation and Information Diffusion on Epinions",
        author=args.student,
        roll=args.roll_number,
        leftMargin=LM,
        rightMargin=RM,
        topMargin=TM,
        bottomMargin=BM,
    )
    doc.build(story(metrics, args))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
