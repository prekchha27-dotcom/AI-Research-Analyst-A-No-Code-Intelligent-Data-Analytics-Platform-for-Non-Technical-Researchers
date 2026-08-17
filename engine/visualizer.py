"""
Visualization Engine
Creates Plotly figures from DataFrames.
All figures are based on actual computed data — never invented.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Optional


# ── Light-theme chart palette ─────────────────────────────────────────────────
# Rule: ALL text (titles, axis labels, tick labels, legends) must be dark navy
#       on the light paper/plot background so charts match the dashboard theme.
_PALETTE = px.colors.qualitative.Set2

_TEXT_DARK  = "#1a2340"   # dark navy — primary chart text
_TEXT_MID   = "#2c3e50"   # charcoal  — axis tick labels
_ACCENT     = "#2196a6"   # sky blue  — gridlines / zero-line

_THEME = dict(
    paper_bgcolor = "#ffffff",   # white chart card background
    plot_bgcolor  = "#f4fbfd",   # very faint sky tint inside plot area
    font          = dict(family="-apple-system, Segoe UI, sans-serif",
                         size=12, color=_TEXT_DARK),
    margin        = dict(l=48, r=24, t=56, b=48),
    xaxis = dict(
        color          = _TEXT_DARK,
        tickfont       = dict(color=_TEXT_MID, size=11),
        title_font     = dict(color=_TEXT_DARK, size=12),
        gridcolor      = "#daeef3",
        linecolor      = "#b2ebf2",
        zerolinecolor  = _ACCENT,
    ),
    yaxis = dict(
        color          = _TEXT_DARK,
        tickfont       = dict(color=_TEXT_MID, size=11),
        title_font     = dict(color=_TEXT_DARK, size=12),
        gridcolor      = "#daeef3",
        linecolor      = "#b2ebf2",
        zerolinecolor  = _ACCENT,
    ),
    legend = dict(
        bgcolor     = "#ffffff",
        bordercolor = "#b2ebf2",
        borderwidth = 1,
        font        = dict(color=_TEXT_DARK, size=12),
    ),
)


def _apply_theme(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=_TEXT_DARK, family="-apple-system, Segoe UI, sans-serif")),
        **_THEME,
    )
    # Ensure every axis that exists is readable (handles subplots / secondary axes)
    fig.update_xaxes(color=_TEXT_DARK, tickfont_color=_TEXT_MID,
                     gridcolor="#daeef3", linecolor="#b2ebf2")
    fig.update_yaxes(color=_TEXT_DARK, tickfont_color=_TEXT_MID,
                     gridcolor="#daeef3", linecolor="#b2ebf2")
    return fig


# ── Bar Chart ─────────────────────────────────────────────────────────────────

def bar_chart(df: pd.DataFrame, x_col: str, y_col: str,
              color_col: Optional[str] = None, title: str = "") -> go.Figure:
    fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                 color_discrete_sequence=_PALETTE, barmode="group",
                 title=title or f"{y_col} by {x_col}")
    return _apply_theme(fig, title or f"{y_col} by {x_col}")


def frequency_bar(df: pd.DataFrame, column: str) -> go.Figure:
    vc = df[column].value_counts().reset_index()
    vc.columns = [column, "Count"]
    fig = px.bar(vc, x=column, y="Count",
                 color_discrete_sequence=_PALETTE,
                 title=f"Frequency Distribution of {column}")
    return _apply_theme(fig, f"Frequency Distribution of {column}")


# ── Pie / Donut ───────────────────────────────────────────────────────────────

def pie_chart(df: pd.DataFrame, column: str, donut: bool = False) -> go.Figure:
    vc   = df[column].value_counts()
    hole = 0.4 if donut else 0.0
    fig  = go.Figure(go.Pie(labels=vc.index.astype(str), values=vc.values,
                            hole=hole, marker_colors=_PALETTE))
    kind = "Donut" if donut else "Pie"
    return _apply_theme(fig, f"{kind} Chart — {column}")


# ── Histogram ─────────────────────────────────────────────────────────────────

def histogram(df: pd.DataFrame, column: str, bins: int = 20,
              color_col: Optional[str] = None) -> go.Figure:
    fig = px.histogram(df, x=column, nbins=bins, color=color_col,
                       color_discrete_sequence=_PALETTE,
                       marginal="box",
                       title=f"Distribution of {column}")
    return _apply_theme(fig, f"Distribution of {column}")


# ── Box Plot ──────────────────────────────────────────────────────────────────

def box_plot(df: pd.DataFrame, y_col: str, x_col: Optional[str] = None,
             title: str = "") -> go.Figure:
    fig = px.box(df, y=y_col, x=x_col, color=x_col,
                 color_discrete_sequence=_PALETTE,
                 points="outliers",
                 title=title or (f"{y_col} by {x_col}" if x_col else f"Box Plot — {y_col}"))
    return _apply_theme(fig, fig.layout.title.text)


# ── Scatter Plot ──────────────────────────────────────────────────────────────

def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                 color_col: Optional[str] = None, trendline: bool = True) -> go.Figure:
    trend = "ols" if trendline else None
    fig   = px.scatter(df, x=x_col, y=y_col, color=color_col,
                       trendline=trend,
                       color_discrete_sequence=_PALETTE,
                       title=f"Scatter: {x_col} vs {y_col}")
    return _apply_theme(fig, f"Scatter: {x_col} vs {y_col}")


# ── Line Graph ────────────────────────────────────────────────────────────────

def line_chart(df: pd.DataFrame, x_col: str, y_col: str,
               color_col: Optional[str] = None) -> go.Figure:
    fig = px.line(df, x=x_col, y=y_col, color=color_col,
                  color_discrete_sequence=_PALETTE,
                  markers=True,
                  title=f"{y_col} over {x_col}")
    return _apply_theme(fig, f"{y_col} over {x_col}")


# ── Stacked Bar ───────────────────────────────────────────────────────────────

def stacked_bar(df: pd.DataFrame, x_col: str, y_col: str, color_col: str) -> go.Figure:
    fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                 barmode="stack",
                 color_discrete_sequence=_PALETTE,
                 title=f"Stacked: {y_col} by {x_col} and {color_col}")
    return _apply_theme(fig, fig.layout.title.text)


# ── Area Chart ────────────────────────────────────────────────────────────────

def area_chart(df: pd.DataFrame, x_col: str, y_col: str,
               color_col: Optional[str] = None) -> go.Figure:
    fig = px.area(df, x=x_col, y=y_col, color=color_col,
                  color_discrete_sequence=_PALETTE,
                  title=f"Area: {y_col} over {x_col}")
    return _apply_theme(fig, fig.layout.title.text)


# ── Heatmap ───────────────────────────────────────────────────────────────────

def heatmap(df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> go.Figure:
    pivot = df.pivot_table(values=z_col, index=y_col, columns=x_col, aggfunc="mean")
    fig   = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.astype(str), y=pivot.index.astype(str),
        colorscale="RdYlGn", showscale=True,
    ))
    return _apply_theme(fig, f"Heatmap: {z_col} by {x_col} × {y_col}")


def correlation_heatmap(corr_matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale="RdBu", zmin=-1, zmax=1,
        text=corr_matrix.round(2).values,
        texttemplate="%{text}",
        showscale=True,
    ))
    return _apply_theme(fig, "Correlation Heatmap")


# ── Visualization Recommendation Engine ───────────────────────────────────────

CHART_RATINGS = {
    # (x_type, y_type): [(chart_type, stars, when_to_use, why_appropriate)]
    ("categorical", "numerical"): [
        ("bar_chart",     5, "Comparing values across categories.",
         "Ideal for showing how a numerical value differs between groups."),
        ("box_plot",      4, "Showing distribution and outliers per group.",
         "Reveals median, spread, and outliers per category — more informative than a bar chart for variability."),
        ("pie_chart",     2, "Showing proportional composition.",
         "Less accurate for comparison; consider only if categories are few and proportions are the focus."),
    ],
    ("categorical", "categorical"): [
        ("bar_chart",     5, "Showing frequency counts per category.",
         "The most intuitive chart for comparing frequencies of categorical data."),
        ("pie_chart",     3, "Showing proportional distribution.",
         "Suitable when you want to show part-to-whole relationships for a single categorical variable."),
    ],
    ("numerical", "numerical"): [
        ("scatter_plot",  5, "Examining the relationship between two numerical variables.",
         "Directly visualises correlation, clusters, and patterns between two continuous variables."),
        ("line_chart",    3, "Showing trend if one variable is ordered (e.g. index).",
         "Appropriate only if there is a meaningful order to the x-axis."),
    ],
    ("datetime", "numerical"): [
        ("line_chart",    5, "Showing change over time.",
         "Line charts are the standard for time-series data — they clearly show trends and fluctuations."),
        ("area_chart",    4, "Showing cumulative or volume change over time.",
         "Good when magnitude matters alongside trend."),
        ("bar_chart",     3, "Showing discrete time-period comparisons.",
         "Useful when periods are distinct (e.g. months, years) and comparison is more important than trend."),
    ],
    ("numerical", "categorical"): [
        ("histogram",     5, "Showing distribution of one numerical variable.",
         "Histograms reveal the shape, spread, and skewness of the data distribution."),
        ("box_plot",      4, "Comparing distributions across categories.",
         "Shows median, quartiles, and outliers in a compact form."),
    ],
}

STAR_LABELS = {5: "★★★★★ Highly Recommended", 4: "★★★★ Recommended",
               3: "★★★ Moderately Suitable",  2: "★★ Less Suitable",
               1: "★ Not Recommended"}


def recommend_visualization(x_type: str, y_type: str, x_col: str, y_col: str) -> list[dict]:
    """
    Recommend visualizations based on variable types.
    Returns a sorted list of recommendations with explanation.
    """
    # Normalise types
    def _norm(t: str) -> str:
        if t in ("numerical", "numerical_discrete"): return "numerical"
        if t in ("categorical", "boolean", "text"):  return "categorical"
        if t == "datetime":                           return "datetime"
        return "other"

    key1 = (_norm(x_type), _norm(y_type))
    key2 = (_norm(y_type), _norm(x_type))

    recs = CHART_RATINGS.get(key1) or CHART_RATINGS.get(key2) or []
    result = []
    for chart, stars, when, why in recs:
        result.append({
            "chart_type":  chart,
            "stars":       stars,
            "star_label":  STAR_LABELS[stars],
            "when_to_use": when,
            "why_appropriate": why,
            "x_col": x_col, "y_col": y_col,
        })
    if not result:
        result.append({
            "chart_type": "bar_chart", "stars": 3,
            "star_label": STAR_LABELS[3],
            "when_to_use": "General comparison.",
            "why_appropriate": "Default recommendation — inspect your data types for a more tailored suggestion.",
            "x_col": x_col, "y_col": y_col,
        })
    return sorted(result, key=lambda r: -r["stars"])


def build_chart(df: pd.DataFrame, chart_type: str, x_col: str,
                y_col: Optional[str] = None, color_col: Optional[str] = None,
                title: str = "") -> go.Figure:
    """Dispatch to the correct chart-building function."""
    # Guard: deduplicate column names so Plotly never sees repeated names.
    # Keep the first occurrence of each duplicate column.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    dispatch = {
        "bar_chart":    lambda: bar_chart(df, x_col, y_col or x_col, color_col, title),
        "pie_chart":    lambda: pie_chart(df, x_col, donut=False),
        "donut_chart":  lambda: pie_chart(df, x_col, donut=True),
        "histogram":    lambda: histogram(df, x_col),
        "line_chart":   lambda: line_chart(df, x_col, y_col or x_col, color_col),
        "scatter_plot": lambda: scatter_plot(df, x_col, y_col or x_col, color_col),
        "box_plot":     lambda: box_plot(df, y_col or x_col, x_col if y_col else None, title),
        "stacked_bar":  lambda: stacked_bar(df, x_col, y_col or x_col, color_col or x_col),
        "area_chart":   lambda: area_chart(df, x_col, y_col or x_col, color_col),
        "heatmap":      lambda: heatmap(df, x_col, color_col or x_col, y_col or x_col),
        "frequency_bar":lambda: frequency_bar(df, x_col),
    }
    fn = dispatch.get(chart_type)
    if fn is None:
        return frequency_bar(df, x_col)
    try:
        return fn()
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Chart error: {str(e)}", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(color="red", size=13))
        return _apply_theme(fig, "Chart Error")
