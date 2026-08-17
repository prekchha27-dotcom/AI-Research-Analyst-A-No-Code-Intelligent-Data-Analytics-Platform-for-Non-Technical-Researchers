# -*- coding: utf-8 -*-
"""
AI Research Analyst — No-Code Research Data Analytics Platform
Main Streamlit application

Run from the research_analytics/ directory:
    streamlit run main.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import io
import traceback

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Analyst",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS  (Light Yellow + Sky Blue — High Contrast) ────────────────────
# All values hard-coded. No CSS custom properties (Streamlit strips them).
# Palette:
#   Page bg      #fdf6d3  pastel yellow page
#   Surface bg   #ffffff  white cards/tables/inputs
#   Sky header   #1565c0  dark blue — tab headers, expander headers (white text)
#   Sky panel    #dbeeff  pale blue panels
#   Sky border   #5ba4c4  visible medium-blue borders
#   Sky hover    #b3d9f0  hover highlight
#   Text         #0d1b2a  near-black — max contrast on all light surfaces
#   Text mid     #1a3050  dark navy labels
#   Text muted   #2c4a6e  medium navy captions
st.markdown("""
<style>
/* ============================================================
   AI RESEARCH ANALYST - HIGH CONTRAST LIGHT THEME
   All hex values hard-coded. No CSS custom properties.
   Page bg: #fdf6d3  Surface: #ffffff  Primary: #1565c0
   Border: #5ba4c4   Text: #0d1b2a     Panel: #dbeeff
   ============================================================ */

/* 1. HIDE CHROME */
#MainMenu, footer { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none; }

/* 2. PAGE BACKGROUND */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .main .block-container,
section[data-testid="stSidebar"] + div {
    background-color: #fdf6d3 !important;
}
/* Restore yellow on main content vertical blocks only — NOT sidebar children */
[data-testid="stMain"] [data-testid="stVerticalBlock"],
[data-testid="stMain"] [data-testid="stHorizontalBlock"],
[data-testid="stMain"] [data-testid="column"] {
    background-color: #fdf6d3 !important;
}
.main .block-container { padding-top: 1.4rem; }

/* 3. GLOBAL TEXT */
html, body, p, li, span, label, div, [data-testid], [class*="st-"] {
    color: #0d1b2a;
}
h1,h2,h3,h4,h5,h6 { color: #0d1b2a !important; font-weight: 700 !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 { color: #0d1b2a !important; font-weight: 700 !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em { color: #0d1b2a !important; }

/* 4. SIDEBAR — scoped entirely to [data-testid="stSidebar"] */

/* Background on the sidebar container AND every child div/section.
   background-color is used (not shorthand) because the page rule above
   also uses background-color !important, and we need equal specificity
   plus a later position in the sheet to win. */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div,
[data-testid="stSidebar"] > div > div > div,
[data-testid="stSidebar"] section,
[data-testid="stSidebarContent"],
[data-testid="stSidebarContent"] > div,
[data-testid="stSidebarContent"] > div > div {
    background-color: #1a237e !important;
    background-image: none !important;
}

/* Text: all markdown containers inside the sidebar */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] small {
    color: #ffffff !important;
}

/* Navigation radio: hide auto-generated label */
[data-testid="stSidebar"] .stRadio > label { display: none; }

/* Nav item list layout */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 1px;
}

/* Each nav item label */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    color: #e8eaf6 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 9px 16px !important;
    border-radius: 6px !important;
    border-left: 3px solid transparent !important;
    cursor: pointer !important;
    background-color: transparent !important;
    background-image: none !important;
    display: flex !important;
    align-items: center !important;
}

/* Hover state */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    color: #ffffff !important;
    background-color: rgba(255,255,255,0.15) !important;
    border-left-color: #90caf9 !important;
}

/* Active/selected nav item — :has() targets label whose radio is checked */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
    color: #ffffff !important;
    background-color: rgba(255,255,255,0.20) !important;
    border-left: 3px solid #90caf9 !important;
    font-weight: 700 !important;
}

/* Hide the actual radio circle widget inside nav labels */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label input[type="radio"] {
    display: none !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

/* Footer caption */
[data-testid="stSidebar"] .stMarkdown p {
    color: #9fa8da !important;
}

/* 5. BUTTONS */
[data-testid="baseButton-primary"],
button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background-color: #1565c0 !important;
    color: #ffffff !important;
    border: 2px solid #0d47a1 !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
}
[data-testid="baseButton-primary"]:hover, button[kind="primary"]:hover {
    background-color: #0d47a1 !important;
    color: #ffffff !important;
}
[data-testid="baseButton-secondary"],
button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
    background-color: #ffffff !important;
    color: #0d1b2a !important;
    border: 2px solid #1565c0 !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
}
[data-testid="baseButton-secondary"]:hover, button[kind="secondary"]:hover {
    background-color: #dbeeff !important;
    color: #0d47a1 !important;
}
.stButton > button {
    background-color: #ffffff !important;
    color: #0d1b2a !important;
    border: 2px solid #5ba4c4 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    background-color: #dbeeff !important;
    color: #0d1b2a !important;
    border-color: #1565c0 !important;
}

/* 6. TABS */
[data-baseweb="tab-list"] {
    background-color: #1565c0 !important;
    border-radius: 8px 8px 0 0 !important;
    gap: 3px !important;
    padding: 5px !important;
}
[data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.15) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    padding: 7px 16px !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background-color: #ffffff !important;
    color: #0d47a1 !important;
    font-weight: 800 !important;
}
[data-baseweb="tab"]:hover:not([aria-selected="true"]) {
    background-color: rgba(255,255,255,0.30) !important;
    color: #ffffff !important;
}
[data-baseweb="tab-panel"] {
    background-color: #ffffff !important;
    border: 2px solid #5ba4c4 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 18px !important;
}
[data-baseweb="tab-panel"] * { color: #0d1b2a !important; }
[data-baseweb="tab-panel"] button { color: #ffffff !important; }

/* 7. EXPANDERS */
[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 2px solid #5ba4c4 !important;
    border-radius: 8px !important;
    margin-bottom: 8px !important;
}
[data-testid="stExpander"] summary, details > summary {
    background-color: #dbeeff !important;
    color: #0d1b2a !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    border-radius: 6px !important;
    padding: 10px 14px !important;
}
[data-testid="stExpander"] summary:hover, details > summary:hover {
    background-color: #b3d9f0 !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary strong { color: #0d1b2a !important; }
[data-testid="stExpander"] > div, [data-testid="stExpander"] details > div {
    background-color: #ffffff !important;
    padding: 14px 16px !important;
}
[data-testid="stExpander"] > div * { color: #0d1b2a !important; }

/* 8. DROPDOWNS */
[data-baseweb="select"],
[data-baseweb="select"] > div,
[data-baseweb="select"] > div > div {
    background-color: #ffffff !important;
    color: #0d1b2a !important;
    border: 2px solid #5ba4c4 !important;
    border-radius: 6px !important;
}
[data-baseweb="select"] span,
[data-baseweb="select"] [role="combobox"],
[data-baseweb="select"] input,
[data-baseweb="select"] [data-testid="stSelectboxValue"] { color: #0d1b2a !important; }
[data-baseweb="select"] svg { fill: #1565c0 !important; }
[data-baseweb="popover"], [data-baseweb="popover"] > div,
[role="listbox"], [data-baseweb="menu"], [data-baseweb="menu"] > ul,
ul[data-testid="stSelectboxVirtualDropdown"],
ul[data-testid="stMultiSelectDropdown"] {
    background-color: #ffffff !important;
    border: 2px solid #5ba4c4 !important;
    border-radius: 8px !important;
    box-shadow: 0 6px 24px rgba(21,101,192,0.18) !important;
}
[role="option"], [data-baseweb="menu-item"], li[role="option"] {
    background-color: #ffffff !important;
    color: #0d1b2a !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 9px 14px !important;
    border-bottom: 1px solid #e8f0fb !important;
}
[role="option"] *, [data-baseweb="menu-item"] *, li[role="option"] * {
    color: #0d1b2a !important;
    background-color: transparent !important;
}
[role="option"]:hover, [data-baseweb="menu-item"]:hover, li[role="option"]:hover {
    background-color: #dbeeff !important;
    color: #0d47a1 !important;
}
[role="option"]:hover *, [data-baseweb="menu-item"]:hover *, li[role="option"]:hover * {
    color: #0d47a1 !important;
}
[aria-selected="true"][role="option"], li[role="option"][aria-selected="true"] {
    background-color: #b3d9f0 !important;
    color: #0d1b2a !important;
    font-weight: 700 !important;
}
[aria-selected="true"][role="option"] *, li[role="option"][aria-selected="true"] * {
    color: #0d1b2a !important;
}
[data-baseweb="tag"], [data-baseweb="tag"] span {
    background-color: #1565c0 !important;
    color: #ffffff !important;
    border-radius: 20px !important;
    font-weight: 700 !important;
    padding: 2px 10px !important;
}
[data-baseweb="tag"] [data-baseweb="tag-action"] { color: #ffffff !important; }

/* 9. TEXT INPUTS */
[data-baseweb="input"], [data-baseweb="input"] > div,
[data-baseweb="textarea"], [data-baseweb="textarea"] > div,
[data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea, [data-testid="stChatInput"] textarea,
textarea, input[type="text"], input[type="number"] {
    background-color: #ffffff !important;
    color: #0d1b2a !important;
    border: 2px solid #5ba4c4 !important;
    border-radius: 6px !important;
    font-size: 14px !important;
}
input:focus, textarea:focus,
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #1565c0 !important;
    box-shadow: 0 0 0 3px rgba(21,101,192,0.20) !important;
    outline: none !important;
    background-color: #f0f8ff !important;
}

/* 10. RADIO & CHECKBOXES */
[data-testid="stCheckbox"] span,
[data-testid="stRadio"] label,
[data-testid="stRadio"] > div label {
    color: #0d1b2a !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* 11. SLIDERS */
[data-testid="stSlider"] [role="slider"] {
    background-color: #1565c0 !important;
    border-color: #0d47a1 !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #1a3050 !important; }

/* 12. ALERTS */
[data-testid="stAlert"], [data-baseweb="notification"] {
    border-radius: 8px !important;
    background-color: #deeffe !important;
    color: #0d1b2a !important;
    border-left: 5px solid #1565c0 !important;
    font-weight: 500 !important;
}
[data-testid="stAlert"] *, [data-baseweb="notification"] * { color: inherit !important; }
.stSuccess, [class*="stSuccess"] {
    background-color: #c8f0d8 !important;
    border-left: 5px solid #1a7a3e !important;
    color: #0a3320 !important;
}
.stWarning, [class*="stWarning"] {
    background-color: #fff0b3 !important;
    border-left: 5px solid #9a6000 !important;
    color: #4a2e00 !important;
}
.stError, [class*="stError"] {
    background-color: #fddede !important;
    border-left: 5px solid #9b1c1c !important;
    color: #5a0000 !important;
}
.stInfo, [class*="stInfo"] {
    background-color: #deeffe !important;
    border-left: 5px solid #1565c0 !important;
    color: #0d1b2a !important;
}

/* 13. TABLES / DATAFRAMES - HIGHEST PRIORITY
   st.table() renders a real <table> in the DOM — fully CSS-controllable.
   st.dataframe() renders on a <canvas> — only config.toml controls it.
   We use st.table() for Variable Details and Data Preview so these rules
   are guaranteed to apply. */

/* st.table wrapper */
[data-testid="stTable"] {
    width: 100% !important;
    overflow-x: auto !important;
}
[data-testid="stTable"] table {
    width: 100% !important;
    border-collapse: collapse !important;
    background-color: #ffffff !important;
    border: 3px solid #1565c0 !important;
    font-size: 13px !important;
}
[data-testid="stTable"] thead tr {
    background-color: #1565c0 !important;
}
[data-testid="stTable"] th {
    background-color: #1565c0 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    border: 2px solid #0d47a1 !important;
    text-align: left !important;
    white-space: nowrap !important;
}
[data-testid="stTable"] td {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    border: 1px solid #5ba4c4 !important;
}
[data-testid="stTable"] tbody tr:nth-child(even) td {
    background-color: #e8f4fb !important;
    color: #000000 !important;
}
[data-testid="stTable"] tbody tr:hover td {
    background-color: #b3d9f0 !important;
    color: #000000 !important;
}

/* Also target table without testid wrapper (some Streamlit versions) */
[data-testid="stMarkdownContainer"] + div table,
.stTable table,
div[class*="stTable"] table {
    border-collapse: collapse !important;
    background: #ffffff !important;
    border: 3px solid #1565c0 !important;
    width: 100% !important;
}
.stTable th, div[class*="stTable"] th {
    background-color: #1565c0 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    padding: 10px 14px !important;
    border: 2px solid #0d47a1 !important;
}
.stTable td, div[class*="stTable"] td {
    background-color: #ffffff !important;
    color: #000000 !important;
    padding: 8px 14px !important;
    border: 1px solid #5ba4c4 !important;
}

/* Dataframes canvas wrapper */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div {
    background-color: #ffffff !important;
    border: 2.5px solid #5ba4c4 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}
.dvn-scroller, .dvn-stack { background-color: #ffffff !important; }
[data-testid="stDataFrame"] [role="columnheader"],
[data-testid="stDataFrame"] th,
[data-testid="glideDataEditor"] [role="columnheader"] {
    background-color: #1565c0 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border-bottom: 2px solid #0d47a1 !important;
    padding: 9px 12px !important;
}
[data-testid="stDataFrame"] [role="gridcell"],
[data-testid="stDataFrame"] td,
[data-testid="glideDataEditor"] [role="gridcell"] {
    background-color: #ffffff !important;
    color: #0d1b2a !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    border-bottom: 1px solid #b3d9f0 !important;
    padding: 7px 12px !important;
}
[data-testid="stDataFrame"] [role="row"]:nth-child(even) [role="gridcell"],
[data-testid="stDataFrame"] tr:nth-child(even) td {
    background-color: #f0f7ff !important;
    color: #0d1b2a !important;
}
[data-testid="stDataFrame"] [role="row"]:hover [role="gridcell"],
[data-testid="stDataFrame"] tr:hover td {
    background-color: #b3d9f0 !important;
    color: #0d1b2a !important;
}
[data-testid="stDataFrame"] [role="gridcell"][aria-selected="true"] {
    background-color: #dbeeff !important;
    color: #0d1b2a !important;
    outline: 2px solid #1565c0 !important;
}
/* st.table — reinforced full-border override (covers all Streamlit versions) */
[data-testid="stTable"] table,
[data-testid="stTable"] > div > table {
    width: 100% !important;
    border-collapse: collapse !important;
    background-color: #ffffff !important;
    border: 3px solid #1565c0 !important;
    font-size: 13px !important;
}
[data-testid="stTable"] th,
[data-testid="stTable"] thead th {
    background-color: #1565c0 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    border: 2px solid #0d47a1 !important;
    text-align: left !important;
    white-space: nowrap !important;
}
[data-testid="stTable"] td,
[data-testid="stTable"] tbody td {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
    border: 1px solid #1565c0 !important;
}
[data-testid="stTable"] tbody tr:nth-child(even) td {
    background-color: #e8f4fb !important;
    color: #000000 !important;
}
[data-testid="stTable"] tbody tr:hover td {
    background-color: #b3d9f0 !important;
    color: #000000 !important;
}
/* Markdown tables */
[data-testid="stMarkdownContainer"] table {
    width: 100% !important;
    border-collapse: collapse !important;
    background-color: #ffffff !important;
    border: 2px solid #1565c0 !important;
    margin: 8px 0 !important;
}
[data-testid="stMarkdownContainer"] th {
    background-color: #1565c0 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    padding: 9px 14px !important;
    border: 2px solid #0d47a1 !important;
    text-align: left !important;
}
[data-testid="stMarkdownContainer"] td {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 7px 14px !important;
    border: 1px solid #1565c0 !important;
}
[data-testid="stMarkdownContainer"] tr:nth-child(even) td {
    background-color: #e8f4fb !important;
    color: #000000 !important;
}
[data-testid="stMarkdownContainer"] tr:hover td { background-color: #b3d9f0 !important; }

/* 14. DOWNLOAD BUTTONS */
[data-testid="stDownloadButton"] button {
    background-color: #e8f5e9 !important;
    color: #1b5e20 !important;
    border: 2px solid #388e3c !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
}
[data-testid="stDownloadButton"] button:hover {
    background-color: #c8e6c9 !important;
    color: #0a3d0a !important;
}

/* 15. FILE UPLOADER */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"] {
    background-color: #f0f8ff !important;
    border: 2px dashed #1565c0 !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] *, [data-testid="stFileUploaderDropzone"] * {
    color: #0d1b2a !important;
}
[data-testid="stFileUploader"] button {
    background-color: #dbeeff !important;
    color: #0d47a1 !important;
    border: 2px solid #1565c0 !important;
}

/* 16. CHAT */
[data-testid="stChatMessage"] {
    background-color: #ffffff !important;
    border: 2px solid #5ba4c4 !important;
    border-radius: 10px !important;
}
[data-testid="stChatMessage"] * { color: #0d1b2a !important; }
[data-testid="stChatInputContainer"],
[data-testid="stChatInputContainer"] textarea {
    background-color: #ffffff !important;
    color: #0d1b2a !important;
    border: 2px solid #5ba4c4 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInputSubmitButton"] button {
    background-color: #1565c0 !important;
    color: #ffffff !important;
    border: none !important;
}

/* 17. TOOLTIPS */
[data-baseweb="tooltip"], [data-baseweb="tooltip"] > div, [role="tooltip"] {
    background-color: #0d1b2a !important;
    color: #ffffff !important;
    border-radius: 4px !important;
    font-size: 12px !important;
}
[role="tooltip"] * { color: #ffffff !important; }

/* 18. MODALS */
[data-baseweb="modal"], [data-baseweb="dialog"], [role="dialog"] {
    background-color: #ffffff !important;
    border: 2px solid #5ba4c4 !important;
    border-radius: 10px !important;
}
[data-baseweb="modal"] *, [data-baseweb="dialog"] *, [role="dialog"] * {
    color: #0d1b2a !important;
}

/* 19. FORMS */
[data-testid="stForm"] {
    background-color: #f8fbff !important;
    border: 2px solid #5ba4c4 !important;
    border-radius: 8px !important;
    padding: 16px !important;
}

/* 20. CAPTIONS */
[data-testid="stCaptionContainer"] p, .stCaption, small, caption {
    color: #2c4a6e !important;
    font-size: 12px !important;
}

/* 21. HR */
hr { border-color: #5ba4c4 !important; border-width: 1.5px !important; opacity: 1 !important; }

/* 22. NUMBER INPUT */
[data-testid="stNumberInput"] button {
    background-color: #dbeeff !important;
    color: #0d47a1 !important;
    border: 2px solid #5ba4c4 !important;
    font-weight: 700 !important;
}

/* 23. PROGRESS BAR */
[data-testid="stProgressBar"] > div { background-color: #dbeeff !important; }
[data-testid="stProgressBar"] > div > div { background-color: #1565c0 !important; }

/* 24. METRIC WIDGETS */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    border: 2px solid #5ba4c4 !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}
[data-testid="stMetricLabel"] { color: #1a3050 !important; font-weight: 700 !important; }
[data-testid="stMetricValue"] { color: #0d47a1 !important; font-weight: 800 !important; }
[data-testid="stMetricDelta"]  { color: #1565c0 !important; font-weight: 700 !important; }

/* CODE BLOCKS */
[data-testid="stMarkdownContainer"] pre,
[data-testid="stMarkdownContainer"] code {
    background-color: #e8f0fb !important;
    color: #0d1b2a !important;
    border: 1.5px solid #5ba4c4 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

/* UTILITY CLASSES */
.metric-card {
    background: #ffffff;
    border: 2px solid #5ba4c4;
    border-left: 6px solid #1565c0;
    border-radius: 10px;
    padding: 16px 18px;
}
.metric-val { font-size: 28px; font-weight: 800; color: #0d47a1; }
.metric-lbl { font-size: 12px; color: #1a3050; margin-top: 2px; font-weight: 700; }

.info-box {
    background: #dbeeff;
    border: 1.5px solid #5ba4c4;
    border-left: 5px solid #1565c0;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #0d1b2a;
    font-weight: 500;
    margin: 8px 0;
}

.warn-box {
    background: #fff8e1;
    border: 1.5px solid #f0b429;
    border-left: 5px solid #9a6000;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #4a2e00;
    font-weight: 500;
    margin: 8px 0;
}

.result-box {
    background: #f0f8ff;
    border: 2px solid #5ba4c4;
    border-left: 5px solid #1565c0;
    border-radius: 8px;
    padding: 14px 18px;
    color: #0d1b2a;
    font-weight: 500;
    margin: 8px 0;
}

.section-title {
    font-size: 20px;
    font-weight: 800;
    color: #0d47a1;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 3px solid #1565c0;
}

.tag {
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    border-radius: 20px;
    padding: 3px 11px;
    margin: 2px;
    border: 1.5px solid rgba(0,0,0,0.15);
}
.tag-num  { background: #bbdefb; color: #01416b; }
.tag-cat  { background: #f8bbd0; color: #6a0030; }
.tag-dt   { background: #c8e6c9; color: #1b5e20; }
.tag-text { background: #fff9c4; color: #5a3e00; }

.step-card {
    background: #ffffff;
    border: 2px solid #5ba4c4;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
    color: #0d1b2a;
}
.step-card:hover { border-color: #1565c0; background: #f0f8ff; }

.viva-card {
    background: #ffffff;
    border: 2px solid #5ba4c4;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
    color: #0d1b2a;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Initialisation ──────────────────────────────────────────────
def _init():
    defaults = {
        "raw_df":              None,       # original uploaded data — NEVER modified
        "working_df":          None,       # cleaned working copy
        "file_name":           "",
        "profile":             None,       # dataset profile dict
        "cleaning_log":        [],         # list of cleaning log dicts
        "analysis_history":    [],         # list of statistical result dicts
        "chat_history":        [],         # list of {role, content}
        "key_findings":        [],
        "last_result":         {},
        "user_role_overrides": {},         # user-confirmed role overrides
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px 14px;border-bottom:1px solid rgba(255,255,255,0.18);margin-bottom:4px;">
        <div style="font-size:20px;font-weight:800;color:#ffffff;letter-spacing:-0.3px;">🔬 AI Research Analyst</div>
        <div style="font-size:11px;color:#b2ebf2;margin-top:3px;font-weight:500;">No-Code Research Data Analytics</div>
    </div>
    """, unsafe_allow_html=True)

    # Status badge
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        st.markdown(f"""
        <div style="background:rgba(253,246,211,0.15);border:1px solid rgba(253,246,211,0.35);
                    border-radius:8px;padding:10px 14px;margin:8px 8px 10px;">
            <div style="font-size:10px;color:#b2ebf2;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Active Dataset</div>
            <div style="font-size:13px;color:#ffffff;font-weight:700;margin-top:3px;">{st.session_state.file_name}</div>
            <div style="font-size:11px;color:#e0f7fa;margin-top:2px;">{len(df):,} rows &times; {len(df.columns)} columns</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.07);border-radius:8px;padding:10px 14px;margin:8px 8px 10px;
                    border:1px dashed rgba(255,255,255,0.25);">
            <div style="font-size:11px;color:#b2ebf2;font-weight:600;">No dataset loaded</div>
            <div style="font-size:11px;color:#e0f7fa;margin-top:2px;">Upload a file to begin</div>
        </div>
        """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Home",
        "📤  Upload Data",
        "🔍  Data Overview",
        "🧬  Variable Classification",
        "🧹  Data Cleaning",
        "📊  Statistical Analysis",
        "📈  Visual Analysis",
        "🤖  AI Research Assistant",
        "🎓  Viva Preparation",
        "🔎  Findings",
        "📝  Research Report",
        "💾  Export",
        "📚  Teaching Mode",
    ], label_visibility="collapsed")

    st.markdown("""
    <div style="padding:14px 16px;margin-top:16px;border-top:1px solid rgba(255,255,255,0.15);
                font-size:10px;color:#b2ebf2;text-align:center;line-height:1.6;">
        Statistical Engine: scipy &middot; statsmodels<br>
        AI Layer: Interpretation only &mdash; no invented values
    </div>
    """, unsafe_allow_html=True)


# ── Helper utilities ──────────────────────────────────────────────────────────
def metric_card(val, label, color="#0d7a8a"):
    return f"""<div class="metric-card" style="border-left-color:{color}">
        <div class="metric-val" style="color:{color}">{val}</div>
        <div class="metric-lbl">{label}</div>
    </div>"""

def info_box(text):
    st.markdown(f'<div class="info-box">ℹ️ {text}</div>', unsafe_allow_html=True)

def warn_box(text):
    st.markdown(f'<div class="warn-box">⚠️ {text}</div>', unsafe_allow_html=True)

def _get_df():
    """Return working_df if available, else raw_df."""
    return st.session_state.working_df if st.session_state.working_df is not None else st.session_state.raw_df

def _require_data():
    """Show error and stop if no data loaded."""
    if st.session_state.raw_df is None:
        st.warning("📤 Please upload a dataset first. Go to **Upload Data** in the sidebar.")
        st.stop()

def _profile(df):
    """
    Central profiling helper — ALWAYS passes user_role_overrides.

    This is the single authoritative call site for profile_dataset() in the UI.
    Every page that needs to (re-)profile the DataFrame must call _profile(df)
    rather than calling profile_dataset() directly.  This guarantees that
    researcher-confirmed role overrides are honoured throughout the session
    regardless of which page triggered the re-profile.

    If profile_dataset() raises an unexpected exception (e.g. a future engine
    change removes a parameter), the error is caught, a user-facing warning is
    shown, and the previous profile is returned unchanged so the app does not
    crash.
    """
    from engine.profiler import profile_dataset
    overrides = st.session_state.get("user_role_overrides", {})
    try:
        return profile_dataset(df, user_overrides=overrides)
    except TypeError as e:
        # Defensive: if signature mismatch occurs, show a clear message
        st.error(
            f"⚠️ Internal error while profiling the dataset: {e}\n\n"
            "This may be caused by an outdated module cache. "
            "Please refresh the page or restart the application."
        )
        # Return existing profile unchanged to avoid cascading crashes
        return st.session_state.get("profile") or {}
    except Exception as e:
        st.error(f"⚠️ Unexpected profiling error: {e}")
        return st.session_state.get("profile") or {}


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS — role helpers (used by the Statistical Analysis page)
# ══════════════════════════════════════════════════════════════════════════════

def _get_role_cols(profile: dict, df) -> dict:
    """
    Derive role-gated column lists from the profile, honouring any researcher
    overrides stored in st.session_state.user_role_overrides.

    Returns a dict with keys:
      subst_num_cols   — substantive numerical (safe for mean/SD/correlation/regression)
      cat_cols         — categorical + boolean + ordinal (safe for freq/chi-square)
      ordinal_cols     — ordinal only (suitable for Spearman, ordinal stats)
      all_analysis_cols— everything except blocked roles
      blocked_cols     — {col: reason_string} for cols excluded from analysis
    """
    BLOCKED_ROLES = {"identifier", "serial_number", "administrative_code",
                     "constant", "free_text", "datetime"}

    BLOCKED_REASON = {
        "identifier":          "Identifier — arbitrary unique value, not a measured quantity",
        "serial_number":       "Serial number — assigned by position, not a measured quantity",
        "administrative_code": "Administrative code — numeric label, arithmetic is meaningless",
        "constant":            "Constant — no variation, cannot be used in any statistical test",
        "free_text":           "Free text — unstructured responses, quantitative methods do not apply",
        "datetime":            "Date/time — not directly usable in arithmetic tests",
    }

    classifications = dict(profile.get("classifications", {}))
    overrides = st.session_state.get("user_role_overrides", {})

    # Merge overrides into classifications
    for col, ovr in overrides.items():
        if col in classifications:
            classifications[col] = dict(classifications[col])
            if "analytical_role" in ovr:
                classifications[col]["analytical_role"] = ovr["analytical_role"]
            if "measurement_level" in ovr:
                classifications[col]["measurement_level"] = ovr["measurement_level"]

    subst_num_cols    = []
    cat_cols          = []
    ordinal_cols      = []
    all_analysis_cols = []
    blocked_cols      = {}

    all_df_cols = list(df.columns) if df is not None else []

    for col in all_df_cols:
        cl   = classifications.get(col, {})
        role = cl.get("analytical_role", "unknown")
        if role in BLOCKED_ROLES:
            blocked_cols[col] = BLOCKED_REASON.get(role, f"Role '{role}' excluded from analysis")
        else:
            all_analysis_cols.append(col)
            if role == "substantive_numerical":
                subst_num_cols.append(col)
            elif role == "ordinal":
                cat_cols.append(col)
                ordinal_cols.append(col)
            elif role in ("categorical", "boolean"):
                cat_cols.append(col)
            # 'unknown' role: include in all_analysis_cols but not in typed lists
            # The researcher can reclassify in Variable Classification

    return {
        "subst_num_cols":    subst_num_cols,
        "cat_cols":          cat_cols,
        "ordinal_cols":      ordinal_cols,
        "all_analysis_cols": all_analysis_cols,
        "blocked_cols":      blocked_cols,
    }


def _method_assumptions(method_name: str, variables: list, roles_ok: bool,
                         issues: list, assumptions: list):
    """
    Render an assumption/eligibility panel before a run button.

    Parameters
    ----------
    method_name  : display name of the statistical method
    variables    : list of selected variable names
    roles_ok     : True = eligible, False = concerns/not eligible
    issues       : plain-language issue strings (may be empty)
    assumptions  : bullet-point assumption strings
    """
    if roles_ok and not issues:
        badge = "✅ Eligible"
        badge_color = "#1e8e5c"
    elif issues:
        badge = "⚠️ Concerns"
        badge_color = "#b45309"
    else:
        badge = "❌ Not eligible"
        badge_color = "#c0392b"

    var_list = ", ".join(f"`{v}`" for v in variables) if variables else "_None selected_"
    issues_html = "".join(f"<li style='color:#92400e;'>{i}</li>" for i in issues) if issues else ""
    assump_html = "".join(f"<li>{a}</li>" for a in assumptions)

    st.markdown(f"""
<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:6px;
            padding:14px 18px;margin-bottom:12px;color:#0d1b2a;">
  <div style="font-weight:700;font-size:14px;margin-bottom:6px;">
    📋 Assumptions &amp; Eligibility Check
  </div>
  <div style="margin-bottom:4px;"><strong>Method:</strong> {method_name}</div>
  <div style="margin-bottom:4px;"><strong>Variables selected:</strong> {var_list}</div>
  <div style="margin-bottom:6px;">
    <strong>Eligibility:</strong>
    <span style="font-weight:700;color:{badge_color};">{badge}</span>
  </div>
  {f'<div style="margin-bottom:6px;"><strong>Issues:</strong><ul style="margin:4px 0 0 18px;">{issues_html}</ul></div>' if issues_html else ''}
  <div><strong>Assumptions:</strong><ul style="margin:4px 0 0 18px;">{assump_html}</ul></div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown('<div class="section-title">🏠 Welcome to AI Research Analyst</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#e0f7fa 0%,#fdf6d3 100%);border:2px solid #b2ebf2;
                border-radius:14px;padding:24px 28px;margin-bottom:20px;">
        <h2 style="color:#0d7a8a;margin-bottom:8px;font-size:22px;">Your No-Code Research Data Analytics Platform</h2>
        <p style="color:#1a2340;font-size:15px;line-height:1.8;">
        Upload your Excel or CSV file and let the system guide you through the complete research analysis workflow &mdash;
        from data understanding and cleaning, through statistical analysis and visualisation,
        to AI-assisted interpretation and report generation.
        <br><br>
        <strong style="color:#0d7a8a;">You do not need to write any code.</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="step-card">
            <div style="font-size:22px;margin-bottom:6px;">📤</div>
            <div style="font-weight:700;color:#0d7a8a;">Step 1 — Upload</div>
            <div style="font-size:12px;color:#2c3e50;margin-top:4px;">
            Upload your Excel (.xlsx, .xls) or CSV file. The system immediately inspects it.
            </div>
        </div>
        <div class="step-card">
            <div style="font-size:22px;margin-bottom:6px;">🔍</div>
            <div style="font-weight:700;color:#0d7a8a;">Step 2 — Understand</div>
            <div style="font-size:12px;color:#2c3e50;margin-top:4px;">
            Explore your dataset: variable types, missing values, duplicates, distributions, and anomalies.
            </div>
        </div>
        <div class="step-card">
            <div style="font-size:22px;margin-bottom:6px;">🧹</div>
            <div style="font-weight:700;color:#0d7a8a;">Step 3 — Clean</div>
            <div style="font-size:12px;color:#2c3e50;margin-top:4px;">
            Detect and resolve missing values, duplicates, inconsistent categories, and outliers.
            The original file is never modified.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-card">
            <div style="font-size:22px;margin-bottom:6px;">📊</div>
            <div style="font-weight:700;color:#0d7a8a;">Step 4 — Analyse</div>
            <div style="font-size:12px;color:#2c3e50;margin-top:4px;">
            Run descriptive statistics, correlation, t-tests, ANOVA, chi-square, and regression
            — no code required.
            </div>
        </div>
        <div class="step-card">
            <div style="font-size:22px;margin-bottom:6px;">📈</div>
            <div style="font-weight:700;color:#0d7a8a;">Step 5 — Visualise</div>
            <div style="font-size:12px;color:#2c3e50;margin-top:4px;">
            Create bar charts, histograms, scatter plots, heatmaps, and more.
            AI recommends the right chart for your variables.
            </div>
        </div>
        <div class="step-card">
            <div style="font-size:22px;margin-bottom:6px;">🤖</div>
            <div style="font-weight:700;color:#0d7a8a;">Step 6 — Interpret</div>
            <div style="font-size:12px;color:#2c3e50;margin-top:4px;">
            Ask the AI Research Assistant to explain your results, justify your method choices,
            and prepare for your viva.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="step-card">
            <div style="font-size:22px;margin-bottom:6px;">🎓</div>
            <div style="font-weight:700;color:#0d7a8a;">Step 7 — Viva Prep</div>
            <div style="font-size:12px;color:#2c3e50;margin-top:4px;">
            Practise answering likely viva questions based on your actual analysis choices.
            </div>
        </div>
        <div class="step-card">
            <div style="font-size:22px;margin-bottom:6px;">📝</div>
            <div style="font-weight:700;color:#0d7a8a;">Step 8 — Report</div>
            <div style="font-size:12px;color:#2c3e50;margin-top:4px;">
            Generate an evidence-based research report with traceable conclusions.
            </div>
        </div>
        <div class="step-card">
            <div style="font-size:22px;margin-bottom:6px;">💾</div>
            <div style="font-weight:700;color:#0d7a8a;">Step 9 — Export</div>
            <div style="font-size:12px;color:#2c3e50;margin-top:4px;">
            Download your cleaned data, statistical tables, charts, and full report.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-box">
        <strong>⚠️ Important Principle:</strong>
        All statistical calculations are performed by established scientific libraries (scipy, statsmodels).
        The AI layer explains and interprets results — it does not calculate or invent statistical values.
        <br><br>
        <strong>Every conclusion in this platform is traceable to:</strong>
        Raw Data → Calculation → Statistical Result → Interpretation → Conclusion
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.raw_df is not None:
        st.success(f"✅ Dataset loaded: **{st.session_state.file_name}** — "
                   f"{len(st.session_state.raw_df):,} rows, "
                   f"{len(st.session_state.raw_df.columns)} columns. "
                   "Use the sidebar to continue your analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: UPLOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
elif "Upload" in page:
    st.markdown('<div class="section-title">📤 Upload Your Research Data</div>', unsafe_allow_html=True)

    info_box("Upload your Excel or CSV file. The original file will never be modified — "
             "all cleaning and analysis is performed on a working copy.")

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Choose a file",
            type=["xlsx", "xls", "csv"],
            help="Supported formats: Excel (.xlsx, .xls) and CSV (.csv)"
        )

    with col_info:
        st.markdown("""
        **What happens after upload?**
        1. File is read into memory
        2. Variable types are auto-detected
        3. Missing values are counted
        4. Duplicates are identified
        5. A full data profile is generated

        **Your data stays local** — it is not sent to any external server.
        """)

    if uploaded is not None:
        try:
            with st.spinner("Reading file and profiling dataset..."):
                if uploaded.name.endswith(".csv"):
                    df = pd.read_csv(uploaded)
                else:
                    df = pd.read_excel(uploaded)

                # Clean column names
                df.columns = [str(c).strip() for c in df.columns]

                # Store original
                st.session_state.raw_df            = df.copy()
                st.session_state.working_df        = df.copy()
                st.session_state.file_name         = uploaded.name
                st.session_state.cleaning_log      = []
                st.session_state.analysis_history  = []
                st.session_state.key_findings      = []
                st.session_state.last_result       = {}
                st.session_state.chat_history      = []
                st.session_state.user_role_overrides = {}   # reset overrides on new file

                # Profile — use _profile() so overrides are always passed
                from engine.profiler import generate_plain_summary
                profile = _profile(df)
                st.session_state.profile = profile

            st.success(f"✅ File uploaded successfully: **{uploaded.name}**")

            # Quick summary
            st.markdown("### Dataset Quick Summary")
            from engine.profiler import generate_plain_summary
            st.markdown(generate_plain_summary(profile))

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(metric_card(f"{profile['n_rows']:,}", "Total Rows"), unsafe_allow_html=True)
            c2.markdown(metric_card(f"{profile['n_cols']}", "Total Columns"), unsafe_allow_html=True)
            c3.markdown(metric_card(f"{profile['total_missing']:,}", "Missing Values",
                                    color="#c0392b" if profile['total_missing'] > 0 else "#1e8e5c"),
                        unsafe_allow_html=True)
            c4.markdown(metric_card(f"{profile['duplicate_rows']}", "Duplicate Rows",
                                    color="#c0392b" if profile['duplicate_rows'] > 0 else "#1e8e5c"),
                        unsafe_allow_html=True)

            st.markdown("### Variable Types Detected")
            cols_info = []
            for col in df.columns:
                t = profile["variable_types"].get(col, "unknown")
                tag_class = {"numerical": "tag-num", "categorical": "tag-cat",
                             "datetime": "tag-dt", "text": "tag-text",
                             "categorical_numeric": "tag-cat"}.get(t, "tag-text")
                cols_info.append(f'<span class="tag {tag_class}">{col}: {t}</span>')
            st.markdown(" ".join(cols_info), unsafe_allow_html=True)

            st.markdown("")
            info_box("Variable types were auto-detected. If any type is incorrectly classified, "
                     "you can change it in the **Data Cleaning** section.")

            if profile["total_missing"] > 0 or profile["duplicate_rows"] > 0:
                st.warning("⚠️ Data quality issues detected. Go to **Data Cleaning** to resolve them before analysis.")
            else:
                st.success("✅ No missing values or duplicates detected. Proceed to **Data Overview** or **Statistical Analysis**.")

        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.caption("Please check that your file is a valid Excel (.xlsx, .xls) or CSV file, "
                       "is not password-protected, and is not empty.")

    # Reset button
    if st.session_state.raw_df is not None:
        st.markdown("---")
        if st.button("🗑 Reset / Remove Dataset", type="secondary"):
            for k in ["raw_df", "working_df", "file_name", "profile",
                      "cleaning_log", "analysis_history", "key_findings",
                      "last_result", "chat_history"]:
                st.session_state[k] = None if k.endswith("_df") or k in ["profile"] else \
                                       "" if k == "file_name" else \
                                       {} if k == "last_result" else []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif "Overview" in page:
    _require_data()
    st.markdown('<div class="section-title">🔍 Data Overview</div>', unsafe_allow_html=True)

    df      = _get_df()
    profile = st.session_state.profile

    if profile is None:
        profile = _profile(df)
        st.session_state.profile = profile

    # Metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.markdown(metric_card(f"{profile['n_rows']:,}", "Rows"), unsafe_allow_html=True)
    m2.markdown(metric_card(f"{profile['n_cols']}", "Variables"), unsafe_allow_html=True)
    m3.markdown(metric_card(f"{len(profile['numerical_cols'])}", "Numerical", "#01416b"), unsafe_allow_html=True)
    m4.markdown(metric_card(f"{len(profile['categorical_cols'])}", "Categorical", "#5a0030"), unsafe_allow_html=True)
    m5.markdown(metric_card(f"{profile['total_missing']:,}", "Missing Values",
                            "#c0392b" if profile["total_missing"] > 0 else "#1e8e5c"), unsafe_allow_html=True)

    st.markdown("")

    # Plain language summary
    from engine.profiler import generate_plain_summary
    st.markdown("### 📋 Automatic Dataset Summary")
    st.markdown(generate_plain_summary(profile))

    st.markdown("---")

    # Per-column statistics — st.table() renders a real DOM table (not a
    # canvas), so the CSS in the global style block always applies and text
    # is always visible.  reset_index(drop=True) removes the pandas index.
    st.markdown("### 📊 Variable Details")
    tab1, tab2, tab3 = st.tabs(["All Variables", "Numerical Summary", "Categorical Summary"])

    with tab1:
        rows = []
        for col in df.columns:
            cs = profile["col_stats"].get(col, {})
            miss = cs.get("missing", 0)
            rows.append({
                "Variable":    col,
                "Type":        cs.get("type", "unknown"),
                "Non-missing": profile["n_rows"] - miss,
                "Missing":     miss,
                "Missing %":   cs.get("missing_pct", 0),
                "Unique":      cs.get("unique", 0),
            })
        st.table(pd.DataFrame(rows).reset_index(drop=True))

    with tab2:
        num_rows = []
        for col in profile["numerical_cols"]:
            cs = profile["col_stats"].get(col, {})
            num_rows.append({
                "Variable":       col,
                "N":              profile["n_rows"] - cs.get("missing", 0),
                "Missing":        cs.get("missing", 0),
                "Mean":           round(cs.get("mean", 0), 4) if cs.get("mean") != "" else "",
                "Median":         round(cs.get("median", 0), 4) if cs.get("median") != "" else "",
                "Std Dev":        round(cs.get("std", 0), 4) if cs.get("std") != "" else "",
                "Min":            cs.get("min", ""),
                "Max":            cs.get("max", ""),
                "Skewness":       round(cs.get("skewness", 0), 4) if cs.get("skewness") != "" else "",
                "Outliers (IQR)": cs.get("outliers", {}).get("count", 0),
            })
        if num_rows:
            st.table(pd.DataFrame(num_rows).reset_index(drop=True))
        else:
            st.info("No numerical variables detected.")

    with tab3:
        for col in profile["categorical_cols"]:
            cs = profile["col_stats"].get(col, {})
            st.markdown(f"**{col}** — {cs.get('unique', 0)} unique categories")
            top = cs.get("top_values", {})
            if top:
                tv = pd.DataFrame(list(top.items()), columns=["Value", "Count"])
                tv["Percentage"] = (tv["Count"] / profile["n_rows"] * 100).round(2).astype(str) + "%"
                st.table(tv.reset_index(drop=True))

    st.markdown("---")

    # Auto EDA Charts
    st.markdown("### 📊 Automatic Exploratory Charts")
    info_box("Quick charts are automatically generated from your data to help you understand key distributions and patterns.")

    try:
        from engine.visualizer import frequency_bar, histogram, correlation_heatmap
        from engine.statistics import correlation_matrix as _corr_matrix
        _eda_imports_ok = True
    except ImportError as _eda_err:
        _eda_imports_ok = False
        st.error(
            f"⚠️ Statistical module configuration error: could not load chart or statistics engine. "
            f"Technical detail: {_eda_err}"
        )

    eda_num = profile.get("analysis_ready_numerical", profile.get("numerical_cols", []))
    eda_cat = profile.get("analysis_ready_categorical", profile.get("categorical_cols", []))

    if _eda_imports_ok:
        eda_tab1, eda_tab2, eda_tab3 = st.tabs(["📊 Categorical Distributions", "📈 Numerical Distributions", "🔗 Correlation Overview"])

        with eda_tab1:
            if not eda_cat:
                st.info("No categorical variables available for charts.")
            else:
                show_cats = eda_cat[:6]  # limit to 6 to avoid slowness
                cols_per_row = 2
                for i in range(0, len(show_cats), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j, cat_col in enumerate(show_cats[i:i+cols_per_row]):
                        try:
                            fig = frequency_bar(df, cat_col)
                            row_cols[j].plotly_chart(fig, use_container_width=True)
                        except Exception as _e:
                            row_cols[j].warning(f"Could not plot {cat_col}: {_e}")
                if len(eda_cat) > 6:
                    st.caption(f"Showing first 6 of {len(eda_cat)} categorical variables.")

        with eda_tab2:
            if not eda_num:
                st.info("No numerical variables available for charts.")
            else:
                show_nums = eda_num[:6]
                cols_per_row = 2
                for i in range(0, len(show_nums), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j, num_col in enumerate(show_nums[i:i+cols_per_row]):
                        try:
                            fig = histogram(df, num_col)
                            row_cols[j].plotly_chart(fig, use_container_width=True)
                        except Exception as _e:
                            row_cols[j].warning(f"Could not plot {num_col}: {_e}")
                if len(eda_num) > 6:
                    st.caption(f"Showing first 6 of {len(eda_num)} numerical variables.")

        with eda_tab3:
            if len(eda_num) < 2:
                st.info("Need at least 2 numerical variables for a correlation overview.")
            else:
                try:
                    corr_cols = eda_num[:8]
                    corr_mat = _corr_matrix(df, corr_cols)
                    fig = correlation_heatmap(corr_mat)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Correlation values range from -1 (strong negative) to +1 (strong positive). "
                               "Values near 0 indicate no linear relationship.")
                except Exception as _e:
                    st.warning(f"Correlation overview could not be generated: {_e}")

    st.markdown("---")

    # Data preview — st.table() renders a real DOM table, always readable
    st.markdown("### 👁 Data Preview")
    n_preview = st.slider("Rows to preview", 5, min(100, len(df)), 10, step=5)
    st.table(df.head(n_preview).reset_index(drop=True))

    # Issues
    issues = []
    if profile["total_missing"] > 0:
        issues.append(f"**{profile['total_missing']} missing values** across {sum(1 for v in profile['missing_counts'].values() if v > 0)} column(s)")
    if profile["duplicate_rows"] > 0:
        issues.append(f"**{profile['duplicate_rows']} duplicate rows**")
    for col, incons in profile.get("inconsistencies", {}).items():
        issues.append(f"**Inconsistent categories** in `{col}` — {len(incons)} group(s)")
    for col in profile["numerical_cols"]:
        n_out = profile["col_stats"].get(col, {}).get("outliers", {}).get("count", 0)
        if n_out > 0:
            issues.append(f"**{n_out} potential outlier(s)** in `{col}`")

    if issues:
        st.markdown("### ⚠️ Data Quality Issues Detected")
        for issue in issues:
            st.markdown(f"- {issue}")
        st.markdown("👉 Go to **Data Cleaning** to resolve these issues before analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VARIABLE CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Variable Classification" in page:
    _require_data()
    st.markdown('<div class="section-title">🧬 Variable Classification & Analysis Suitability</div>',
                unsafe_allow_html=True)

    info_box(
        "The system automatically determines each variable's data type AND its analytical role "
        "(e.g. substantive numerical, identifier, categorical, ordinal). "
        "This prevents identifiers and serial numbers from being incorrectly treated as "
        "measurement variables. You can override any classification below."
    )

    df      = _get_df()
    profile = st.session_state.profile
    if profile is None:
        profile = _profile(df)
        st.session_state.profile = profile

    classifications = profile.get("classifications", {})

    # Role badge colours
    ROLE_COLOURS = {
        "substantive_numerical": "#b3e5fc",
        "categorical":           "#fce4ec",
        "ordinal":               "#c8e6c9",
        "boolean":               "#fff9c4",
        "datetime":              "#dcedc8",
        "identifier":            "#ffccbc",
        "serial_number":         "#ffccbc",
        "administrative_code":   "#ffe0b2",
        "constant":              "#e0e0e0",
        "free_text":             "#f8bbd0",
        "unknown":               "#f5f5f5",
    }
    ROLE_TEXT_COLOURS = {
        "substantive_numerical": "#01416b",
        "categorical":           "#6a0030",
        "ordinal":               "#1b5e20",
        "boolean":               "#5a3e00",
        "datetime":              "#2e5c00",
        "identifier":            "#7f2600",
        "serial_number":         "#7f2600",
        "administrative_code":   "#7f3000",
        "constant":              "#424242",
        "free_text":             "#6a0050",
        "unknown":               "#555555",
    }

    ROLE_LABELS = {
        "substantive_numerical": "Substantive Numerical",
        "categorical":           "Categorical (Nominal)",
        "ordinal":               "Ordinal Scale",
        "boolean":               "Boolean / Binary",
        "datetime":              "Date / Time",
        "identifier":            "Identifier / ID",
        "serial_number":         "Serial Number",
        "administrative_code":   "Administrative Code",
        "constant":              "Constant (No Variation)",
        "free_text":             "Free Text",
        "unknown":               "Unknown",
    }

    ALL_ROLES = list(ROLE_LABELS.keys())
    ALL_LEVELS = ["nominal", "ordinal", "interval", "ratio", "datetime",
                  "identifier", "constant", "text"]

    # Load or initialise user overrides from session state
    if "user_role_overrides" not in st.session_state:
        st.session_state.user_role_overrides = {}

    # ── Confirmation warnings at the top ───────────────────────────────────────
    needs_confirm = {col: cl for col, cl in classifications.items()
                     if cl.get("needs_confirmation", False)}
    if needs_confirm:
        st.warning(
            f"⚠️ **{len(needs_confirm)} column(s) require your confirmation.** "
            "The system is not certain of their analytical role. "
            "Please review and confirm below."
        )

    # ── Summary table ──────────────────────────────────────────────────────────
    st.markdown("### Overview")
    summary_rows = []
    for col in df.columns:
        cl = classifications.get(col, {})
        role  = cl.get("analytical_role", "unknown")
        level = cl.get("measurement_level", "unknown")
        conf  = cl.get("confidence", 0.0)
        miss  = profile.get("missing_counts", {}).get(col, 0)
        uniq  = profile.get("col_stats", {}).get(col, {}).get("unique", 0)
        suitable = ", ".join(cl.get("suitable_analyses", [])) or "— (see warnings)"
        warn_count = len(cl.get("warnings", []))
        summary_rows.append({
            "Column":            col,
            "Data Type":         cl.get("data_type", "?"),
            "Analytical Role":   ROLE_LABELS.get(role, role),
            "Measurement Level": level,
            "Unique Values":     uniq,
            "Missing Values":    miss,
            "Suitable Analyses": suitable[:60] + "…" if len(suitable) > 60 else suitable,
            "Confidence":        f"{conf:.0%}",
            "Warnings":          warn_count,
        })
    sum_df = pd.DataFrame(summary_rows)
    st.table(sum_df.reset_index(drop=True))

    st.markdown("---")

    # ── Per-column detail + override ──────────────────────────────────────────
    st.markdown("### Detailed Classification & Override")
    info_box(
        "Expand any column to see the full evidence, warnings, and suitable statistical analyses. "
        "If the system's classification is incorrect, change it here and click **Apply Override**. "
        "Your confirmed classification will be used throughout cleaning, statistics, and reporting."
    )

    for col in df.columns:
        cl   = classifications.get(col, {})
        role = cl.get("analytical_role", "unknown")
        conf = cl.get("confidence", 0.0)
        bg   = ROLE_COLOURS.get(role, "#f5f5f5")
        tc   = ROLE_TEXT_COLOURS.get(role, "#333333")
        confirm_flag = "⚠️ " if cl.get("needs_confirmation", False) else ""

        with st.expander(
            f"{confirm_flag}**{col}** — {ROLE_LABELS.get(role, role)} "
            f"({'%.0f%%' % (conf*100)} confidence)",
            expanded=cl.get("needs_confirmation", False)
        ):
            c_info, c_ctrl = st.columns([3, 2])

            with c_info:
                # Role badge
                st.markdown(
                    f'<span style="background:{bg};color:{tc};font-weight:700;'
                    f'padding:4px 12px;border-radius:20px;font-size:12px;">'
                    f'{ROLE_LABELS.get(role, role)}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(f"**Data Type:** `{cl.get('data_type', '?')}`  |  "
                            f"**Measurement Level:** `{cl.get('measurement_level', '?')}`  |  "
                            f"**Confidence:** `{'%.0f%%' % (conf*100)}`")

                # Evidence
                evidence = cl.get("evidence", [])
                if evidence:
                    st.markdown("**Evidence:**")
                    for e in evidence:
                        st.markdown(f"- {e}")

                # Suitable analyses
                suitable = cl.get("suitable_analyses", [])
                if suitable:
                    st.markdown(f"**Suitable analyses:** {', '.join(suitable)}")
                else:
                    unsuitable = cl.get("unsuitable_reason", "")
                    if unsuitable:
                        st.markdown(
                            f'<div class="warn-box"><strong>Not suitable for standard statistical analysis.</strong><br>'
                            f'{unsuitable}</div>',
                            unsafe_allow_html=True
                        )

                # Warnings
                warnings_list = cl.get("warnings", [])
                for w in warnings_list:
                    st.warning(w)

                # Imputation recommendation
                imp_rec = profile.get("imputation_recommendations", {}).get(col, {})
                if imp_rec:
                    n_miss = profile.get("missing_counts", {}).get(col, 0)
                    if n_miss > 0:
                        st.markdown(f"**Missing value recommendation:** `{imp_rec.get('method', 'N/A')}`")
                        st.markdown(f"*Reason: {imp_rec.get('reason', '')}*")
                        why = imp_rec.get("why_appropriate", "")
                        if why:
                            with st.expander("❓ Why is this recommended?", expanded=False):
                                st.markdown(why)
                                warn = imp_rec.get("warning", "")
                                if warn:
                                    st.warning(warn)

            with c_ctrl:
                st.markdown("**Override Classification**")
                new_role = st.selectbox(
                    "Analytical Role",
                    options=ALL_ROLES,
                    index=ALL_ROLES.index(role) if role in ALL_ROLES else 0,
                    key=f"ovr_role_{col}",
                    format_func=lambda r: ROLE_LABELS.get(r, r)
                )
                new_level = st.selectbox(
                    "Measurement Level",
                    options=ALL_LEVELS,
                    index=ALL_LEVELS.index(cl.get("measurement_level", "nominal"))
                          if cl.get("measurement_level", "nominal") in ALL_LEVELS else 0,
                    key=f"ovr_level_{col}",
                )
                if st.button(f"Apply Override for '{col}'", key=f"btn_ovr_{col}",
                             type="primary"):
                    st.session_state.user_role_overrides[col] = {
                        "analytical_role":   new_role,
                        "measurement_level": new_level,
                    }
                    # Re-profile with updated overrides (user_overrides is read inside _profile)
                    st.session_state.profile = _profile(_get_df())
                    st.success(f"✅ Override applied: '{col}' is now classified as "
                               f"**{ROLE_LABELS.get(new_role, new_role)}** "
                               f"({new_level}). This will be used in all analyses.")
                    st.rerun()

                # Show current override if active
                if col in st.session_state.get("user_role_overrides", {}):
                    ovr = st.session_state.user_role_overrides[col]
                    st.caption(
                        f"⚙️ **Manually overridden:** "
                        f"{ROLE_LABELS.get(ovr['analytical_role'], ovr['analytical_role'])} / "
                        f"{ovr['measurement_level']}"
                    )
                    if st.button(f"↩ Reset to Auto", key=f"btn_reset_{col}"):
                        del st.session_state.user_role_overrides[col]
                        st.session_state.profile = _profile(_get_df())
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
elif "Cleaning" in page:
    _require_data()
    st.markdown('<div class="section-title">🧹 Data Cleaning</div>', unsafe_allow_html=True)

    # ── Session-state initialisations ─────────────────────────────────────────
    if "working_df_snapshots" not in st.session_state:
        st.session_state.working_df_snapshots = []   # stack of (df_copy, log_copy) for undo
    if "cleaning_log" not in st.session_state:
        st.session_state.cleaning_log = []

    # ── Helper: save snapshot before any destructive operation ────────────────
    def _save_snapshot():
        st.session_state.working_df_snapshots.append((
            st.session_state.working_df.copy(),
            list(st.session_state.cleaning_log),
        ))
        # Keep at most 20 undo levels to avoid memory bloat
        if len(st.session_state.working_df_snapshots) > 20:
            st.session_state.working_df_snapshots.pop(0)

    # ── Helper: build the impact disclosure box shown before every Apply ───────
    def _impact_disclosure(what_changes: str, what_lost: str, bias_risk: str,
                           rows_affected: int, rows_total: int,
                           reversible: bool, is_appropriate=None):
        pct = round(rows_affected / max(rows_total, 1) * 100, 1) if rows_total else 0
        appropriate_txt = (
            "✅ Methodologically appropriate for this variable type."
            if is_appropriate is True else
            "⚠️ Researcher decision required — the system cannot determine if this is appropriate."
            if is_appropriate is None else
            "❌ NOT recommended for this variable type."
        )
        st.markdown(f"""
<div style="background:#fff8e1;border:2px solid #f0b429;border-left:6px solid #9a6000;
            border-radius:8px;padding:14px 18px;margin:10px 0;">
<div style="font-size:13px;font-weight:800;color:#5a3500;margin-bottom:10px;">
⚠️ IMPACT DISCLOSURE — Read before applying</div>
<table style="width:100%;border-collapse:collapse;font-size:12px;">
<tr><td style="padding:5px 8px;font-weight:700;color:#3a2000;width:35%;border-bottom:1px solid #f0d080;">
📝 What will change?</td>
<td style="padding:5px 8px;color:#1f2328;border-bottom:1px solid #f0d080;">{what_changes}</td></tr>
<tr><td style="padding:5px 8px;font-weight:700;color:#3a2000;border-bottom:1px solid #f0d080;">
🗑️ What may be lost?</td>
<td style="padding:5px 8px;color:#7f1d1d;border-bottom:1px solid #f0d080;">{what_lost}</td></tr>
<tr><td style="padding:5px 8px;font-weight:700;color:#3a2000;border-bottom:1px solid #f0d080;">
⚡ Bias / distortion risk</td>
<td style="padding:5px 8px;color:#7f2600;border-bottom:1px solid #f0d080;">{bias_risk if bias_risk else "Low — no known bias introduced."}</td></tr>
<tr><td style="padding:5px 8px;font-weight:700;color:#3a2000;border-bottom:1px solid #f0d080;">
📊 Rows affected</td>
<td style="padding:5px 8px;color:#1f2328;border-bottom:1px solid #f0d080;">
<strong>{rows_affected:,} of {rows_total:,} rows ({pct}%)</strong></td></tr>
<tr><td style="padding:5px 8px;font-weight:700;color:#3a2000;border-bottom:1px solid #f0d080;">
↩️ Reversible?</td>
<td style="padding:5px 8px;color:#1f2328;border-bottom:1px solid #f0d080;">
{"✅ Yes — an Undo snapshot will be saved before this operation." if reversible else "❌ No — this change cannot be undone."}</td></tr>
<tr><td style="padding:5px 8px;font-weight:700;color:#3a2000;">
🔬 Methodological appropriateness</td>
<td style="padding:5px 8px;color:#1f2328;">{appropriate_txt}</td></tr>
</table>
</div>""", unsafe_allow_html=True)

    # ── Data-state banner ─────────────────────────────────────────────────────
    n_ops = len(st.session_state.cleaning_log)
    imputed_ops   = sum(1 for e in st.session_state.cleaning_log if e.get("data_state") == "imputed")
    transform_ops = sum(1 for e in st.session_state.cleaning_log if e.get("data_state") == "transformed")
    state_color   = "#dc2626" if (imputed_ops + transform_ops) > 0 else "#16a34a"
    state_label   = (
        "⚠️ Working Copy — contains IMPUTED and/or TRANSFORMED values"
        if (imputed_ops + transform_ops) > 0
        else ("📋 Working Copy — cleaned (no imputed/transformed values)" if n_ops > 0
              else "📋 Working Copy — identical to original uploaded data")
    )
    st.markdown(
        f'<div style="background:#f0f8ff;border:2px solid #5ba4c4;border-left:6px solid {state_color};'
        f'border-radius:8px;padding:10px 16px;margin-bottom:12px;font-size:13px;">'
        f'<strong>Dataset State:</strong> {state_label} &nbsp;|&nbsp; '
        f'{len(st.session_state.working_df):,} rows × {len(st.session_state.working_df.columns)} columns &nbsp;|&nbsp; '
        f'<strong>{n_ops}</strong> operation(s) in log'
        f'</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div style="background:#dbeeff;border:1px solid #5ba4c4;border-radius:6px;'
        'padding:8px 14px;margin-bottom:8px;font-size:12px;color:#0d1b2a;">'
        '🔒 <strong>Original uploaded data is immutable.</strong> '
        'All operations below apply only to the <em>Working Copy</em>. '
        'The original can be restored at any time via "↩ Revert All".'
        '</div>', unsafe_allow_html=True
    )

    # ── Undo / Revert controls ────────────────────────────────────────────────
    ctrl_a, ctrl_b, ctrl_c = st.columns([2, 2, 3])
    with ctrl_a:
        n_snaps = len(st.session_state.working_df_snapshots)
        if st.button(f"↩ Undo Last Operation ({n_snaps} available)",
                     type="secondary", key="undo_last",
                     disabled=(n_snaps == 0)):
            prev_df, prev_log = st.session_state.working_df_snapshots.pop()
            st.session_state.working_df   = prev_df
            st.session_state.cleaning_log = prev_log
            st.session_state.profile      = _profile(prev_df)
            st.success("↩ Last operation undone. Working copy restored to previous state.")
            st.rerun()
    with ctrl_b:
        if st.button("↩ Revert All — Restore Original Data",
                     type="secondary", key="reset_to_original"):
            st.session_state.working_df            = st.session_state.raw_df.copy()
            st.session_state.cleaning_log          = []
            st.session_state.working_df_snapshots  = []
            st.session_state.profile               = _profile(st.session_state.raw_df)
            st.success("✅ Working copy fully restored to original uploaded data. All operations cleared.")
            st.rerun()

    df       = _get_df()
    profile  = st.session_state.profile or {}
    all_cols = list(df.columns)
    from engine.cleaner import (impute_column, recommend_imputation_method,
                                  remove_duplicates, standardise_categories,
                                  fix_data_types, cap_outliers, get_cleaning_summary)

    clean_tab, log_tab = st.tabs(["🛠 Cleaning Operations", "📋 Cleaning Log"])

    with clean_tab:
        # ══════════════════════════════════════════════════════════════════════
        # A. Missing Values
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("#### A. Missing Values")
        missing_cols = [c for c, v in profile.get("missing_counts", {}).items() if v > 0]
        if not missing_cols:
            st.success("✅ No missing values in the current working copy.")
        else:
            classifications = profile.get("classifications", {})
            imp_recs        = profile.get("imputation_recommendations", {})

            for col in missing_cols:
                n_miss  = profile["missing_counts"][col]
                pct     = profile["missing_pct"][col]
                n_total = len(df)
                vtype   = profile.get("variable_types", {}).get(col, "unknown")
                cl      = classifications.get(col, {})
                role    = cl.get("analytical_role", "unknown")
                imp_rec = imp_recs.get(col, {})

                rec_method  = imp_rec.get("method", "mode")
                rec_reason  = imp_rec.get("reason", "")
                rec_why     = imp_rec.get("why_appropriate", "")
                rec_warning = imp_rec.get("warning", "")

                # ── Key flags ─────────────────────────────────────────────────
                is_all_missing = (n_miss >= n_total) or (rec_method == "cannot_impute_all_missing")
                is_id          = (rec_method == "do_not_impute") and not is_all_missing
                high_miss      = float(pct) > 30 and not is_all_missing

                # Expander label
                if is_all_missing:
                    exp_label = (f"🚫 **{col}** — 100% missing ({n_total} of {n_total}) — "
                                 f"[{role.replace('_',' ').title()}] — IMPUTATION NOT POSSIBLE")
                elif is_id:
                    exp_label = (f"⛔ **{col}** — {n_miss} missing ({pct}%) — "
                                 f"[{role.replace('_',' ').title()}] — NOT recommended for imputation")
                else:
                    exp_label = (f"{'⚠️' if high_miss else '📌'} **{col}** — "
                                 f"{n_miss} missing ({pct}%) — [{role.replace('_',' ').title()}]")

                with st.expander(exp_label, expanded=is_all_missing or (not is_id)):
                    col_a, col_b = st.columns([2, 3])
                    with col_a:
                        st.markdown(f"**Data Type:** `{vtype}`")
                        st.markdown(f"**Analytical Role:** `{role}`")
                        st.markdown(f"**Measurement Level:** `{cl.get('measurement_level','?')}`")
                        st.markdown(f"**Missing:** {n_miss} of {n_total} values ({pct}%)")

                        # ── Before-preview: explicit labels when all missing ──
                        st.markdown("**Current values — Working Copy (first 10):**")
                        if is_all_missing:
                            import pandas as _pd
                            blank_preview = _pd.DataFrame(
                                {"Row": list(range(1, 11)),
                                 f"{col} — Value": ["⚠ Missing"] * 10}
                            )
                            st.table(blank_preview)
                        else:
                            st.table(df[[col]].head(10).reset_index())

                    with col_b:
                        # ══════════════════════════════════════════════════════
                        # CASE 1: 100% missing — hard block, no imputation UI
                        # ══════════════════════════════════════════════════════
                        if is_all_missing:
                            # Detect import/type-conversion error:
                            # raw_df has values but working_df shows all-missing
                            raw_df = st.session_state.get("raw_df")
                            is_conversion_error = (
                                raw_df is not None
                                and col in raw_df.columns
                                and int(raw_df[col].notna().sum()) > 0
                            )

                            if is_conversion_error:
                                raw_n_obs = int(raw_df[col].notna().sum())
                                st.error(
                                    f"🔴 **DATA IMPORT / TYPE-CONVERSION ERROR**\n\n"
                                    f"The **original uploaded file** has **{raw_n_obs} non-missing values** "
                                    f"in `{col}`, but the Working Copy shows **0 non-missing values (100% missing)**.\n\n"
                                    f"This means values were lost during data import or type conversion — "
                                    f"they were NOT missing in your source data. "
                                    f"Imputing this column would fabricate data to replace values that "
                                    f"already exist in your source file.\n\n"
                                    f"**Action required:** Re-upload your dataset or check the column's "
                                    f"data type parsing settings."
                                )
                            else:
                                st.error(
                                    f"🚫 **Imputation is IMPOSSIBLE for this column.**\n\n"
                                    f"**All {n_total} values are missing (100%).** "
                                    f"There is no observed data in this column to base any "
                                    f"statistical estimate on.\n\n"
                                    f"Mean, median, mode, and KNN imputation all require at least "
                                    f"some observed values. Filling 100% missing values would "
                                    f"fabricate the entire column from nothing — this is not "
                                    f"imputation, it is invention."
                                )

                            with st.expander("❓ Why can't I impute a 100%-missing column?"):
                                st.markdown(rec_why or (
                                    "Imputation estimates values using the distribution of **observed** "
                                    "values in the same column (or similar rows for KNN). "
                                    "When no values are observed, there is no distribution to "
                                    "estimate from — any value generated would have zero empirical "
                                    "basis and would misrepresent your dataset."
                                ))
                                st.warning(rec_warning or (
                                    "Do NOT impute this column. Investigate why all values are missing: "
                                    "check for import errors, structural skip patterns, or whether "
                                    "this column belongs to a different data collection instrument."
                                ))

                            st.markdown("**After-preview:** Not available — no values to estimate from.")

                            # Permanently disabled Apply button
                            st.button(
                                f"▶ Apply to Working Copy — [BLOCKED]",
                                key=f"apply_imp_{col}",
                                type="primary",
                                disabled=True,
                                help="Imputation is not possible when all values are missing."
                            )
                            continue   # skip remaining UI for this column

                        # ══════════════════════════════════════════════════════
                        # CASE 2: ID / do-not-impute hard block
                        # ══════════════════════════════════════════════════════
                        if is_id:
                            st.error(
                                f"⛔ **Statistical imputation is BLOCKED for this column.**\n\n"
                                f"Reason: {rec_reason}\n\n"
                                f"Imputing an identifier or serial number does not recover "
                                f"original information — it fabricates values that have no "
                                f"research meaning and will corrupt any analysis that uses this column."
                            )
                            with st.expander("❓ Why is imputation not appropriate here?"):
                                st.markdown(rec_why)
                                if rec_warning:
                                    st.warning(rec_warning)
                            st.info("If you believe the column role is mis-classified, "
                                    "go to **Variable Classification** and override the role first.")
                            force_action = st.checkbox(
                                "⚠️ I have re-checked the role and accept full responsibility — "
                                "show me options anyway",
                                key=f"force_{col}", value=False
                            )
                            if not force_action:
                                continue

                        # ══════════════════════════════════════════════════════
                        # CASE 3: Normal partial-missing — standard imputation UI
                        # ══════════════════════════════════════════════════════

                        # ── High-missingness special warning ───────────────
                        if high_miss:
                            st.markdown(
                                f'<div class="warn-box"><strong>⚠️ High Missingness Warning — {pct}% missing</strong><br>'
                                f'More than 30% of values in <code>{col}</code> are missing. '
                                f'Any imputation at this level is <strong>replacing a large portion of your data with estimates or synthetic values</strong>, '
                                f'not recovering original information. '
                                f'This can severely distort frequencies, percentages, correlations, '
                                f'statistical tests, and research conclusions. '
                                f'Consider whether this variable should be included in analysis at all.</div>',
                                unsafe_allow_html=True
                            )

                        st.markdown(f"**💡 System recommendation:** `{rec_method if not is_id else 'none'}`")
                        st.markdown(f"**Reason:** {rec_reason}")

                        if rec_why:
                            with st.expander("❓ Why is this method recommended?"):
                                st.markdown(rec_why)
                                if rec_warning:
                                    st.warning(rec_warning)

                        # Method selector
                        if is_id:
                            method_options = ["constant_unknown", "drop_rows"]
                        elif role in ("categorical", "boolean", "ordinal", "free_text"):
                            method_options = ["mode", "constant_unknown", "drop_rows"]
                        elif role == "datetime":
                            method_options = ["forward_fill", "drop_rows", "constant_unknown"]
                        else:
                            method_options = ["mean", "median", "mode", "knn",
                                              "constant_unknown", "drop_rows"]

                        safe_default = (rec_method if rec_method in method_options
                                        else method_options[0])

                        method = st.selectbox(
                            "Choose treatment method",
                            options=method_options,
                            index=method_options.index(safe_default),
                            key=f"imp_{col}",
                            help=(
                                "mean: Replaces missing values with column mean. Reduces variance. MCAR assumption.\n"
                                "median: Replaces with median. Robust to skew. Reduces variance.\n"
                                "mode: Replaces with most frequent value. Inflates that category.\n"
                                "knn: Estimates from similar rows. Reduces variance. Assumes similar observations exist.\n"
                                "constant_unknown: Creates a new 'Unknown' category — not data recovery.\n"
                                "forward_fill: Borrows from previous row. Assumes temporal order.\n"
                                "drop_rows: Permanently deletes rows — may introduce selection bias."
                            )
                        )

                        # Method-specific bias warnings
                        bias_map = {
                            "mean":             "Reduces variance; weakens correlations and statistical tests. Assumes MCAR.",
                            "median":           "Reduces variance. Best for skewed numerical data if MCAR assumed.",
                            "mode":             "Inflates the most frequent category. Distorts frequency tables and chi-square.",
                            "knn":              "Estimated values — reduces variance, may distort regression and scatter plots.",
                            "constant_unknown": "Creates new synthetic 'Unknown' category. Increases N for that category artificially.",
                            "forward_fill":     "Borrows adjacent values — assumes temporal ordering. May introduce autocorrelation.",
                            "drop_rows":        "Deletes entire rows. If missing is not random (MNAR), selection bias is introduced.",
                        }
                        appropriate_map = {
                            "mean":   role in ("substantive_numerical",),
                            "median": role in ("substantive_numerical",),
                            "mode":   role in ("categorical", "boolean", "ordinal"),
                            "knn":    role in ("substantive_numerical",),
                            "constant_unknown": None,  # always researcher decision
                            "forward_fill": role in ("datetime",),
                            "drop_rows": None,
                        }

                        # Live preview
                        try:
                            import pandas as _pd
                            preview_df, prev_log = impute_column(df, col, method)
                            if prev_log.get("ok", True):
                                st.markdown("**Preview — how values will change (first 10 rows):**")
                                before_vals = df[[col]].head(10).rename(columns={col: f"{col} — BEFORE"})
                                after_vals  = preview_df[[col]].head(10).rename(columns={col: f"{col} — AFTER (estimated/replaced)"})
                                st.table(_pd.concat([before_vals.reset_index(drop=True),
                                                     after_vals.reset_index(drop=True)], axis=1))
                            else:
                                st.warning(f"Preview not available: {prev_log['detail']}")
                        except Exception as _prev_err:
                            st.warning(f"Preview could not be generated: {_prev_err}")

                        # Impact disclosure
                        what_changes = (
                            f"'{col}': {n_miss} missing values will be "
                            + ("deleted (entire rows removed)" if method == "drop_rows"
                               else f"replaced with {method}-based estimates — original values are LOST")
                        )
                        what_lost = (
                            f"{n_miss} original missing-value positions. "
                            + ("Entire rows and all their other column values." if method == "drop_rows"
                               else "The fact that these values were missing. Downstream analyses treat replacements as real observed data.")
                        )
                        _impact_disclosure(
                            what_changes=what_changes,
                            what_lost=what_lost,
                            bias_risk=bias_map.get(method, ""),
                            rows_affected=n_miss,
                            rows_total=n_total,
                            reversible=True,
                            is_appropriate=appropriate_map.get(method, None)
                        )

                        # Researcher confirmation checkbox
                        confirmed = st.checkbox(
                            f"✅ I have read the impact disclosure above and confirm I want to apply "
                            f"**{method}** to '{col}' on the Working Copy.",
                            key=f"confirm_imp_{col}", value=False
                        )

                        if st.button(
                            f"▶ Apply to Working Copy — {method} on '{col}'",
                            key=f"apply_imp_{col}",
                            type="primary",
                            disabled=not confirmed
                        ):
                            _save_snapshot()
                            new_df, log_entry = impute_column(
                                st.session_state.working_df, col, method)
                            log_entry["researcher_confirmed"] = True
                            if log_entry.get("ok", True):
                                st.session_state.working_df = new_df
                                st.session_state.cleaning_log.append(log_entry)
                                st.session_state.profile = _profile(st.session_state.working_df)
                                profile = st.session_state.profile
                                st.success(
                                    f"✅ Working Copy updated — {log_entry['values_changed']} value(s) "
                                    f"{('deleted (rows removed)' if method == 'drop_rows' else 'replaced/estimated')} "
                                    f"in '{col}'. Original data unchanged. Undo is available."
                                )
                            else:
                                st.session_state.cleaning_log.append(log_entry)
                                st.error(f"❌ Operation failed: {log_entry['detail']}")
                            st.rerun()

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # B. Duplicate Rows
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("#### B. Duplicate Rows")
        n_dups = profile.get("duplicate_rows", df.duplicated().sum())
        n_total_b = len(df)
        pct_dups = round(n_dups / max(n_total_b, 1) * 100, 1)
        if n_dups == 0:
            st.success("✅ No duplicate rows detected in the current working copy.")
        else:
            st.warning(
                f"⚠️ **{n_dups} fully duplicate row(s)** detected ({pct_dups}% of {n_total_b:,} rows). "
                f"These are rows where every column value is identical to another row."
            )
            st.markdown("**Duplicate rows (showing all copies, up to 20):**")
            dup_df = df[df.duplicated(keep=False)]
            st.table(dup_df.head(20).reset_index(drop=True))

            _impact_disclosure(
                what_changes=f"Remove {n_dups} duplicate rows from the Working Copy.",
                what_lost=(f"{n_dups} rows permanently deleted from the working copy ({pct_dups}% of data). "
                           f"If these rows represent valid repeated observations "
                           f"(e.g. longitudinal data, time-series), their removal changes your sample."),
                bias_risk="Duplicate removal is appropriate only when duplicates are data entry errors. "
                          "If each row represents a separate event/time point, removal is methodologically incorrect.",
                rows_affected=n_dups,
                rows_total=n_total_b,
                reversible=True,
                is_appropriate=None
            )

            dup_confirmed = st.checkbox(
                f"✅ I confirm these are data entry duplicates, not valid repeated observations, "
                f"and I want to remove all {n_dups} from the Working Copy.",
                key="confirm_dups", value=False
            )
            if st.button(
                f"▶ Apply to Working Copy — Remove {n_dups} Duplicate Row(s)",
                type="primary", key="apply_dups",
                disabled=not dup_confirmed
            ):
                _save_snapshot()
                new_df, log_entry = remove_duplicates(st.session_state.working_df)
                log_entry["researcher_confirmed"] = True
                if log_entry.get("ok", True):
                    st.session_state.working_df = new_df
                    st.session_state.cleaning_log.append(log_entry)
                    st.session_state.profile = _profile(st.session_state.working_df)
                    st.success(
                        f"✅ Working Copy updated — {log_entry['values_changed']} duplicate row(s) "
                        f"deleted. {len(new_df):,} rows remain. Original data unchanged. Undo available."
                    )
                else:
                    st.session_state.cleaning_log.append(log_entry)
                    st.error(f"❌ Operation failed: {log_entry['detail']}")
                st.rerun()

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # C. Inconsistent Category Names
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("#### C. Inconsistent Category Names")
        incons = profile.get("inconsistencies", {})
        if not incons:
            st.success("✅ No inconsistent category names detected.")
        else:
            for col, groups in incons.items():
                n_total_c = len(df)
                with st.expander(f"📌 **{col}** — inconsistencies detected", expanded=True):
                    st.markdown(
                        f"The following values in `{col}` appear to be the same concept "
                        f"but are written differently. **Review carefully** — do not merge "
                        f"categories that represent genuinely different values."
                    )
                    # Show original category distribution
                    orig_counts = df[col].value_counts().head(20)
                    st.markdown("**Original category distribution (Working Copy):**")
                    orig_tbl = orig_counts.reset_index()
                    orig_tbl.columns = ["Category Value", "Count"]
                    st.table(orig_tbl)

                    mapping = {}
                    for norm_key, variants in groups.items():
                        if len(variants) > 1:
                            st.markdown(f"**Variant group:** `{variants}`")
                            std = st.text_input(
                                "Standardise all these variants to:",
                                value=variants[0],
                                key=f"std_{col}_{norm_key}",
                            )
                            for v in variants:
                                mapping[v] = std

                    if mapping:
                        mapping_preview = "\n".join(f"- `{k}` → `{v}`" for k, v in mapping.items())
                        affected_n = int(sum(
                            (df[col].astype(str) == str(k)).sum()
                            for k in mapping
                        ))
                        _impact_disclosure(
                            what_changes=(f"Recode {affected_n} values in '{col}'. "
                                          f"Mapping to apply:\n{mapping_preview}"),
                            what_lost=(f"Original category labels for {affected_n} values will be "
                                       f"permanently replaced in the Working Copy. "
                                       f"Granularity is reduced if distinct categories are merged."),
                            bias_risk="Category merging reduces granularity. "
                                      "Chi-square tests and frequency tables will change. "
                                      "Sub-group differences may be masked.",
                            rows_affected=affected_n,
                            rows_total=n_total_c,
                            reversible=True,
                            is_appropriate=None
                        )
                        std_confirmed = st.checkbox(
                            f"✅ I have verified the mapping above is correct and want to recode '{col}' "
                            f"in the Working Copy.",
                            key=f"confirm_std_{col}", value=False
                        )
                        if st.button(
                            f"▶ Apply to Working Copy — Recode '{col}'",
                            key=f"apply_std_{col}", type="primary",
                            disabled=not std_confirmed
                        ):
                            _save_snapshot()
                            new_df, log_entry = standardise_categories(
                                st.session_state.working_df, col, mapping)
                            log_entry["researcher_confirmed"] = True
                            if log_entry.get("ok", True):
                                st.session_state.working_df = new_df
                                st.session_state.cleaning_log.append(log_entry)
                                st.session_state.profile = _profile(st.session_state.working_df)
                                st.success(
                                    f"✅ Working Copy updated — {log_entry['values_changed']} value(s) "
                                    f"recoded in '{col}'. Original data unchanged. Undo available."
                                )
                            else:
                                st.session_state.cleaning_log.append(log_entry)
                                st.error(f"❌ Operation failed: {log_entry['detail']}")
                            st.rerun()

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # D. Outlier Detection & Capping
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("#### D. Outlier Detection")
        num_cols = profile.get("numerical_cols", [])
        if not num_cols:
            st.info("No numerical columns found for outlier analysis.")
        else:
            any_outliers = any(
                profile["col_stats"].get(col, {}).get("outliers", {}).get("count", 0) > 0
                for col in num_cols
            )
            if not any_outliers:
                st.success("✅ No potential outliers detected in any numerical variable (IQR method).")
            for col in num_cols:
                cs = profile["col_stats"].get(col, {})
                outlier_info = cs.get("outliers", {})
                n_out = outlier_info.get("count", 0)
                n_total_d = len(df)
                if n_out > 0:
                    pct_out = round(n_out / max(n_total_d, 1) * 100, 1)
                    with st.expander(
                        f"📌 **{col}** — {n_out} potential outlier(s) ({pct_out}%)",
                        expanded=False
                    ):
                        st.markdown(f"**IQR Bounds:** [{outlier_info['lower_bound']}, {outlier_info['upper_bound']}]")
                        st.markdown(
                            f"**Q1:** {outlier_info['q1']}  |  "
                            f"**Q3:** {outlier_info['q3']}  |  "
                            f"**IQR:** {outlier_info['iqr']}"
                        )
                        st.markdown(
                            f"**Potential outlier values (first 20):** `{outlier_info['outlier_values']}`"
                        )
                        st.markdown(
                            '<div class="warn-box"><strong>⚠️ Researcher Decision Required</strong><br>'
                            'Outliers detected by IQR are <em>statistically extreme</em> — '
                            'they are <strong>not automatically errors</strong>. '
                            'They may represent genuine high/low performers, rare events, or important findings. '
                            'Only cap or remove outliers if you have a substantive reason to believe '
                            'they are data entry errors or measurement mistakes.</div>',
                            unsafe_allow_html=True
                        )

                        action = st.selectbox(
                            "Action",
                            ["Leave as-is (recommended — review values first)",
                             "Cap using IQR method (Winsorization — replace with boundary values)",
                             "Cap using Z-score method (± 3 SD — replace with boundary values)"],
                            key=f"out_{col}"
                        )
                        if "Cap" in action:
                            out_method = "iqr" if "IQR" in action else "zscore"
                            _impact_disclosure(
                                what_changes=(
                                    f"{n_out} values in '{col}' will be REPLACED with the "
                                    f"{'IQR' if out_method == 'iqr' else 'Z-score ±3SD'} boundary value. "
                                    f"They are not deleted — their values are artificially clipped."
                                ),
                                what_lost=(
                                    f"True extreme values in '{col}' ({pct_out}% of observations). "
                                    f"The real range of the variable will be artificially compressed."
                                ),
                                bias_risk=(
                                    "Winsorization reduces variance and compresses the distribution tails. "
                                    "Mean, SD, correlations, and regression coefficients will all change. "
                                    "If these are genuine observations, capping introduces artificial homogeneity."
                                ),
                                rows_affected=n_out,
                                rows_total=n_total_d,
                                reversible=True,
                                is_appropriate=None
                            )
                            out_confirmed = st.checkbox(
                                f"✅ I have verified that these {n_out} values are data errors (not genuine observations) "
                                f"and I want to cap outliers in '{col}' in the Working Copy.",
                                key=f"confirm_out_{col}", value=False
                            )
                            if st.button(
                                f"▶ Apply to Working Copy — Cap Outliers in '{col}'",
                                key=f"apply_out_{col}",
                                type="primary",
                                disabled=not out_confirmed
                            ):
                                _save_snapshot()
                                new_df, log_entry = cap_outliers(
                                    st.session_state.working_df, col, out_method)
                                log_entry["researcher_confirmed"] = True
                                if log_entry.get("ok", True):
                                    st.session_state.working_df = new_df
                                    st.session_state.cleaning_log.append(log_entry)
                                    st.session_state.profile = _profile(st.session_state.working_df)
                                    st.success(
                                        f"✅ Working Copy updated — {log_entry['values_changed']} value(s) "
                                        f"capped (Winsorized) in '{col}'. Original data unchanged. Undo available."
                                    )
                                else:
                                    st.session_state.cleaning_log.append(log_entry)
                                    st.error(f"❌ Operation failed: {log_entry['detail']}")
                                st.rerun()

        st.markdown("---")

        # ══════════════════════════════════════════════════════════════════════
        # E. Fix Data Types
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("#### E. Fix Data Types")
        st.markdown(
            '<div class="info-box">ℹ️ Use this section if a variable was stored in the wrong format '
            '(e.g. dates saved as text, or numbers as text). '
            '<strong>Invalid values that cannot be converted will become NaN (missing) — '
            'they are not deleted.</strong> This introduces new missing values into the Working Copy.</div>',
            unsafe_allow_html=True
        )
        fix_col    = st.selectbox("Select variable to convert", all_cols, key="fix_type_col")
        fix_target = st.selectbox(
            "Convert to type",
            ["numerical", "datetime", "categorical"],
            key="fix_type_target",
            help=(
                "numerical: parses text numbers to numeric. Non-parseable values become NaN.\n"
                "datetime: parses date/time strings. Non-parseable values become NaT.\n"
                "categorical: converts to text/string. Numerical precision is lost."
            )
        )
        current_type  = profile.get("variable_types", {}).get(fix_col, "unknown")
        current_dtype = str(df[fix_col].dtype) if fix_col in df.columns else "unknown"
        st.caption(f"Currently detected type: `{current_type}` | pandas dtype: `{current_dtype}`")

        # Preview how many coercion failures there will be
        coerce_fails = 0
        if fix_col in df.columns:
            try:
                if fix_target == "numerical":
                    test_s = pd.to_numeric(df[fix_col], errors="coerce")
                    coerce_fails = int(test_s.isnull().sum()) - int(df[fix_col].isnull().sum())
                elif fix_target == "datetime":
                    test_s = pd.to_datetime(df[fix_col], errors="coerce")
                    coerce_fails = int(test_s.isnull().sum()) - int(df[fix_col].isnull().sum())
            except Exception:
                coerce_fails = 0

        _impact_disclosure(
            what_changes=(
                f"'{fix_col}' will be converted from {current_dtype} to {fix_target}. "
                + (f"{coerce_fails} value(s) that cannot be parsed will become NaN/NaT (new missing values)."
                   if coerce_fails > 0 else "No coercion failures predicted.")
            ),
            what_lost=(
                f"Original {current_dtype} representations of '{fix_col}' values will be permanently "
                f"replaced in the Working Copy."
                + (f" Additionally, {coerce_fails} value(s) will be converted to missing (NaN/NaT) "
                   f"and their original content is lost."
                   if coerce_fails > 0 else "")
            ),
            bias_risk=(
                f"Converting to {fix_target} introduces {coerce_fails} new missing value(s). "
                f"These will affect completeness statistics and may need to be treated in Section A."
            ) if coerce_fails > 0 else "",
            rows_affected=coerce_fails if coerce_fails > 0 else 0,
            rows_total=len(df),
            reversible=True,
            is_appropriate=(True if current_type != fix_target else None)
        )

        type_confirmed = st.checkbox(
            f"✅ I confirm I want to convert '{fix_col}' from {current_dtype} to {fix_target} "
            f"in the Working Copy{(' — and accept that ' + str(coerce_fails) + ' value(s) will become NaN.') if coerce_fails > 0 else '.'}",
            key="confirm_fix_type", value=False
        )
        if st.button(
            f"▶ Apply to Working Copy — Convert '{fix_col}' to {fix_target}",
            type="primary", key="apply_fix_type",
            disabled=not type_confirmed
        ):
            _save_snapshot()
            new_df, log_entry = fix_data_types(
                st.session_state.working_df, fix_col, fix_target)
            log_entry["researcher_confirmed"] = True
            if log_entry.get("ok", True):
                st.session_state.working_df = new_df
                st.session_state.cleaning_log.append(log_entry)
                st.session_state.profile = _profile(st.session_state.working_df)
                st.success(
                    f"✅ Working Copy updated — '{fix_col}' converted to {fix_target}. "
                    f"{log_entry['detail']} Original data unchanged. Undo available."
                )
            else:
                st.session_state.cleaning_log.append(log_entry)
                st.error(f"❌ Operation failed: {log_entry['detail']}")
            st.rerun()

    with log_tab:
        st.markdown("### Audit Log — Data Cleaning Operations")
        st.markdown(
            '<div class="info-box">ℹ️ <strong>This log records every operation applied to the Working Copy.</strong> '
            'It distinguishes between original data, cleaned data, imputed data, transformed data, '
            'and researcher-confirmed data. '
            '"Researcher-confirmed" means you explicitly ticked the confirmation checkbox before applying. '
            'This log is your research transparency record.</div>',
            unsafe_allow_html=True
        )
        log = st.session_state.cleaning_log
        if not log:
            st.info("No cleaning operations have been applied to the Working Copy yet.")
        else:
            # Data-state breakdown
            state_counts = {}
            for e in log:
                s = e.get("data_state", "working_copy")
                state_counts[s] = state_counts.get(s, 0) + 1
            state_summary = "  |  ".join(
                f"**{s.replace('_',' ').title()}**: {n}"
                for s, n in state_counts.items()
            )
            st.markdown(f"**Operations by data state:** {state_summary}")
            if any(e.get("data_state") in ("imputed", "transformed") for e in log):
                st.warning(
                    "⚠️ Your Working Copy contains IMPUTED and/or TRANSFORMED values. "
                    "When reporting results, clearly distinguish which variables contain "
                    "estimated or recoded values. Never describe imputed data as 'original' data."
                )
            st.markdown(get_cleaning_summary(log))
            st.markdown("---")
            log_df = pd.DataFrame(log)
            # Show key columns prominently
            key_cols = ["timestamp", "operation", "column", "method", "values_changed",
                        "rows_affected", "rows_total", "data_state",
                        "researcher_confirmed", "bias_risk", "detail"]
            display_cols = [c for c in key_cols if c in log_df.columns]
            st.table(log_df[display_cols].reset_index(drop=True))

            buf = io.BytesIO()
            log_df.to_excel(buf, index=False)
            st.download_button(
                "📥 Download Full Audit Log (Excel)",
                data=buf.getvalue(),
                file_name="data_cleaning_audit_log.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Statistical" in page:
    _require_data()
    st.markdown('<div class="section-title">📊 Statistical Analysis</div>', unsafe_allow_html=True)

    # ── Pre-flight import check — surface engine errors early and clearly ──────
    # All statistical functions are imported here once. Tab-level code uses these
    # directly. This avoids repeated per-tab ImportErrors and ensures that a
    # configuration problem shows a single clear message instead of crashing
    # individual tabs unpredictably.
    _stats_engine_ok = True
    try:
        from engine.statistics import (
            descriptive_stats, descriptive_stats_extended, frequency_table, crosstab,
            pearson_correlation, spearman_correlation, kendall_correlation,
            correlation_matrix, recommend_correlation_method,
            chi_square_test, chi_square_goodness_of_fit, chi_square_expected_check,
            fishers_exact_test,
            point_biserial_correlation,
            independent_ttest, paired_ttest,
            one_way_anova, welch_anova_test,
            linear_regression, logistic_regression,
            cronbach_alpha,
            mann_whitney_u_test, wilcoxon_signed_rank_test,
            kruskal_wallis_test, friedman_test,
            normality_check, levene_test,
            bonferroni_correction, holm_bonferroni_correction,
            pca_analysis,
            detect_numeric_stored_as_text,
        )
        from engine.interpreter import interpret_result
        from engine.recommender import recommend_statistical_method
        from engine.visualizer import correlation_heatmap as _corr_heatmap
        from engine.decision_engine import (
            run_decision_engine, build_context_from_profile,
            interpret_research_question,
        )
        from engine.stat_methods import METHOD_KB
    except ImportError as _stat_imp_err:
        _stats_engine_ok = False
        st.error(
            "⚠️ **Statistical module configuration error.**\n\n"
            "The application could not load one or more statistical analysis functions. "
            "This is usually caused by a missing dependency or a module version mismatch.\n\n"
            f"**Technical detail:** `{_stat_imp_err}`\n\n"
            "**Action:** Check that all requirements are installed (`pip install -r requirements.txt`), "
            "then restart the Streamlit server."
        )
        st.stop()

    df      = _get_df()
    profile = st.session_state.profile or {}

    # ── Role-aware column lists ────────────────────────────────────────────────
    _roles        = _get_role_cols(profile, df)
    subst_num_cols = _roles["subst_num_cols"]
    cat_cols       = _roles["cat_cols"]
    ordinal_cols   = _roles["ordinal_cols"]
    all_analysis_cols = _roles["all_analysis_cols"]
    blocked_cols   = _roles["blocked_cols"]

    # Fall back to profile lists if role engine produced nothing
    # (e.g. old profile without classifications)
    if not subst_num_cols and not cat_cols:
        subst_num_cols = profile.get("analysis_ready_numerical", profile.get("numerical_cols", []))
        cat_cols       = profile.get("analysis_ready_categorical", profile.get("categorical_cols", []))
        ordinal_cols   = []

    # ── Blocked-columns banner ─────────────────────────────────────────────────
    if blocked_cols:
        with st.expander(
            f"🚫 {len(blocked_cols)} variable(s) excluded from analysis selectors — click to see why",
            expanded=False
        ):
            st.markdown(
                "The following columns have been **automatically excluded** from all analysis "
                "selectors because their variable role makes quantitative analysis inappropriate. "
                "If a classification is wrong, go to **Variable Classification** to override it."
            )
            rows_blocked = [
                {"Column": col, "Reason": reason}
                for col, reason in blocked_cols.items()
            ]
            st.table(pd.DataFrame(rows_blocked).reset_index(drop=True))
            st.info(
                "💡 To include a blocked variable, open **Variable Classification** in the sidebar, "
                "find the column, and set its role to an appropriate type "
                "(e.g. *Substantive Numerical* or *Categorical*)."
            )

    # ── Method recommendation — 16-step decision engine ───────────────────────
    with st.expander("🤔 Which statistical method should I use?", expanded=False):
        st.markdown("""
<div class="info-box">
<strong>How this works — 4 steps:</strong><br>
<strong>Step 1</strong> — Optionally describe your research question in plain language.<br>
<strong>Step 2 &amp; 3</strong> — Select the two variables you want to analyse. The system reads their classified type, measurement level, and data quality automatically.<br>
<strong>Step 4</strong> — The decision engine runs 16 internal checks and recommends the statistically appropriate method for <em>that specific combination</em>, with full rationale, assumption checks, alternatives, and effect-size guidance.<br>
The recommendation is based on variable roles and data properties — not on abstract labels you type in.
</div>
""", unsafe_allow_html=True)

        _avail_cols = [c for c in df.columns if c not in blocked_cols]
        _classifications_rec = profile.get("classifications", {})

        # Research question input
        _rq = st.text_input(
            "📝 Research question (optional — plain language)",
            key="rec_rq",
            placeholder='e.g. "Does income differ by sex?" or "Is age associated with household size?"',
            help="Describe what you are trying to find out. This helps the engine understand your intent."
        )

        if len(_avail_cols) < 2:
            st.warning("Need at least 2 analysis-ready variables. "
                       "Check Variable Classification to unblock columns.")
        else:
            _rc1, _rc2 = st.columns(2)

            # ── Variable A profile card ────────────────────────────────────
            with _rc1:
                st.markdown("**Variable A**")
                _col_a = st.selectbox("Select Variable A", _avail_cols, key="rec_var_a")
                _cl_a    = _classifications_rec.get(_col_a, {})
                _role_a  = _cl_a.get("analytical_role", "unknown")
                _level_a = _cl_a.get("measurement_level", "unknown")
                _miss_a  = float(profile.get("missing_pct", {}).get(_col_a, 0))
                _uniq_a  = int(profile.get("col_stats", {}).get(_col_a, {}).get("unique", 0))
                _n_valid_a = int(len(df) - profile.get("missing_counts", {}).get(_col_a, 0))
                _warn_a  = _cl_a.get("warnings", [])
                _miss_a_pct_flag = "🟡" if _miss_a > 10 else ("🔴" if _miss_a > 30 else "🟢")
                st.markdown(f"""
<div class="step-card">
🏷 <strong>Role:</strong> <code>{_role_a}</code><br>
📐 <strong>Level:</strong> <code>{_level_a}</code><br>
📊 <strong>Unique:</strong> {_uniq_a} &nbsp;|&nbsp; {_miss_a_pct_flag} <strong>Missing:</strong> {_miss_a:.1f}%<br>
✅ <strong>Valid obs:</strong> {_n_valid_a}
{f'<br>⚠️ <em style="font-size:11px">{_warn_a[0][:80]}…</em>' if _warn_a else ''}
</div>""", unsafe_allow_html=True)

            # ── Variable B profile card ────────────────────────────────────
            with _rc2:
                st.markdown("**Variable B**")
                _col_b_opts = [c for c in _avail_cols if c != _col_a]
                _col_b = st.selectbox("Select Variable B", _col_b_opts, key="rec_var_b")
                _cl_b    = _classifications_rec.get(_col_b, {})
                _role_b  = _cl_b.get("analytical_role", "unknown")
                _level_b = _cl_b.get("measurement_level", "unknown")
                _miss_b  = float(profile.get("missing_pct", {}).get(_col_b, 0))
                _uniq_b  = int(profile.get("col_stats", {}).get(_col_b, {}).get("unique", 0))
                _n_valid_b = int(len(df) - profile.get("missing_counts", {}).get(_col_b, 0))
                _warn_b  = _cl_b.get("warnings", [])
                _n_groups_engine = int(df[_col_b].nunique()) if _col_b in df.columns else 0
                _miss_b_pct_flag = "🟡" if _miss_b > 10 else ("🔴" if _miss_b > 30 else "🟢")
                st.markdown(f"""
<div class="step-card">
🏷 <strong>Role:</strong> <code>{_role_b}</code><br>
📐 <strong>Level:</strong> <code>{_level_b}</code><br>
📊 <strong>Unique:</strong> {_uniq_b} &nbsp;|&nbsp; {_miss_b_pct_flag} <strong>Missing:</strong> {_miss_b:.1f}%<br>
✅ <strong>Valid obs:</strong> {_n_valid_b}
{f'<br>⚠️ <em style="font-size:11px">{_warn_b[0][:80]}…</em>' if _warn_b else ''}
</div>""", unsafe_allow_html=True)

            _paired_design = st.checkbox(
                "Paired / repeated measurements (same subjects measured twice)",
                key="rec_paired", value=False
            )

            if st.button("🔍 Run Decision Engine (16-step analysis)", type="primary", key="rec_pair_btn"):
                # Build context and run engine
                _n_pairs = int(df[[_col_a, _col_b]].dropna().shape[0])
                _ctx = build_context_from_profile(
                    var_a=_col_a, var_b=_col_b, profile=profile, df=df
                )
                _ctx.paired = _paired_design
                _ctx.research_question = _rq.strip()

                # Research question intent parse
                if _rq.strip():
                    _rq_intent = interpret_research_question(_rq, profile)
                    st.markdown(f"""
<div style="background:#e8f5e9;border:1px solid #4caf50;border-radius:6px;padding:10px 14px;margin-bottom:10px;font-size:13px;">
📝 <strong>Research question parsed:</strong>
Intent = <strong>{_rq_intent.get('intent','?')}</strong>
| Confidence = {_rq_intent.get('confidence',0):.0%}
{f"| Outcome hint: <em>{_rq_intent.get('outcome_hint','')}</em>" if _rq_intent.get('outcome_hint') else ''}
{f"| Predictor hint: <em>{_rq_intent.get('predictor_hint','')}</em>" if _rq_intent.get('predictor_hint') else ''}
</div>
""", unsafe_allow_html=True)

                _n_prev_analyses = len(st.session_state.get("analysis_history", []))
                _dec = run_decision_engine(_ctx, df=df, n_tests_in_session=max(1, _n_prev_analyses))

                if _dec.blocked:
                    st.error(f"🚫 **Analysis blocked:** {_dec.block_reason}")
                else:
                    # ── Primary recommendation ─────────────────────────────
                    _meth_meta = METHOD_KB.get(_dec.primary_method)
                    st.markdown(f"""
<div class="result-box">
<div style="font-size:17px;font-weight:800;color:#0d47a1;margin-bottom:10px;">
    ✅ Recommended Method: {_meth_meta.name if _meth_meta else _dec.primary_method}
</div>
<p><strong>Why this method for {_col_a} + {_col_b}:</strong><br>{_dec.primary_rationale}</p>
</div>
""", unsafe_allow_html=True)

                    # ── Simple / Advanced tabs ─────────────────────────────
                    _st1, _st2 = st.tabs(["📖 Simple Explanation", "🔬 Advanced Details"])
                    with _st1:
                        if _meth_meta:
                            st.markdown(f"**{_meth_meta.name}** — {_meth_meta.purpose}")
                            st.markdown(f"**Effect size:** {', '.join(_meth_meta.effect_size_measures) if _meth_meta.effect_size_measures else 'N/A'}")
                            st.markdown(f"**95% CI available:** {'Yes' if _meth_meta.ci_available else 'No / requires bootstrap'}")
                    with _st2:
                        st.markdown("**Assumptions to verify:**")
                        for _a in _dec.primary_assumptions:
                            st.markdown(f"- {_a}")
                        if _dec.effect_size_note:
                            st.markdown(f"**Effect size guidance:** {_dec.effect_size_note}")
                        if _dec.ci_note:
                            st.markdown(f"**Confidence interval:** {_dec.ci_note}")

                    # ── Assumption checks ──────────────────────────────────
                    _checks = _dec.assumption_check_results
                    if _checks:
                        st.markdown("#### 🔎 Automated Assumption Checks")
                        for _chk in _checks:
                            _status_icon = {"ok": "✅", "warn": "⚠️", "fail": "❌"}.get(_chk.get("status","warn"), "⚠️")
                            _sev_color   = {"low": "#d4edda", "medium": "#fff3cd", "high": "#f8d7da"}.get(_chk.get("severity","medium"), "#fff3cd")
                            st.markdown(f"""
<div style="background:{_sev_color};border-radius:5px;padding:9px 14px;margin:5px 0;font-size:13px;">
{_status_icon} <strong>{_chk.get('assumption','')}</strong><br>
{_chk.get('detail','')}<br>
<em>Implication: {_chk.get('implication','')}</em><br>
{f'<strong>Suggestion:</strong> {_chk.get("suggestion","")}' if _chk.get("suggestion") else ''}
</div>""", unsafe_allow_html=True)

                    # ── Alternatives ───────────────────────────────────────
                    if _dec.alternatives:
                        with st.expander(f"📚 {len(_dec.alternatives)} alternative method(s)", expanded=False):
                            for _alt in _dec.alternatives:
                                _alt_meta = METHOD_KB.get(_alt.get("method_key", ""))
                                st.markdown(f"""
**{_alt_meta.name if _alt_meta else _alt.get('method_key','')}**
— {_alt.get('why_alternative','')}

*When to prefer this:* {_alt.get('when_prefer','')}
""")

                    # ── Data quality warnings ──────────────────────────────
                    _dq = _dec.data_quality_warnings
                    if _dq:
                        warn_box("**Data quality issues for this pair:**<br>" + "<br>".join(f"• {w}" for w in _dq))

                    # ── Multiple testing note ──────────────────────────────
                    if _dec.multiple_testing_note and _n_prev_analyses >= 2:
                        st.markdown(f"""
<div class="warn-box">📊 <strong>Multiple Testing:</strong> {_dec.multiple_testing_note}</div>
""", unsafe_allow_html=True)

                    # ── Step log (for transparency) ────────────────────────
                    with st.expander("🔍 View 16-step decision log", expanded=False):
                        for _i, _step in enumerate(_dec.step_log, 1):
                            st.markdown(f"**Step {_i}:** {_step}")

                    # ── Limitations ────────────────────────────────────────
                    if _dec.limitations:
                        with st.expander("⚠️ Limitations to acknowledge", expanded=False):
                            for _lim in _dec.limitations:
                                st.markdown(f"- {_lim}")

    st.markdown("---")

    # ── Multiple testing banner (session-level) ────────────────────────────────
    _n_analyses_run = len(st.session_state.get("analysis_history", []))
    if _n_analyses_run >= 3:
        _fp_prob = round((1 - 0.95 ** _n_analyses_run) * 100, 1)
        _alpha_corr = round(0.05 / _n_analyses_run, 4)
        st.warning(
            f"⚠️ **Multiple Testing Alert:** You have run **{_n_analyses_run}** statistical tests in this session. "
            f"At α = 0.05, the probability of at least one false positive is approximately **{_fp_prob}%**. "
            f"Consider Bonferroni correction: α* = 0.05 ÷ {_n_analyses_run} = **{_alpha_corr}**. "
            "Use the Bonferroni utility in the 🔢 Non-Parametric tab."
        )

    # ── Numeric-as-text warning banner ────────────────────────────────────────
    _nat_result = detect_numeric_stored_as_text(df)
    if _nat_result["n_flagged"] > 0:
        with st.expander(
            f"⚠️ {_nat_result['n_flagged']} column(s) may contain numeric values stored as text — click to review",
            expanded=False
        ):
            st.warning(_nat_result["recommendation"])
            for _nat_col in _nat_result["flagged_columns"]:
                _safe_icon = "✅" if _nat_col["safe_to_convert"] else "⚠️"
                st.markdown(
                    f"{_safe_icon} **`{_nat_col['column']}`** — "
                    f"{_nat_col['pct_numeric_looking']}% of non-missing values look numeric "
                    f"({_nat_col['n_numeric_looking']} of {_nat_col['n_total'] - _nat_col['n_missing']}). "
                    f"Sample: `{_nat_col['sample_values'][:3]}`. "
                    f"{'Safe to convert.' if _nat_col['safe_to_convert'] else _nat_col['note']}"
                )
            st.caption(
                "Do NOT convert here automatically. Go to **Data Cleaning → Recode** "
                "to handle type conversion after reviewing each column."
            )

    # Analysis tabs
    tab_desc, tab_freq, tab_cross, tab_corr, tab_ttest, tab_anova, tab_chi2, tab_reg, tab_alpha, tab_nonpar, tab_adv = st.tabs([
        "📋 Descriptive", "📊 Frequency", "🔲 Cross-tab",
        "🔗 Correlation", "⚖️ t-Test", "🏗 ANOVA", "🔲 Chi-Square", "📈 Regression", "🔁 Reliability",
        "🔢 Non-Parametric", "🔬 Advanced"
    ])

    # ── Descriptive Statistics ─────────────────────────────────────────────────
    with tab_desc:
        st.markdown("#### Descriptive Statistics")
        info_box(
            "Descriptive statistics summarise the central tendency, spread, and shape of your variables. "
            "Numerical variables show mean/SD/skewness; categorical variables show frequency counts."
        )

        # ── Numerical sub-section ──────────────────────────────────────────────
        st.markdown("##### 🔢 Numerical Variables")
        if not subst_num_cols:
            st.info("No substantive numerical columns available. "
                    "If numerical columns exist but are blocked, check Variable Classification.")
        else:
            sel_num = st.multiselect(
                "Select numerical variables",
                subst_num_cols,
                default=subst_num_cols[:min(4, len(subst_num_cols))],
                key="desc_num_sel"
            )
            # Allow researcher to include a blocked variable with a warning
            include_blocked = st.checkbox(
                "Include a blocked variable (not recommended — use only if classification is wrong)",
                value=False, key="desc_include_blocked"
            )
            if include_blocked and blocked_cols:
                extra_col = st.selectbox(
                    "Select blocked variable to include",
                    list(blocked_cols.keys()), key="desc_blocked_extra"
                )
                reason = blocked_cols.get(extra_col, "")
                st.warning(
                    f"⚠️ **{extra_col}** has been included despite being classified as blocked. "
                    f"Reason for exclusion: {reason}. "
                    "Interpret any results involving this variable with extreme caution."
                )
                if extra_col not in sel_num:
                    sel_num = sel_num + [extra_col]

            _desc_mode = st.radio(
                "Table type",
                ["Standard (mean, SD, quartiles)", "Extended (+ CV, P10, P90, variance, IQR)"],
                key="desc_mode", horizontal=True
            )
            if sel_num and st.button("Run Descriptive Statistics", type="primary", key="run_desc"):
                # Filter to only columns that can be coerced to float
                valid_num = []
                for c in sel_num:
                    try:
                        df[c].dropna().astype(float)
                        valid_num.append(c)
                    except (ValueError, TypeError):
                        st.warning(f"Column '{c}' could not be converted to numeric — skipped.")
                if valid_num:
                    result_df = (
                        descriptive_stats_extended(df, valid_num)
                        if "Extended" in _desc_mode
                        else descriptive_stats(df, valid_num)
                    )
                    n_used = int(result_df["N"].min()) if "N" in result_df.columns else "N/A"
                    st.caption(
                        f"N used: each variable computed on its own non-missing rows "
                        f"(minimum across selected variables: {n_used})."
                    )
                    st.table(result_df.reset_index(drop=True))

                    # Skewness interpretation note
                    if "Skewness" in result_df.columns:
                        skew_notes = []
                        for _, row in result_df.iterrows():
                            sk = row["Skewness"]
                            try:
                                sk = float(sk)
                            except (TypeError, ValueError):
                                continue
                            if abs(sk) < 0.5:
                                interp = "approximately symmetric"
                            elif abs(sk) < 1.0:
                                interp = "moderately skewed"
                            else:
                                interp = "highly skewed"
                            skew_notes.append(f"**{row['Variable']}**: skewness = {sk:.4f} ({interp})")
                        if skew_notes:
                            with st.expander("📐 Skewness interpretation", expanded=False):
                                for note in skew_notes:
                                    st.markdown(f"- {note}")
                                st.caption(
                                    "Rule of thumb: |skewness| < 0.5 = approximately symmetric, "
                                    "0.5–1.0 = moderate skew, > 1.0 = high skew. "
                                    "Highly skewed variables may violate normality assumptions "
                                    "for t-tests, ANOVA, and Pearson correlation."
                                )

                    buf = io.BytesIO()
                    result_df.to_excel(buf, index=False)
                    st.download_button(
                        "📥 Download Table", data=buf.getvalue(),
                        file_name="descriptive_statistics.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        # ── Categorical sub-section ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("##### 🏷️ Categorical Variables")
        if not cat_cols:
            st.info("No categorical columns available.")
        else:
            sel_cat = st.multiselect(
                "Select categorical variables for frequency summary",
                cat_cols,
                default=cat_cols[:min(3, len(cat_cols))],
                key="desc_cat_sel"
            )
            if sel_cat and st.button("Run Categorical Summary", type="primary", key="run_desc_cat"):
                for col in sel_cat:
                    st.markdown(f"**{col}**")
                    is_ord = col in ordinal_cols
                    ft = frequency_table(df, col, is_ordinal=is_ord)
                    n_missing = int(df[col].isna().sum()) + int((df[col] == "").sum())
                    st.caption(
                        f"N total rows: {len(df)} | Missing / blank: {n_missing}"
                        + (" | Ordinal variable — Cumulative % shown" if is_ord else "")
                    )
                    st.table(ft.reset_index(drop=True))
                    st.markdown("")

    # ── Frequency Table ────────────────────────────────────────────────────────
    with tab_freq:
        st.markdown("#### Frequency / Percentage Table")
        info_box(
            "Shows how often each value appears. "
            "% of Total uses all rows (including missing) as denominator. "
            "Valid % uses only non-missing rows as denominator."
        )

        # Offer all non-blocked columns (not just cat_cols — researchers may want freq on any var)
        freq_available = all_analysis_cols if all_analysis_cols else list(df.columns)
        if not freq_available:
            st.info("No columns available for frequency analysis.")
        else:
            col_sel = st.selectbox("Select variable", freq_available, key="freq_col")

            # Determine if ordinal
            classifications_now = profile.get("classifications", {})
            col_role = classifications_now.get(col_sel, {}).get("analytical_role", "unknown")
            # Honour override
            ovr_role = st.session_state.get("user_role_overrides", {}).get(col_sel, {}).get("analytical_role")
            if ovr_role:
                col_role = ovr_role
            is_ord = col_role in ("ordinal",)

            if is_ord:
                st.info("ℹ️ This variable is classified as **Ordinal** — Cumulative % will be included.")

            if st.button("Generate Frequency Table", type="primary", key="run_freq"):
                ft = frequency_table(df, col_sel, is_ordinal=is_ord)
                n_missing = int(df[col_sel].isna().sum()) + int((df[col_sel] == "").sum())
                n_total   = len(df)
                n_valid   = n_total - n_missing

                st.caption(
                    f"**Denominator note:** "
                    f"% of Total = frequency ÷ {n_total} rows (all rows, including missing). "
                    f"Valid % = frequency ÷ {n_valid} non-missing rows."
                )
                st.table(ft.reset_index(drop=True))

                # Near-duplicate category detector (case-insensitive)
                raw_vals = df[col_sel].dropna().astype(str)
                raw_vals = raw_vals[raw_vals != ""]
                lower_map: dict = {}
                for v in raw_vals:
                    key = v.strip().lower()
                    lower_map.setdefault(key, [])
                    if v not in lower_map[key]:
                        lower_map[key].append(v)
                near_dupes = {k: vs for k, vs in lower_map.items() if len(vs) > 1}
                if near_dupes:
                    lines = []
                    for variants in near_dupes.values():
                        counts = {v: int((raw_vals == v).sum()) for v in variants}
                        parts  = ", ".join(f'"{v}" (n={c})' for v, c in counts.items())
                        lines.append(f"- {parts}")
                    warn_box(
                        "**⚠️ Possible capitalisation variants detected** — the following values "
                        "appear to be the same category written differently. They are shown separately "
                        "in the table above. Do NOT merge here; use **Data Cleaning → Recode Values** "
                        "if these should be unified.\n\n" + "\n".join(lines)
                    )

                buf = io.BytesIO()
                ft.to_excel(buf, index=False)
                st.download_button(
                    "📥 Download", data=buf.getvalue(),
                    file_name=f"frequency_{col_sel}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    # ── Cross-Tabulation ───────────────────────────────────────────────────────
    with tab_cross:
        st.markdown("#### Cross-Tabulation")
        info_box(
            "Shows how two categorical/ordinal variables are distributed together. "
            "Frequency, Row %, and Column % are calculated with correct denominators. "
            "An optional Chi-Square test of independence is available after generating the table."
        )
        if len(cat_cols) < 2:
            st.info(
                "Need at least 2 categorical or ordinal columns for cross-tabulation. "
                f"Currently available: {len(cat_cols)}. "
                "Check **Variable Classification** if expected columns are missing."
            )
        else:
            c1, c2 = st.columns(2)
            row_col = c1.selectbox("Row variable", cat_cols, key="xt_row")
            col_col = c2.selectbox(
                "Column variable",
                [c for c in cat_cols if c != row_col],
                key="xt_col"
            )

            # Near-duplicate detection for both variables
            for _xt_col, _xt_label in [(row_col, "Row variable"), (col_col, "Column variable")]:
                _raw = df[_xt_col].dropna().astype(str)
                _raw = _raw[_raw.str.strip() != ""]
                _lmap: dict = {}
                for _v in _raw:
                    _k = _v.strip().lower()
                    _lmap.setdefault(_k, [])
                    if _v not in _lmap[_k]:
                        _lmap[_k].append(_v)
                _nd = {k: vs for k, vs in _lmap.items() if len(vs) > 1}
                if _nd:
                    _parts = "; ".join(
                        " / ".join(f'"{v}"(n={int((_raw == v).sum())})' for v in vs)
                        for vs in _nd.values()
                    )
                    st.warning(
                        f"⚠️ **{_xt_label} '{_xt_col}'** — possible capitalisation variants detected: "
                        f"{_parts}. These appear as separate categories below. "
                        "If they should be merged, use **Data Cleaning → Recode Values** first."
                    )

            if st.button("Generate Cross-Tabulation", type="primary", key="run_xt"):
                xt = crosstab(df, row_col, col_col)

                # Missing note
                n_miss_row = int(df[row_col].isna().sum()) + int((df[row_col] == "").sum())
                n_miss_col = int(df[col_col].isna().sum()) + int((df[col_col] == "").sum())
                if n_miss_row > 0 or n_miss_col > 0:
                    st.caption(
                        f"Note: **{row_col}** has {n_miss_row} missing/blank values, "
                        f"**{col_col}** has {n_miss_col} missing/blank values. "
                        "These are labelled '⚠ Missing / Blank' in the tables below."
                    )

                t1, t2, t3 = st.tabs(["Frequency", "Row %", "Column %"])
                with t1:
                    st.caption(f"N total = {len(df)} rows.")
                    st.table(xt["freq"])
                with t2:
                    st.caption(
                        "Row %: each cell = (cell count ÷ row total) × 100. "
                        "Rows sum to 100%."
                    )
                    st.table(xt["row_pct"])
                with t3:
                    st.caption(
                        "Column %: each cell = (cell count ÷ column total) × 100. "
                        "Columns sum to 100%."
                    )
                    st.table(xt["col_pct"])

                # ── Optional Chi-Square from Cross-tab ─────────────────────────
                st.markdown("---")
                st.markdown("##### Optional: Chi-Square Test of Independence")
                st.caption(
                    "Tests whether the two variables are statistically associated or independent. "
                    "Only appropriate when expected cell frequencies are ≥ 5 in most cells."
                )
                run_chi_from_xt = st.checkbox(
                    f"Run Chi-Square test on '{row_col}' × '{col_col}'",
                    key="run_chi_from_xt", value=False
                )
                if run_chi_from_xt:
                    # Use only rows where BOTH variables are non-missing
                    chi_df = df[[row_col, col_col]].dropna()
                    chi_df = chi_df[(chi_df[row_col] != "") & (chi_df[col_col] != "")]
                    n_used_chi = len(chi_df)
                    n_excluded = len(df) - n_used_chi
                    if n_excluded > 0:
                        st.caption(
                            f"Chi-Square uses {n_used_chi} complete rows "
                            f"({n_excluded} rows excluded due to missing values in either variable)."
                        )
                    chi_result = chi_square_test(chi_df, row_col, col_col)
                    if "error" not in chi_result:
                        if chi_result.get("warning"):
                            warn_box(chi_result["warning"])
                        st.markdown(
                            f"**χ²({chi_result['df']}) = {chi_result['chi2']:.4f}**, "
                            f"p = {chi_result['p_value']:.6f}, "
                            f"N = {chi_result['n']}, "
                            f"Cramér's V = {chi_result['cramers_v']:.4f}"
                        )
                        v = chi_result['cramers_v']
                        p = chi_result['p_value']
                        effect = "strong" if v >= 0.5 else "moderate" if v >= 0.3 else "weak" if v >= 0.1 else "negligible"
                        sig_txt = "statistically significant" if p < 0.05 else "not statistically significant"
                        st.markdown(
                            f"**Statistical interpretation:** The association between '{row_col}' and '{col_col}' "
                            f"is **{sig_txt}** (p = {p:.6f})."
                        )
                        st.markdown(
                            f"**Plain language:** The data provide "
                            f"{'evidence of an association' if p < 0.05 else 'no statistically significant evidence of an association'} "
                            f"between '{row_col}' and '{col_col}'. "
                            f"Effect size (Cramér's V = {v:.4f}) indicates a **{effect}** association. "
                            f"This result does not imply that one variable causes the other."
                        )
                    else:
                        st.error(f"Chi-Square could not be computed: {chi_result['error']}")

    # ── Correlation ────────────────────────────────────────────────────────────
    with tab_corr:
        st.markdown("#### Correlation Analysis")
        info_box(
            "Measures the strength and direction of association between two numerical variables. "
            "Choose the method based on variable type and distribution — Pearson for continuous/normal, "
            "Spearman for ordinal/skewed, Kendall for ordinal/small samples with ties."
        )
        if len(subst_num_cols) < 2:
            st.info(
                "Need at least 2 substantive numerical columns for correlation. "
                f"Currently available: {len(subst_num_cols)}. "
                "If columns are missing, check **Variable Classification** to override their role."
            )
        else:
            c1, c2 = st.columns(2)
            v1 = c1.selectbox("Variable 1", subst_num_cols, key="cor_v1")
            v2_opts = [c for c in subst_num_cols if c != v1]
            v2 = c2.selectbox("Variable 2", v2_opts if v2_opts else subst_num_cols, key="cor_v2")

            # ── Auto-recommendation ────────────────────────────────────────────
            if v1 != v2:
                rec_info = recommend_correlation_method(df, v1, v2)
                rec_method = rec_info.get("recommended", "Pearson") or "Pearson"
                rec_reason = rec_info.get("reason", "")
                rec_warns  = rec_info.get("warnings", [])

                st.markdown(
                    f'<div style="background:#e0f7fa;border:1px solid #5ba4c4;border-left:5px solid #1565c0;'
                    f'border-radius:6px;padding:10px 14px;margin:8px 0;font-size:13px;color:#0d1b2a;">'
                    f'💡 <strong>Recommended method: {rec_method}</strong><br>'
                    f'{rec_reason}'
                    f'{"<br><ul style=\"margin:6px 0 0 14px\">" + "".join(f"<li style=\"color:#92400e\">{w}</li>" for w in rec_warns) + "</ul>" if rec_warns else ""}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                method_opts = ["Pearson", "Spearman", "Kendall"]
                default_idx = method_opts.index(rec_method) if rec_method in method_opts else 0
                method = st.selectbox(
                    "Method",
                    method_opts,
                    index=default_idx,
                    key="cor_meth",
                    help=(
                        "Pearson: linear association, requires continuous & approximately normal variables.\n"
                        "Spearman: rank-based, suitable for ordinal/skewed data or outliers.\n"
                        "Kendall: rank-based, best for small samples or many tied ranks."
                    )
                )
            else:
                method = "Pearson"

            # Assumption panel
            assumptions_map = {
                "Pearson": [
                    "Both variables should be continuous (interval or ratio scale).",
                    "Both variables should be approximately normally distributed.",
                    "The relationship should be approximately linear.",
                    "Outliers strongly distort Pearson — use Spearman if outliers are present.",
                ],
                "Spearman": [
                    "Variables must be at least ordinal.",
                    "Does not assume normality — suitable for skewed distributions.",
                    "Measures monotonic (not necessarily linear) association.",
                    "More robust to outliers than Pearson.",
                ],
                "Kendall": [
                    "Variables must be at least ordinal.",
                    "Best for small samples or data with many tied ranks.",
                    "Tau values are typically smaller than Spearman rho for the same data — this is expected.",
                    "No normality assumption required.",
                ],
            }
            _method_assumptions(
                method_name=f"{method} Correlation",
                variables=[v1, v2],
                roles_ok=True,
                issues=[],
                assumptions=assumptions_map.get(method, [])
            )

            if st.button("Run Correlation", type="primary", key="run_corr"):
                if method == "Pearson":
                    result = pearson_correlation(df, v1, v2)
                elif method == "Spearman":
                    result = spearman_correlation(df, v1, v2)
                else:
                    result = kendall_correlation(df, v1, v2)

                if "error" not in result:
                    st.session_state.analysis_history.append(result)
                    st.session_state.last_result = result
                    n_used = result.get("n", "N/A")
                    ci = result.get("ci_95")

                    # Structured result display
                    st.markdown(f"**N used** (complete pairs, missing excluded): **{n_used}**")
                    if ci and ci[0] is not None:
                        st.markdown(f"**95% Confidence Interval** (Fisher z): [{ci[0]:.4f}, {ci[1]:.4f}]")
                    elif method == "Kendall":
                        st.caption("95% CI not shown for Kendall (requires bootstrap).")

                    st.markdown(result.get("interpretation", ""))
                    st.caption(f"**Assumption note:** {result.get('assumption_note', '')}")
                else:
                    st.error(result["error"])

            # ── Point-Biserial Correlation ──────────────────────────────────
            st.markdown("---")
            st.markdown("##### Point-Biserial Correlation (continuous × binary)")
            st.caption(
                "Use when you have one continuous variable and one binary (2-category) variable. "
                "Mathematically equivalent to Pearson r applied to a 0/1 coded binary variable."
            )
            if not subst_num_cols or not cat_cols:
                st.info("Need at least one numerical and one categorical (binary) variable.")
            else:
                _pb_c1, _pb_c2 = st.columns(2)
                _pb_cont = _pb_c1.selectbox("Continuous variable", subst_num_cols, key="pb_cont")
                _pb_bin  = _pb_c2.selectbox("Binary variable (2 groups only)", cat_cols, key="pb_bin")
                _pb_n_grp = int(df[_pb_bin].dropna().nunique()) if _pb_bin in df.columns else 0
                if _pb_n_grp != 2:
                    st.warning(f"⚠️ '{_pb_bin}' has {_pb_n_grp} unique values — point-biserial requires exactly 2.")
                if st.button("Run Point-Biserial Correlation", type="primary", key="run_pb"):
                    _pb_result = point_biserial_correlation(df, _pb_cont, _pb_bin)
                    if "error" not in _pb_result:
                        st.session_state.analysis_history.append(_pb_result)
                        st.session_state.last_result = _pb_result
                        st.caption(f"N used (complete pairs): {_pb_result.get('n', 'N/A')}")
                        ci_pb = _pb_result.get("ci_95")
                        if ci_pb and ci_pb[0] is not None:
                            st.markdown(f"**95% CI (Fisher z):** [{ci_pb[0]:.4f}, {ci_pb[1]:.4f}]")
                        st.markdown(interpret_result(_pb_result))
                    else:
                        st.error(_pb_result["error"])

            st.markdown("---")
            st.markdown("##### Correlation Matrix (multiple variables)")
            col_matrix = st.multiselect(
                "Select variables for correlation matrix",
                subst_num_cols, default=[], key="cor_matrix"
            )
            matrix_method = st.selectbox(
                "Matrix method", ["pearson", "spearman", "kendall"],
                key="cor_matrix_method",
                format_func=str.title
            )
            if col_matrix and len(col_matrix) >= 2 and st.button("Generate Correlation Matrix", key="run_matrix"):
                n_matrix = int(df[col_matrix].apply(pd.to_numeric, errors="coerce").dropna().shape[0])
                st.markdown(
                    f"N used (listwise — all {len(col_matrix)} variables present): **{n_matrix}**  "
                    f"| Method: **{matrix_method.title()}**"
                )
                corr_mat = correlation_matrix(df, col_matrix, method=matrix_method)
                st.table(corr_mat)
                fig = _corr_heatmap(corr_mat)
                st.plotly_chart(fig, use_container_width=True)
                warn_box("Correlation does not establish causation.")

    # ── t-Test ─────────────────────────────────────────────────────────────────
    with tab_ttest:
        st.markdown("#### t-Test")
        info_box("Compares the mean of a numerical variable between groups.")

        ttest_type = st.radio(
            "Test type",
            ["Independent samples (two different groups)",
             "Paired samples (same subjects, two conditions)"],
            key="tt_type", horizontal=True
        )
        c1, c2 = st.columns(2)

        if "Independent" in ttest_type:
            num_v = c1.selectbox("Numerical variable", subst_num_cols or ["— no numerical cols —"], key="tt_num")
            grp_v = c2.selectbox(
                "Grouping variable (must have exactly 2 groups)",
                cat_cols or ["— no categorical cols —"], key="tt_grp"
            )

            # Issues check
            tt_issues = []
            if not subst_num_cols:
                tt_issues.append("No substantive numerical variables available.")
            if not cat_cols:
                tt_issues.append("No categorical grouping variables available.")
            elif grp_v in cat_cols:
                n_grps = int(df[grp_v].dropna().nunique())
                if n_grps != 2:
                    tt_issues.append(
                        f"Grouping variable '{grp_v}' has {n_grps} unique groups "
                        f"— independent t-test requires exactly 2."
                    )

            _method_assumptions(
                method_name="Independent Samples t-Test",
                variables=[num_v, grp_v] if subst_num_cols and cat_cols else [],
                roles_ok=not tt_issues,
                issues=tt_issues,
                assumptions=[
                    "Numerical variable should be approximately normally distributed within each group.",
                    "Groups must be independent (different subjects).",
                    "Equal variance assumption tested via Levene's test (Welch correction applied if violated).",
                    "Dependent variable must be continuous (interval or ratio scale).",
                ]
            )

            # ── Pre-test assumption checks ─────────────────────────────────
            if subst_num_cols and cat_cols and not tt_issues:
                with st.expander("🔬 Pre-test assumption checks (Normality + Levene)", expanded=False):
                    _nc = normality_check(df[num_v].dropna(), num_v)
                    if "error" not in _nc:
                        _nc_icon = "✅" if _nc.get("is_normal") else "⚠️"
                        st.markdown(f"{_nc_icon} **Normality ({num_v}):** {_nc.get('test_used','')} "
                                    f"p = {_nc.get('p_value','?')} | Skewness = {_nc.get('skewness','?')} "
                                    f"— {_nc.get('interpretation','')}")
                        if not _nc.get("is_normal"):
                            st.caption(_nc.get("recommendation",""))
                    _lev = levene_test(df, num_v, grp_v)
                    if "error" not in _lev:
                        _lev_icon = "✅" if _lev.get("equal_variance") else "⚠️"
                        st.markdown(f"{_lev_icon} **Levene's test (equal variance):** p = {_lev.get('p_value','?')} "
                                    f"— {_lev.get('interpretation','')}")
                        if not _lev.get("equal_variance"):
                            st.caption(_lev.get("recommendation",""))

            if st.button("Run Independent t-Test", type="primary", key="run_tt"):
                result = independent_ttest(df, num_v, grp_v)
                if "error" not in result:
                    n1, n2 = result.get("n1", "?"), result.get("n2", "?")
                    g1_label, g2_label = result.get("group1", "Group 1"), result.get("group2", "Group 2")
                    st.caption(f"N used: {g1_label} = {n1}, {g2_label} = {n2} (total = {result.get('n', '?')})")
                    st.session_state.analysis_history.append(result)
                    st.session_state.last_result = result
                    st.markdown(interpret_result(result))
                    # Effect size CI
                    _d = result.get("cohens_d")
                    if _d is not None:
                        from engine.statistics import compute_effect_size_ci
                        _d_ci = compute_effect_size_ci("cohens_d", _d, result.get("n1",0), result.get("n2",0))
                        if _d_ci.get("ci_lower") is not None:
                            st.caption(f"Cohen's d = {_d:.4f} | 95% CI: [{_d_ci['ci_lower']:.4f}, {_d_ci['ci_upper']:.4f}]")
                else:
                    st.error(result["error"])
        else:
            v1 = c1.selectbox("Measurement 1", subst_num_cols or ["— no numerical cols —"], key="pt_v1")
            v2 = c2.selectbox(
                "Measurement 2",
                [c for c in subst_num_cols if c != v1] or ["— no numerical cols —"],
                key="pt_v2"
            )

            _method_assumptions(
                method_name="Paired Samples t-Test",
                variables=[v1, v2] if subst_num_cols else [],
                roles_ok=len(subst_num_cols) >= 2,
                issues=(["Need at least 2 substantive numerical columns."] if len(subst_num_cols) < 2 else []),
                assumptions=[
                    "Each pair of measurements must come from the same subject.",
                    "Differences between paired measurements should be approximately normally distributed.",
                    "Both variables must be continuous (interval or ratio scale).",
                ]
            )

            if st.button("Run Paired t-Test", type="primary", key="run_pt"):
                result = paired_ttest(df, v1, v2)
                if "error" not in result:
                    st.caption(f"N used (complete pairs): {result.get('n', 'N/A')}")
                    st.session_state.analysis_history.append(result)
                    st.session_state.last_result = result
                    st.markdown(interpret_result(result))
                else:
                    st.error(result["error"])

    # ── ANOVA ──────────────────────────────────────────────────────────────────
    with tab_anova:
        st.markdown("#### One-Way ANOVA / Welch ANOVA")
        info_box(
            "Compares the mean of a numerical variable across three or more independent groups. "
            "Use ANOVA instead of multiple t-tests to control the false positive rate. "
            "Use **Welch ANOVA** when Levene's test is significant (unequal variances)."
        )
        _anova_variant = st.radio(
            "ANOVA variant",
            ["Standard ANOVA (assumes equal variance)", "Welch ANOVA (does not assume equal variance)"],
            key="anova_variant", horizontal=True
        )
        if not subst_num_cols or not cat_cols:
            st.info(
                "Need at least one substantive numerical variable and one categorical grouping variable. "
                f"Currently: {len(subst_num_cols)} numerical, {len(cat_cols)} categorical."
            )
        else:
            c1, c2 = st.columns(2)
            num_v = c1.selectbox("Numerical (outcome) variable", subst_num_cols, key="an_num")
            grp_v = c2.selectbox("Grouping variable", cat_cols, key="an_grp")

            # Issues check
            anova_issues = []
            n_grps_anova = int(df[grp_v].dropna().nunique())
            if n_grps_anova < 2:
                anova_issues.append(f"Grouping variable '{grp_v}' has fewer than 2 non-missing groups.")
            if n_grps_anova == 2:
                anova_issues.append(
                    f"'{grp_v}' has only 2 groups — consider using a t-Test instead of ANOVA."
                )

            _method_assumptions(
                method_name="One-Way ANOVA",
                variables=[num_v, grp_v],
                roles_ok=not anova_issues,
                issues=anova_issues,
                assumptions=[
                    "Observations within each group are independent.",
                    "Numerical variable should be approximately normally distributed within each group.",
                    "Homogeneity of variance across groups (Levene's test recommended).",
                    "Dependent variable must be continuous (interval or ratio scale).",
                    "Post-hoc Tukey HSD computed automatically if result is significant and groups > 2.",
                ]
            )

            # ── Pre-test: Normality per group + Levene ────────────────────
            if not anova_issues:
                with st.expander("🔬 Pre-test assumption checks (Normality + Levene)", expanded=False):
                    _lev_an = levene_test(df, num_v, grp_v)
                    if "error" not in _lev_an:
                        _lev_an_icon = "✅" if _lev_an.get("equal_variance") else "⚠️"
                        st.markdown(f"{_lev_an_icon} **Levene's test:** p = {_lev_an.get('p_value','?')} "
                                    f"— {_lev_an.get('interpretation','')}")
                        if not _lev_an.get("equal_variance"):
                            st.caption("⚠️ Variances differ significantly. Consider Welch ANOVA or Kruskal-Wallis (Non-Parametric tab).")
                    _nc_an = normality_check(df[num_v].dropna(), num_v)
                    if "error" not in _nc_an:
                        _nc_an_icon = "✅" if _nc_an.get("is_normal") else "⚠️"
                        st.markdown(f"{_nc_an_icon} **Normality ({num_v}):** {_nc_an.get('test_used','')} "
                                    f"p = {_nc_an.get('p_value','?')} — {_nc_an.get('interpretation','')}")
                        if not _nc_an.get("is_normal"):
                            st.caption(_nc_an.get("recommendation",""))

            _anova_btn_label = "Run Welch ANOVA" if "Welch" in _anova_variant else "Run One-Way ANOVA"
            if st.button(_anova_btn_label, type="primary", key="run_anova"):
                if "Welch" in _anova_variant:
                    result = welch_anova_test(df, num_v, grp_v)
                else:
                    result = one_way_anova(df, num_v, grp_v)
                if "error" not in result:
                    # N per group summary
                    gs = result.get("group_summary", {})
                    if gs:
                        gs_rows = [{"Group": g, "N": d.get("n","?"),
                                    "Mean": d.get("mean","?"), "SD": d.get("std","?")}
                                   for g, d in gs.items()]
                        st.caption(f"N used (total): {result.get('n', 'N/A')}")
                        st.markdown("**Group summary:**")
                        st.table(pd.DataFrame(gs_rows).reset_index(drop=True))
                    st.session_state.analysis_history.append(result)
                    st.session_state.last_result = result
                    st.markdown(interpret_result(result))
                    if result.get("posthoc_tukey") is not None:
                        st.markdown("**Post-hoc: Tukey HSD**")
                        st.table(result["posthoc_tukey"].reset_index(drop=True))
                    elif result.get("posthoc_note"):
                        info_box(result["posthoc_note"])
                    elif result.get("significant") and result.get("n_groups", 0) > 2:
                        info_box(
                            "Post-hoc test could not be computed. "
                            "Consider running pairwise t-tests with Bonferroni correction."
                        )
                else:
                    st.error(result["error"])

    # ── Chi-Square ─────────────────────────────────────────────────────────────
    with tab_chi2:
        st.markdown("#### Chi-Square & Fisher's Exact Tests")
        info_box(
            "Tests whether two categorical variables are statistically associated or independent. "
            "Use **Fisher's Exact Test** when expected cell counts are small (any cell < 5 in a 2×2 table). "
            "Use **Goodness-of-Fit** to test whether one variable follows a theoretical distribution."
        )
        _chi_test_type = st.radio(
            "Test type",
            ["Chi-Square Test of Independence", "Fisher's Exact Test (2×2 only)", "Chi-Square Goodness-of-Fit (one variable)"],
            key="chi_test_type", horizontal=False
        )
        if len(cat_cols) < 2:
            st.info(
                "Need at least 2 categorical or ordinal variables. "
                f"Currently available: {len(cat_cols)}."
            )
        # ── Goodness-of-Fit (single variable — no need for cat_cols ≥ 2) ─────
        if "Goodness-of-Fit" in _chi_test_type:
            _gof_avail = all_analysis_cols if all_analysis_cols else list(df.columns)
            if not _gof_avail:
                st.info("No columns available.")
            else:
                _gof_col = st.selectbox("Select variable", _gof_avail, key="gof_col")
                st.info("Goodness-of-fit tests whether observed frequencies match a theoretical distribution. "
                        "By default, the uniform (equal proportions) distribution is tested.")
                if st.button("Run Chi-Square Goodness-of-Fit", type="primary", key="run_gof"):
                    _gof_result = chi_square_goodness_of_fit(df, _gof_col)
                    if "error" not in _gof_result:
                        st.caption(f"N used: {_gof_result.get('n', 'N/A')} of {_gof_result.get('n_total','N/A')}")
                        if _gof_result.get("warning"):
                            warn_box(_gof_result["warning"])
                        st.session_state.analysis_history.append(_gof_result)
                        st.session_state.last_result = _gof_result
                        st.markdown(interpret_result(_gof_result))
                        _gof_cat_tbl = _gof_result.get("category_table")
                        if _gof_cat_tbl is not None:
                            st.markdown("**Category-level observed vs. expected:**")
                            st.table(_gof_cat_tbl.reset_index(drop=True))
                    else:
                        st.error(_gof_result["error"])
        elif len(cat_cols) < 2:
            st.info(
                "Need at least 2 categorical or ordinal variables. "
                f"Currently available: {len(cat_cols)}."
            )
        else:
            c1, c2 = st.columns(2)
            v1 = c1.selectbox("Categorical variable 1", cat_cols, key="chi_v1")
            v2 = c2.selectbox(
                "Categorical variable 2",
                [c for c in cat_cols if c != v1], key="chi_v2"
            )

            if "Fisher" in _chi_test_type:
                _method_assumptions(
                    method_name="Fisher's Exact Test",
                    variables=[v1, v2],
                    roles_ok=True,
                    issues=[],
                    assumptions=[
                        "Both variables must be binary (exactly 2 categories each) — 2×2 table required.",
                        "Observations must be independent.",
                        "Fixed marginal totals.",
                        "Use when any expected cell count < 5 in a 2×2 table.",
                    ]
                )
                if st.button("Run Fisher's Exact Test", type="primary", key="run_fisher"):
                    _fisher_result = fishers_exact_test(df, v1, v2)
                    if "error" not in _fisher_result:
                        st.caption(f"N used: {_fisher_result.get('n', 'N/A')} of {_fisher_result.get('n_total','N/A')}")
                        ct = _fisher_result.pop("contingency_table", None)
                        st.session_state.analysis_history.append(_fisher_result)
                        st.session_state.last_result = _fisher_result
                        st.markdown(interpret_result(_fisher_result))
                        if ct is not None:
                            st.markdown("**Contingency Table (observed counts):**")
                            st.table(ct)
                    else:
                        st.error(_fisher_result["error"])
            else:
                # Standard chi-square independence
                _method_assumptions(
                    method_name="Chi-Square Test of Independence",
                    variables=[v1, v2],
                    roles_ok=True,
                    issues=[],
                    assumptions=[
                        "Observations must be independent (one row per subject).",
                        "Both variables must be categorical (nominal or ordinal).",
                        "Expected cell frequency ≥ 5 in at least 80% of cells (warning shown if violated).",
                        "Chi-square tests association only — it does not indicate direction or causation.",
                    ]
                )

                # ── Pre-test: Expected cell check ────────────────────────
                with st.expander("🔬 Pre-test: Expected cell frequency check", expanded=False):
                    _exp_chk = chi_square_expected_check(df, v1, v2)
                    if "error" not in _exp_chk:
                        _exp_icon = "✅" if _exp_chk.get("is_chi_square_appropriate") else "❌"
                        st.markdown(
                            f"{_exp_icon} **{_exp_chk.get('n_cells_below_5',0)} of {_exp_chk.get('n_cells',0)} cells** "
                            f"({_exp_chk.get('pct_cells_below_5',0):.1f}%) have expected frequency < 5. "
                            f"Min expected = {_exp_chk.get('min_expected',0):.2f}."
                        )
                        if not _exp_chk.get("is_chi_square_appropriate"):
                            st.warning(_exp_chk.get("warning_message",""))
                            if _exp_chk.get("recommend_fisher"):
                                st.info("ℹ️ This is a 2×2 table — consider switching to **Fisher's Exact Test** above.")

                if st.button("Run Chi-Square Test", type="primary", key="run_chi2"):
                    result = chi_square_test(df, v1, v2)
                    if "error" not in result:
                        n_miss_v1 = int(df[v1].isna().sum())
                        n_miss_v2 = int(df[v2].isna().sum())
                        if n_miss_v1 > 0 or n_miss_v2 > 0:
                            st.caption(
                                f"Note: rows with missing values are excluded from the test. "
                                f"Missing: {v1} = {n_miss_v1}, {v2} = {n_miss_v2}."
                            )
                        st.caption(f"N used (complete cases): {result.get('n', 'N/A')}")
                        if result.get("warning"):
                            warn_box(result["warning"])
                        ct = result.pop("contingency_table", None)
                        st.session_state.analysis_history.append(result)
                        st.session_state.last_result = result
                        st.markdown(interpret_result(result))
                        if ct is not None:
                            st.markdown("**Contingency Table (observed counts):**")
                            st.table(ct)
                    else:
                        st.error(result["error"])

    # ── Regression ─────────────────────────────────────────────────────────────
    with tab_reg:
        st.markdown("#### Regression Analysis")
        info_box("Examines how a dependent variable is predicted by one or more independent variables.")
        reg_type = st.radio(
            "Regression type",
            ["Linear Regression (numerical outcome)",
             "Logistic Regression (binary categorical outcome)"],
            key="reg_type", horizontal=True
        )

        if "Linear" in reg_type:
            dep    = st.selectbox(
                "Dependent (outcome) variable",
                subst_num_cols or ["— no numerical cols —"], key="lr_dep"
            )
            indeps = st.multiselect(
                "Independent (predictor) variable(s)",
                [c for c in subst_num_cols if c != dep], key="lr_ind"
            )

            reg_issues = []
            if not subst_num_cols:
                reg_issues.append("No substantive numerical variables available.")
            if indeps and len(df[[dep] + indeps].dropna()) < len(indeps) + 2:
                reg_issues.append(
                    "Insufficient complete cases for regression with the selected predictors."
                )

            _method_assumptions(
                method_name="Linear Regression (OLS)",
                variables=[dep] + indeps,
                roles_ok=not reg_issues,
                issues=reg_issues,
                assumptions=[
                    "Linear relationship between each predictor and the outcome.",
                    "Independence of observations (one row per subject).",
                    "Homoscedasticity: residuals should have constant variance.",
                    "Normality of residuals (not of the raw variables).",
                    "No severe multicollinearity among predictors (check VIF if many predictors).",
                    "Identifiers must never be used as predictors.",
                ]
            )

            if indeps and st.button("Run Linear Regression", type="primary", key="run_lr"):
                result = linear_regression(df, dep, indeps)
                if "error" not in result:
                    st.caption(f"N used (complete cases): {result.get('n', 'N/A')}")
                    coefs = result.pop("coefficients", None)
                    st.session_state.analysis_history.append(result)
                    st.session_state.last_result = result
                    st.markdown(interpret_result(result))
                    if coefs is not None:
                        st.markdown("**Regression Coefficients:**")
                        st.table(coefs.reset_index(drop=True))
                else:
                    st.error(result["error"])
        else:
            # Logistic regression: dependent can be binary cat or binary num
            logit_dep_opts = cat_cols + subst_num_cols
            dep    = st.selectbox(
                "Dependent variable (must be binary: 0/1 or two categories)",
                logit_dep_opts or ["— no columns available —"],
                key="logit_dep"
            )
            indeps = st.multiselect(
                "Independent (predictor) variable(s)",
                subst_num_cols, key="logit_ind"
            )

            _method_assumptions(
                method_name="Logistic Regression",
                variables=[dep] + indeps,
                roles_ok=True,
                issues=[],
                assumptions=[
                    "Dependent variable must be binary (exactly 2 categories).",
                    "Observations must be independent.",
                    "No severe multicollinearity among predictors.",
                    "Sufficient sample size: recommended ≥ 10 events per predictor.",
                    "Predictors must be numerical or properly encoded categorical variables.",
                    "Identifiers must never be used as predictors.",
                ]
            )

            if indeps and st.button("Run Logistic Regression", type="primary", key="run_logit"):
                run_df = df.copy()
                if run_df[dep].dtype == object:
                    uniq = run_df[dep].dropna().unique()
                    if len(uniq) == 2:
                        run_df[dep] = (run_df[dep] == uniq[1]).astype(int)
                    else:
                        st.error("Logistic regression requires a binary dependent variable (exactly 2 categories).")
                        st.stop()
                result = logistic_regression(run_df, dep, indeps)
                if "error" not in result:
                    st.caption(f"N used (complete cases): {result.get('n', 'N/A')}")
                    coefs = result.pop("coefficients", None)
                    st.session_state.analysis_history.append(result)
                    st.session_state.last_result = result
                    st.markdown(interpret_result(result))
                    if coefs is not None:
                        st.markdown("**Coefficients and Odds Ratios:**")
                        st.table(coefs.reset_index(drop=True))
                else:
                    st.error(result["error"])

    # ── Cronbach's Alpha ────────────────────────────────────────────────────────
    with tab_alpha:
        st.markdown("#### Cronbach's Alpha — Reliability Analysis")
        info_box(
            "Cronbach's alpha measures the internal consistency (reliability) of a group of numerical "
            "survey items that are intended to measure the same construct. "
            "A value of α ≥ 0.70 is generally considered acceptable for research."
        )
        with st.expander("ℹ️ When to use Cronbach's Alpha?", expanded=False):
            st.markdown("""
            - You have a **Likert-scale survey** (e.g. 1–5 rating questions) where multiple questions
              measure the same underlying concept (e.g. job satisfaction, anxiety, customer loyalty).
            - You want to check whether your scale items are **consistent** with each other.
            - A high alpha (close to 1.0) means responses are correlated across items.
            - **Important:** Alpha measures consistency — it does NOT measure validity (whether you are
              measuring the right thing).
            """)

        if len(subst_num_cols) < 2:
            st.info(
                "Need at least 2 substantive numerical variables for reliability analysis. "
                f"Currently available: {len(subst_num_cols)}."
            )
        else:
            items_sel = st.multiselect(
                "Select the scale items (numerical variables representing the same construct)",
                subst_num_cols,
                default=[],
                key="alpha_items",
                help="Select 2 or more numerical variables that all measure the same concept."
            )

            _method_assumptions(
                method_name="Cronbach's Alpha",
                variables=items_sel,
                roles_ok=len(items_sel) >= 2,
                issues=(["Select at least 2 items to compute alpha."] if len(items_sel) < 2 else []),
                assumptions=[
                    "All items should measure the same underlying construct (unidimensionality).",
                    "Items should be numerical (Likert scale, rating scale, etc.).",
                    "Negatively worded items may need to be reverse-scored before computing alpha.",
                    "Alpha increases artificially with more items — interpret in context.",
                    "Identifiers and serial numbers must not be included as scale items.",
                ]
            )

            if len(items_sel) >= 2 and st.button("Run Cronbach's Alpha", type="primary", key="run_alpha"):
                result = cronbach_alpha(df, items_sel)
                if "error" not in result:
                    st.session_state.analysis_history.append(result)
                    st.session_state.last_result = result
                    st.caption(f"N used (complete cases across all items): {result.get('n', 'N/A')}")

                    a = result["alpha"]
                    colour = ("#1e8e5c" if a >= 0.7 else "#d4a017" if a >= 0.6 else "#c0392b")
                    st.markdown(f"""
                    <div class="result-box">
                        <div style="font-size:28px;font-weight:800;color:{colour};">
                            α = {a:.4f}
                        </div>
                        <div style="font-size:16px;font-weight:700;color:#1a2340;margin-top:4px;">
                            {result['reliability']}
                        </div>
                        <p style="margin-top:10px;">{result['interpretation']}</p>
                        <p><strong>Assumption note:</strong> {result['assumption_note']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if result.get("warning"):
                        warn_box(result["warning"])

                    st.markdown("**Item-Total Correlations** (each item vs. sum of remaining items):")
                    itc = result["item_total_corr"]
                    itc_df = pd.DataFrame(
                        [(item, corr, "Good" if corr >= 0.3 else "Low — consider reviewing")
                         for item, corr in itc.items()],
                        columns=["Item", "Corrected Item-Total Correlation", "Assessment"]
                    )
                    st.table(itc_df.reset_index(drop=True))
                    info_box(
                        "An item-total correlation below 0.30 suggests the item may not be measuring "
                        "the same construct as the others. Consider reviewing or removing that item."
                    )

                    buf = io.BytesIO()
                    itc_df.to_excel(buf, index=False)
                    st.download_button(
                        "📥 Download Reliability Results",
                        data=buf.getvalue(),
                        file_name="cronbach_alpha.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.error(result["error"])
            elif len(items_sel) < 2:
                st.info("Select at least 2 numerical items above to run the analysis.")

    # ── Non-Parametric Tests ───────────────────────────────────────────────────
    with tab_nonpar:
        st.markdown("#### 🔢 Non-Parametric & Assumption-Free Tests")
        info_box(
            "Non-parametric tests do not assume normally distributed data. "
            "They work on ranks rather than raw values, making them suitable for "
            "skewed data, ordinal variables, small samples, or when parametric assumptions are violated. "
            "They are not automatically 'better' — use them when the parametric alternative's assumptions cannot be met."
        )

        np_tab1, np_tab2, np_tab3, np_tab4, np_tab5 = st.tabs([
            "⚖️ Mann-Whitney U", "🔁 Wilcoxon Signed-Rank",
            "🏗 Kruskal-Wallis", "🔬 Normality Check", "📊 Bonferroni Correction"
        ])

        # ── Mann-Whitney U ─────────────────────────────────────────────────
        with np_tab1:
            st.markdown("##### Mann-Whitney U Test")
            st.markdown("""
**When to use:** Two independent groups; outcome is numerical but normality is violated, sample is small, or data is ordinal.
**Non-parametric equivalent of:** Independent Samples t-Test.
**Effect size:** Rank-biserial r (ranges −1 to +1, like correlation).
""")
            if not subst_num_cols or not cat_cols:
                st.info("Need at least one numerical and one categorical variable.")
            else:
                _mw_c1, _mw_c2 = st.columns(2)
                _mw_num = _mw_c1.selectbox("Numerical variable", subst_num_cols, key="mw_num")
                _mw_grp = _mw_c2.selectbox("Grouping variable (must have 2 groups)", cat_cols, key="mw_grp")
                _mw_n_grps = int(df[_mw_grp].dropna().nunique()) if _mw_grp in df.columns else 0
                if _mw_n_grps != 2:
                    st.warning(f"⚠️ '{_mw_grp}' has {_mw_n_grps} groups — Mann-Whitney U requires exactly 2.")
                if st.button("Run Mann-Whitney U Test", type="primary", key="run_mw"):
                    _mw_res = mann_whitney_u_test(df, _mw_num, _mw_grp)
                    if "error" not in _mw_res:
                        st.session_state.analysis_history.append(_mw_res)
                        st.session_state.last_result = _mw_res
                        st.caption(f"N used: {_mw_res.get('group1','G1')} = {_mw_res.get('n1','?')}, "
                                   f"{_mw_res.get('group2','G2')} = {_mw_res.get('n2','?')} "
                                   f"(total = {_mw_res.get('n','?')})")
                        st.markdown(f"""
<div class="result-box">
<div style="font-size:15px;font-weight:700;color:#0d47a1;">Mann-Whitney U Test</div>
<p><strong>U statistic:</strong> {_mw_res.get('u_statistic','?')} &nbsp;|&nbsp;
<strong>p-value:</strong> {_mw_res.get('p_value','?')} &nbsp;|&nbsp;
<strong>{'✅ Significant' if _mw_res.get('significant') else '❌ Not significant'}</strong></p>
<p><strong>Effect size (rank-biserial r):</strong> {_mw_res.get('rank_biserial_r','?')}</p>
<p>{_mw_res.get('interpretation','')}</p>
<p><em>{_mw_res.get('assumption_note','')}</em></p>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.error(_mw_res["error"])

        # ── Wilcoxon Signed-Rank ───────────────────────────────────────────
        with np_tab2:
            st.markdown("##### Wilcoxon Signed-Rank Test")
            st.markdown("""
**When to use:** Two paired/related measurements from the same subjects; differences are not normally distributed.
**Non-parametric equivalent of:** Paired Samples t-Test.
**Effect size:** Rank-biserial r.
""")
            if len(subst_num_cols) < 2:
                st.info("Need at least 2 numerical variables.")
            else:
                _wc_c1, _wc_c2 = st.columns(2)
                _wc_v1 = _wc_c1.selectbox("Measurement 1", subst_num_cols, key="wc_v1")
                _wc_v2 = _wc_c2.selectbox("Measurement 2",
                    [c for c in subst_num_cols if c != _wc_v1] or subst_num_cols, key="wc_v2")
                if st.button("Run Wilcoxon Signed-Rank Test", type="primary", key="run_wc"):
                    _wc_res = wilcoxon_signed_rank_test(df, _wc_v1, _wc_v2)
                    if "error" not in _wc_res:
                        st.session_state.analysis_history.append(_wc_res)
                        st.session_state.last_result = _wc_res
                        st.caption(f"N used (complete pairs): {_wc_res.get('n','?')}")
                        if _wc_res.get("warning"):
                            warn_box(_wc_res["warning"])
                        st.markdown(f"""
<div class="result-box">
<div style="font-size:15px;font-weight:700;color:#0d47a1;">Wilcoxon Signed-Rank Test</div>
<p><strong>W statistic:</strong> {_wc_res.get('statistic','?')} &nbsp;|&nbsp;
<strong>p-value:</strong> {_wc_res.get('p_value','?')} &nbsp;|&nbsp;
<strong>{'✅ Significant' if _wc_res.get('significant') else '❌ Not significant'}</strong></p>
<p><strong>Effect size (rank-biserial r):</strong> {_wc_res.get('rank_biserial_r','?')}</p>
<p>{_wc_res.get('interpretation','')}</p>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.error(_wc_res["error"])

        # ── Kruskal-Wallis ─────────────────────────────────────────────────
        with np_tab3:
            st.markdown("##### Kruskal-Wallis Test")
            st.markdown("""
**When to use:** Three or more independent groups; normality assumption is violated or data is ordinal.
**Non-parametric equivalent of:** One-Way ANOVA.
**Effect size:** η²H (eta-squared H).
**Post-hoc:** If significant, use pairwise Mann-Whitney U with Bonferroni correction.
""")
            if not subst_num_cols or not cat_cols:
                st.info("Need at least one numerical and one categorical variable.")
            else:
                _kw_c1, _kw_c2 = st.columns(2)
                _kw_num = _kw_c1.selectbox("Numerical variable", subst_num_cols, key="kw_num")
                _kw_grp = _kw_c2.selectbox("Grouping variable", cat_cols, key="kw_grp")
                if st.button("Run Kruskal-Wallis Test", type="primary", key="run_kw"):
                    _kw_res = kruskal_wallis_test(df, _kw_num, _kw_grp)
                    if "error" not in _kw_res:
                        st.session_state.analysis_history.append(_kw_res)
                        st.session_state.last_result = _kw_res
                        _kw_gs = _kw_res.get("group_summary", {})
                        if _kw_gs:
                            _kw_gs_rows = [{"Group": g, "N": d.get("n","?"),
                                            "Median": d.get("median","?"), "Mean Rank": d.get("mean_rank","?")}
                                           for g, d in _kw_gs.items()]
                            st.caption(f"N used (total): {_kw_res.get('n','?')}")
                            st.table(pd.DataFrame(_kw_gs_rows).reset_index(drop=True))
                        st.markdown(f"""
<div class="result-box">
<div style="font-size:15px;font-weight:700;color:#0d47a1;">Kruskal-Wallis Test</div>
<p><strong>H statistic:</strong> {_kw_res.get('h_statistic','?')} &nbsp;|&nbsp;
<strong>p-value:</strong> {_kw_res.get('p_value','?')} &nbsp;|&nbsp;
<strong>{'✅ Significant' if _kw_res.get('significant') else '❌ Not significant'}</strong></p>
<p><strong>Effect size (η²H):</strong> {_kw_res.get('eta_squared_h','?')}</p>
<p>{_kw_res.get('interpretation','')}</p>
</div>
""", unsafe_allow_html=True)
                        if _kw_res.get("posthoc_note"):
                            info_box(_kw_res["posthoc_note"])
                    else:
                        st.error(_kw_res["error"])

        # ── Normality Check ────────────────────────────────────────────────
        with np_tab4:
            st.markdown("##### Normality Test")
            st.markdown("""
Tests whether a numerical variable is approximately normally distributed.
- **Shapiro-Wilk** (n ≤ 50): most powerful for small samples.
- **D'Agostino-Pearson K²** (n > 50): better for larger samples.
- p > 0.05 → no significant departure from normality (does not prove normality, only fails to reject it).
""")
            if not subst_num_cols:
                st.info("No numerical variables available.")
            else:
                _nc_sel = st.multiselect("Select variables to check", subst_num_cols,
                                         default=subst_num_cols[:min(3, len(subst_num_cols))], key="nc_sel")
                if _nc_sel and st.button("Run Normality Checks", type="primary", key="run_nc"):
                    for _nc_col in _nc_sel:
                        _nc_r = normality_check(df[_nc_col].dropna(), _nc_col)
                        if "error" not in _nc_r:
                            _nc_icon = "✅" if _nc_r.get("is_normal") else "⚠️"
                            st.markdown(f"""
<div class="step-card">
<strong>{_nc_col}</strong> &nbsp;|&nbsp; N = {_nc_r.get('n','?')} &nbsp;|&nbsp;
Test: {_nc_r.get('test_used','')} &nbsp;|&nbsp;
p = {_nc_r.get('p_value','?')} &nbsp;|&nbsp;
{_nc_icon} {'Normal' if _nc_r.get('is_normal') else 'Non-normal'}<br>
Skewness = {_nc_r.get('skewness','?')} &nbsp;|&nbsp; Kurtosis = {_nc_r.get('kurtosis','?')}<br>
<em>{_nc_r.get('interpretation','')}</em><br>
<strong>Recommendation:</strong> {_nc_r.get('recommendation','')}
</div>
""", unsafe_allow_html=True)
                        else:
                            st.error(f"{_nc_col}: {_nc_r['error']}")

        # ── Bonferroni Correction ──────────────────────────────────────────
        with np_tab5:
            st.markdown("##### Bonferroni Multiple Testing Correction")
            st.markdown("""
When you run many statistical tests on the same dataset, the probability of getting at least one false positive increases.
**Bonferroni correction** adjusts the significance threshold to control the family-wise error rate.
- Enter the p-values from your analyses below.
- The system adjusts them and shows which remain significant after correction.
""")
            _bf_input = st.text_area(
                "Enter p-values (one per line or comma-separated)",
                value="", key="bf_input",
                placeholder="e.g.\n0.03\n0.04\n0.002\n0.08",
                height=120
            )
            _bf_alpha = st.number_input("Original α level", min_value=0.001, max_value=0.20,
                                         value=0.05, step=0.001, format="%.3f", key="bf_alpha")
            if st.button("Apply Bonferroni Correction", type="primary", key="run_bf"):
                # Parse p-values
                import re as _re
                _raw_p = _re.split(r"[,\s\n]+", _bf_input.strip())
                try:
                    _p_vals = [float(p.strip()) for p in _raw_p if p.strip()]
                    if not _p_vals:
                        st.warning("No p-values entered.")
                    else:
                        _bf_res = bonferroni_correction(_p_vals, alpha=float(_bf_alpha))
                        if "error" not in _bf_res:
                            st.markdown(f"""
<div class="result-box">
<strong>N tests:</strong> {_bf_res['n_tests']} &nbsp;|&nbsp;
<strong>α original:</strong> {_bf_res['alpha_original']} &nbsp;|&nbsp;
<strong>α Bonferroni corrected:</strong> {_bf_res['alpha_corrected']:.4f}<br>
<strong>Significant before correction:</strong> {_bf_res['n_significant_before']} &nbsp;|&nbsp;
<strong>Significant after correction:</strong> {_bf_res['n_significant_after']}
</div>
""", unsafe_allow_html=True)
                            _bf_rows = []
                            for _i, (_p_orig, _p_adj, _sig_b, _sig_a) in enumerate(zip(
                                _bf_res['p_values_original'],
                                _bf_res['p_values_adjusted'],
                                _bf_res['significant_original'],
                                _bf_res['significant_corrected']
                            ), 1):
                                _bf_rows.append({
                                    "Test #": _i,
                                    "Original p": round(_p_orig, 6),
                                    "Adjusted p (×N)": round(_p_adj, 6),
                                    "Sig. before": "✅" if _sig_b else "❌",
                                    "Sig. after (Bonferroni)": "✅" if _sig_a else "❌",
                                })
                            st.table(pd.DataFrame(_bf_rows).reset_index(drop=True))
                            st.markdown(f"**Interpretation:** {_bf_res.get('interpretation','')}")
                        else:
                            st.error(_bf_res["error"])
                except ValueError:
                    st.error("Invalid p-values entered. Please enter numbers only.")


    # ── Advanced Tab: Friedman, PCA, Holm-Bonferroni ──────────────────────────
    with tab_adv:
        st.markdown("#### 🔬 Advanced Statistical Methods")
        info_box(
            "Advanced methods including Friedman repeated-measures test, "
            "Principal Component Analysis (PCA), and Holm-Bonferroni multiple testing correction. "
            "These require specific data structures — read the requirements before running."
        )

        adv_tab1, adv_tab2, adv_tab3 = st.tabs([
            "🔄 Friedman Test (repeated)", "🧭 PCA (dimension reduction)", "📊 Holm-Bonferroni Correction"
        ])

        # ── Friedman Test ──────────────────────────────────────────────────
        with adv_tab1:
            st.markdown("##### Friedman Test — Non-Parametric Repeated Measures")
            st.markdown("""
**When to use:** 3 or more related/repeated measurements from the same subjects; normality cannot be assumed.
**Non-parametric alternative to:** Repeated-Measures ANOVA.
**Each row = one subject. Each selected column = one measurement condition/time point.**
**Effect size:** Kendall's W (concordance coefficient).
**Post-hoc:** Pairwise Wilcoxon signed-rank tests with Bonferroni/Holm correction.
""")
            if len(subst_num_cols) < 3:
                st.info(f"Need at least 3 numerical variables for Friedman test. Currently: {len(subst_num_cols)}.")
            else:
                _fr_cols = st.multiselect(
                    "Select 3 or more measurement columns (each column = one condition/time point)",
                    subst_num_cols,
                    default=subst_num_cols[:min(3, len(subst_num_cols))],
                    key="fr_cols"
                )
                if len(_fr_cols) >= 3 and st.button("Run Friedman Test", type="primary", key="run_fr"):
                    _fr_result = friedman_test(df, _fr_cols)
                    if "error" not in _fr_result:
                        st.session_state.analysis_history.append(_fr_result)
                        st.session_state.last_result = _fr_result
                        st.caption(f"N subjects (complete blocks): {_fr_result.get('n_subjects', 'N/A')} | Conditions: {_fr_result.get('n_conditions', 'N/A')}")
                        st.markdown(interpret_result(_fr_result))
                        _fr_gs = _fr_result.get("group_summary", {})
                        if _fr_gs:
                            _fr_gs_rows = [{"Condition": c, "N": d.get("n","?"),
                                            "Median": d.get("median","?"), "Mean": d.get("mean","?")}
                                           for c, d in _fr_gs.items()]
                            st.markdown("**Condition summary:**")
                            st.table(pd.DataFrame(_fr_gs_rows).reset_index(drop=True))
                        if _fr_result.get("posthoc_note"):
                            info_box(_fr_result["posthoc_note"])
                    else:
                        st.error(_fr_result["error"])
                elif len(_fr_cols) < 3:
                    st.info("Select at least 3 columns above.")

        # ── PCA ────────────────────────────────────────────────────────────
        with adv_tab2:
            st.markdown("##### Principal Component Analysis (PCA)")
            st.markdown("""
**When to use:** Reduce many correlated numerical variables to fewer uncorrelated components.
**Requirements:** At least 3 continuous (interval/ratio) variables, ≥ 10 complete observations (ideally ≥ 5 per variable).
**Variables are automatically standardised (z-scored) before PCA.**
**This is exploratory — components are data-driven, not theory-driven.**
""")
            if len(subst_num_cols) < 3:
                st.info(f"Need at least 3 numerical variables for PCA. Currently: {len(subst_num_cols)}.")
            else:
                _pca_cols = st.multiselect(
                    "Select variables for PCA (all must be continuous numerical)",
                    subst_num_cols,
                    default=subst_num_cols[:min(6, len(subst_num_cols))],
                    key="pca_cols"
                )
                _pca_n_comp = st.number_input(
                    "Maximum components to extract (0 = all)",
                    min_value=0, max_value=min(20, len(subst_num_cols)),
                    value=0, step=1, key="pca_n_comp"
                )
                if len(_pca_cols) >= 3 and st.button("Run PCA", type="primary", key="run_pca"):
                    _pca_n = int(_pca_n_comp) if _pca_n_comp > 0 else None
                    _pca_result = pca_analysis(df, _pca_cols, n_components=_pca_n)
                    if "error" not in _pca_result:
                        if _pca_result.get("warning"):
                            warn_box(_pca_result["warning"])
                        st.session_state.analysis_history.append({
                            "test": _pca_result["test"],
                            "n": _pca_result["n_obs"],
                            "n_total": _pca_result["n_total"],
                            "interpretation": _pca_result["interpretation"],
                        })
                        st.session_state.last_result = _pca_result
                        st.caption(f"N observations (complete cases): {_pca_result.get('n_obs','N/A')} of {_pca_result.get('n_total','N/A')}")
                        st.markdown(interpret_result(_pca_result))

                        # Variance table
                        _ev_rows = []
                        for i, (ev, evr, cv) in enumerate(zip(
                            _pca_result["eigenvalues"],
                            _pca_result["explained_variance_ratio"],
                            _pca_result["cumulative_variance"]
                        ), 1):
                            _ev_rows.append({
                                "Component": f"PC{i}",
                                "Eigenvalue": ev,
                                "Variance %": f"{round(evr*100,1)}%",
                                "Cumulative %": f"{round(cv*100,1)}%",
                                "Retain (Kaiser ≥ 1)": "✅" if ev >= 1.0 else "❌",
                            })
                        st.markdown("**Variance explained per component:**")
                        st.table(pd.DataFrame(_ev_rows).reset_index(drop=True))

                        # Loadings
                        _ld = _pca_result.get("loadings_df")
                        if _ld is not None:
                            st.markdown("**Component loadings (|≥0.40| considered substantial):**")
                            st.table(_ld)

                        st.caption(_pca_result["assumption_note"])
                    else:
                        st.error(_pca_result["error"])
                elif len(_pca_cols) < 3:
                    st.info("Select at least 3 variables above.")

        # ── Holm-Bonferroni ────────────────────────────────────────────────
        with adv_tab3:
            st.markdown("##### Holm-Bonferroni Multiple Testing Correction")
            st.markdown("""
The **Holm-Bonferroni** procedure is uniformly more powerful than standard Bonferroni correction
while providing the same family-wise error rate (FWER) protection.

**When to use:** When you have run several statistical tests and want a less conservative correction than Bonferroni.

**How it works:** Tests are ranked by p-value from smallest to largest. Each test is compared to
a progressively relaxed threshold (α / remaining tests). Once one test fails, all subsequent tests fail.
""")
            _holm_input = st.text_area(
                "Enter p-values (one per line or comma-separated)",
                value="", key="holm_input",
                placeholder="e.g.\n0.03\n0.04\n0.002\n0.08",
                height=120
            )
            _holm_alpha = st.number_input("Original α level", min_value=0.001, max_value=0.20,
                                           value=0.05, step=0.001, format="%.3f", key="holm_alpha")
            if st.button("Apply Holm-Bonferroni Correction", type="primary", key="run_holm"):
                import re as _re2
                _raw_holm = _re2.split(r"[,\s\n]+", _holm_input.strip())
                try:
                    _holm_pvals = [float(p.strip()) for p in _raw_holm if p.strip()]
                    if not _holm_pvals:
                        st.warning("No p-values entered.")
                    else:
                        _holm_res = holm_bonferroni_correction(_holm_pvals, alpha=float(_holm_alpha))
                        if "error" not in _holm_res:
                            st.markdown(f"""
<div class="result-box">
<strong>N tests:</strong> {_holm_res['n_tests']} &nbsp;|&nbsp;
<strong>α original:</strong> {_holm_res['alpha_original']} &nbsp;|&nbsp;
<strong>Method:</strong> Holm-Bonferroni<br>
<strong>Significant before correction:</strong> {_holm_res['n_significant_before']} &nbsp;|&nbsp;
<strong>Significant after Holm:</strong> {_holm_res['n_significant_after']}
</div>
""", unsafe_allow_html=True)
                            _holm_rows = []
                            for _i, (_p_orig, _p_adj, _sig_b, _sig_a) in enumerate(zip(
                                _holm_res['p_values_original'],
                                _holm_res['p_values_adjusted'],
                                _holm_res['significant_original'],
                                _holm_res['significant_corrected']
                            ), 1):
                                _holm_rows.append({
                                    "Test #": _i,
                                    "Original p": round(_p_orig, 6),
                                    "Adjusted p (Holm)": round(_p_adj, 6),
                                    "Sig. before": "✅" if _sig_b else "❌",
                                    "Sig. after (Holm)": "✅" if _sig_a else "❌",
                                })
                            st.table(pd.DataFrame(_holm_rows).reset_index(drop=True))
                            st.markdown(f"**Interpretation:** {_holm_res.get('interpretation','')}")
                        else:
                            st.error(_holm_res["error"])
                except ValueError:
                    st.error("Invalid p-values entered. Please enter numbers only.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VISUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Visual" in page:
    _require_data()
    st.markdown('<div class="section-title">📈 Visual Analysis</div>', unsafe_allow_html=True)

    df      = _get_df()
    profile = st.session_state.profile or {}
    num_cols  = profile.get("analysis_ready_numerical", profile.get("numerical_cols", []))
    cat_cols  = profile.get("analysis_ready_categorical", profile.get("categorical_cols", []))
    dt_cols   = profile.get("datetime_cols", [])
    all_cols  = list(df.columns)

    try:
        from engine.visualizer import build_chart, recommend_visualization
        from engine.interpreter import explain_graph
        from engine.statistics import correlation_matrix as _viz_corr_matrix
        from engine.visualizer import correlation_heatmap as _viz_corr_heatmap
        _viz_imports_ok = True
    except ImportError as _viz_imp_err:
        _viz_imports_ok = False
        st.error(
            f"⚠️ Visual analysis module configuration error: {_viz_imp_err}"
        )
        st.stop()

    viz_tab, rec_tab = st.tabs(["🎨 Create Visualisation", "💡 Recommend Visualisation"])

    with viz_tab:
        st.markdown("#### Create a Chart")
        col_chart, col_opts = st.columns([2, 1])

        with col_opts:
            chart_type = st.selectbox("Chart type", [
                "bar_chart", "pie_chart", "histogram", "line_chart",
                "scatter_plot", "box_plot", "stacked_bar", "area_chart",
                "frequency_bar", "correlation_heatmap",
            ], format_func=lambda x: x.replace("_", " ").title())

            x_col = st.selectbox("X-axis / Primary variable", all_cols, key="viz_x")
            y_col = st.selectbox("Y-axis / Secondary variable (if needed)",
                                 ["(none)"] + [c for c in all_cols if c != x_col], key="viz_y")
            color_col = st.selectbox("Colour by (optional)",
                                     ["(none)"] + cat_cols, key="viz_color")
            title = st.text_input("Chart title (optional)", value="", key="viz_title")
            y_col_use    = None if y_col == "(none)" else y_col
            color_col_use = None if color_col == "(none)" else color_col

        with col_chart:
            if chart_type == "correlation_heatmap":
                if len(num_cols) < 2:
                    st.info("Need at least 2 numerical columns.")
                else:
                    sel = st.multiselect("Select numerical columns for heatmap", num_cols,
                                         default=num_cols[:min(6, len(num_cols))], key="hm_cols")
                    if sel and len(sel) >= 2:
                        corr = _viz_corr_matrix(df, sel)
                        fig  = _viz_corr_heatmap(corr)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                if st.button("Generate Chart", type="primary", key="gen_chart"):
                    try:
                        fig = build_chart(df, chart_type, x_col, y_col_use, color_col_use, title)
                        st.plotly_chart(fig, use_container_width=True)

                        # Explain graph
                        with st.expander("🤖 Explain This Graph (AI)", expanded=False):
                            cs = profile.get("col_stats", {}).get(y_col_use or x_col, {})
                            explanation = explain_graph(chart_type, x_col,
                                                        y_col_use or x_col, cs)
                            st.markdown(explanation)
                    except Exception as e:
                        st.error(f"Chart error: {str(e)}")

    with rec_tab:
        st.markdown("#### AI Visual Recommendation")
        info_box("Select two variables and the system will recommend the most appropriate chart type.")

        r1, r2 = st.columns(2)
        rx = r1.selectbox("Variable 1 (X-axis)", all_cols, key="rec_x")
        ry = r2.selectbox("Variable 2 (Y-axis)", all_cols, key="rec_y")

        if st.button("💡 Recommend Chart", type="primary", key="run_rec_viz"):
            xt = profile.get("variable_types", {}).get(rx, "categorical")
            yt = profile.get("variable_types", {}).get(ry, "categorical")
            recs = recommend_visualization(xt, yt, rx, ry)
            for rec in recs:
                colour = {"5": "#e0f7fa", "4": "#fdf6d3",
                          "3": "#fffde7", "2": "#fff3cd"}.get(str(rec["stars"]), "#ffffff")
                st.markdown(f"""
                <div style="background:{colour};border:1px solid #ddd;border-radius:10px;
                            padding:14px 18px;margin-bottom:10px;">
                    <div style="font-size:15px;font-weight:700;color:#0d7a8a;">
                        {rec['star_label']} — {rec['chart_type'].replace('_',' ').title()}
                    </div>
                    <p><strong>When to use:</strong> {rec['when_to_use']}</p>
                    <p><strong>Why appropriate:</strong> {rec['why_appropriate']}</p>
                </div>
                """, unsafe_allow_html=True)

            # Generate top recommended chart
            if recs:
                top = recs[0]
                try:
                    fig = build_chart(df, top["chart_type"], rx, ry, None, "")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: AI RESEARCH ASSISTANT
# ══════════════════════════════════════════════════════════════════════════════
elif "AI Research" in page:
    st.markdown('<div class="section-title">🤖 AI Research Assistant</div>', unsafe_allow_html=True)

    info_box("Ask the AI Research Assistant about your statistical results, methodology, "
             "interpretation, and how to defend your choices. The AI explains results computed "
             "by the statistical engine — it does not invent values.")

    from engine.interpreter import answer_research_question

    # Suggested questions — grouped by category
    with st.expander("💬 Suggested Questions (click any to ask)", expanded=True):
        suggestion_groups = {
            "Understanding Results": [
                "What does my p-value mean?",
                "What does the correlation coefficient r mean?",
                "What does R-squared mean?",
                "What does Cohen's d mean?",
                "What does eta squared mean?",
            ],
            "Statistics Basics": [
                "What are the types of data?",
                "What is variance and why do we square deviations?",
                "What is standard deviation?",
                "What is a z-score?",
                "What is MAD (mean absolute deviation)?",
                "What is a percentile?",
                "What is variation?",
            ],
            "Distributions & Inference": [
                "What is the normal distribution?",
                "What is the Central Limit Theorem?",
                "What is a sampling distribution?",
                "What is the difference between descriptive and inferential statistics?",
                "What is a random variable?",
                "What is a binomial distribution?",
            ],
            "Method Justification": [
                "Why was ANOVA used instead of a t-test?",
                "Why was Pearson used instead of Spearman?",
                "Why was chi-square used?",
                "Why was linear regression used?",
            ],
            "Critical Thinking": [
                "Does this result prove causation?",
                "What is the difference between statistical significance and practical significance?",
                "What are the assumptions of this test?",
                "What are the limitations of my analysis?",
                "What is covariance and how does it differ from correlation?",
            ],
            "Reporting & Viva": [
                "How should I report this result in APA format in my thesis?",
                "How do I justify this method in my viva?",
                "How were missing values handled in my dataset?",
                "What post-hoc test should I use after ANOVA?",
            ],
        }
        for group, sugs in suggestion_groups.items():
            st.markdown(f"**{group}**")
            sug_cols = st.columns(2)
            _grp_key = group.replace(" ", "_").replace("&", "and")
            for si, sug in enumerate(sugs):
                if sug_cols[si % 2].button(sug, key=f"sug_{_grp_key}_{si}"):
                    st.session_state.chat_history.append({"role": "user", "content": sug})
                    context = {
                        "last_result": st.session_state.last_result,
                        "dataset_profile": st.session_state.profile,
                        "cleaning_log": st.session_state.cleaning_log,
                    }
                    answer = answer_research_question(sug, context)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask your research question here..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        context = {
            "last_result": st.session_state.last_result,
            "dataset_profile": st.session_state.profile,
            "cleaning_log": st.session_state.cleaning_log,
        }
        answer = answer_research_question(prompt, context)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑 Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: VIVA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════
elif "Viva" in page:
    st.markdown('<div class="section-title">🎓 Viva Preparation</div>', unsafe_allow_html=True)

    info_box("Practise answering likely viva questions based on your actual analysis choices. "
             "Questions are generated from the statistical methods you have used.")

    from engine.interpreter import generate_viva_questions, answer_research_question

    analyses = st.session_state.analysis_history
    if not analyses:
        st.warning("No analyses have been performed yet. Complete some statistical analyses first, "
                   "then return here to practise viva questions.")
    else:
        questions = generate_viva_questions(analyses)
        categories = list(dict.fromkeys(q["category"] for q in questions))
        cat_filter = st.multiselect("Filter by category", categories, default=categories)
        filtered_q = [q for q in questions if q["category"] in cat_filter]

        st.markdown(f"**{len(filtered_q)} viva question(s) generated from your analysis.**")
        st.markdown("---")

        for i, q in enumerate(filtered_q, 1):
            with st.expander(f"**Q{i}: {q['question']}** *(Category: {q['category']})*", expanded=False):
                st.markdown(f"**Hint:** {q['hint']}")
                st.markdown("---")
                # Practice answer input
                answer_key = f"viva_ans_{i}"
                user_answer = st.text_area("Write your answer below to practise:",
                                           key=answer_key, height=100,
                                           placeholder="Type your practice answer here...")
                if user_answer and st.button(f"🤖 Get AI Feedback on My Answer", key=f"viva_fb_{i}"):
                    context = {"last_result": st.session_state.last_result}
                    ai_response = answer_research_question(q["question"], context)
                    st.markdown(f"**AI Reference Answer:**\n\n{ai_response}")



# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FINDINGS
# ══════════════════════════════════════════════════════════════════════════════
elif "Findings" in page:
    _require_data()
    st.markdown('<div class="section-title">🔎 Findings</div>', unsafe_allow_html=True)

    info_box(
        "Record and review your key findings here. Every finding should be traceable to a "
        "statistical result. Use the AI chatbot to ask 'Why this conclusion?' for any finding."
    )

    from engine.interpreter import answer_research_question as _arq

    analyses = st.session_state.analysis_history
    findings = st.session_state.key_findings
    profile  = st.session_state.profile or {}

    # ── Add a finding ──────────────────────────────────────────────────────────
    st.markdown("### Add a Key Finding")
    with st.form("add_finding_form", clear_on_submit=True):
        new_finding_text = st.text_area(
            "Describe your finding",
            placeholder=(
                "e.g. A significant positive correlation was found between Study Hours and Exam Score "
                "(r = 0.72, p = 0.003). Students who study more tend to score higher."
            ),
            height=90,
        )
        submitted_f = st.form_submit_button("Add Finding", type="primary")
        if submitted_f and new_finding_text.strip():
            st.session_state.key_findings.append(new_finding_text.strip())
            st.rerun()

    # ── Auto-suggest from analyses ─────────────────────────────────────────────
    if analyses and st.button("🤖 Auto-Generate Findings from My Analyses", key="auto_findings"):
        auto_findings = [r["interpretation"] for r in analyses if r.get("interpretation")]
        new_ones = [f for f in auto_findings if f not in st.session_state.key_findings]
        st.session_state.key_findings.extend(new_ones)
        st.success(f"Added {len(new_ones)} finding(s) from your analyses.")
        st.rerun()

    st.markdown("---")

    # ── Display findings ───────────────────────────────────────────────────────
    if not findings:
        st.info("No findings added yet. Add findings manually above or click **Auto-Generate** to populate from your analyses.")
    else:
        st.markdown(f"### Your Findings ({len(findings)} recorded)")
        for fi, finding in enumerate(findings, 1):
            with st.expander(f"**Finding {fi}:** {finding[:80]}{'...' if len(finding)>80 else ''}", expanded=True):
                st.markdown(finding)
                st.markdown("---")
                col_ev, col_del = st.columns([4, 1])
                with col_ev:
                    if st.button(f"🤖 Why this conclusion?", key=f"why_{fi}"):
                        context = {
                            "last_result": st.session_state.last_result,
                            "dataset_profile": profile,
                            "cleaning_log": st.session_state.cleaning_log,
                        }
                        explanation = _arq(f"Interpret and explain this finding: {finding}", context)
                        st.markdown(f"**AI Evidence and Reasoning:**\n\n{explanation}")
                with col_del:
                    if st.button("Remove", key=f"del_f_{fi}"):
                        st.session_state.key_findings.pop(fi - 1)
                        st.rerun()

    # ── Statistical results summary ────────────────────────────────────────────
    if analyses:
        st.markdown("---")
        st.markdown("### Statistical Results Summary")
        info_box("All results below were computed by the statistical engine — the AI does not calculate these values.")
        for ri, result in enumerate(analyses, 1):
            test = result.get("test", f"Analysis {ri}")
            sig  = result.get("significant", None)
            sig_badge = (" — Significant (p < 0.05)" if sig else " — Not significant") if sig is not None else ""
            with st.expander(f"**{ri}. {test}**{sig_badge}", expanded=False):
                if "r" in result:
                    st.markdown(f"**r = {result['r']}**  |  p = {result.get('p_value','?')}  |  n = {result.get('n','?')}")
                if "t_statistic" in result:
                    st.markdown(f"**t = {result['t_statistic']}**  |  p = {result.get('p_value','?')}  |  Cohen's d = {result.get('cohens_d','?')}")
                if "f_statistic" in result and "chi2" not in result:
                    st.markdown(f"**F = {result['f_statistic']}**  |  p = {result.get('p_value','?')}  |  eta2 = {result.get('eta_squared','?')}")
                if "chi2" in result:
                    st.markdown(f"**chi2 = {result['chi2']}**  |  df = {result.get('df','?')}  |  p = {result.get('p_value','?')}  |  Cramer's V = {result.get('cramers_v','?')}")
                if "r_squared" in result:
                    st.markdown(f"**R2 = {result['r_squared']}**  |  Adj R2 = {result.get('adj_r_squared','?')}  |  F = {result.get('f_statistic','?')}  |  p = {result.get('p_value','?')}")
                if "alpha" in result:
                    st.markdown(f"**alpha = {result['alpha']}**  |  {result.get('reliability','')}  |  n items = {result.get('n_items','?')}")
                st.markdown(f"**Interpretation:** {result.get('interpretation','')}")
                if result.get("assumption_note"):
                    st.caption(f"Assumptions: {result['assumption_note']}")
                if result.get("warning"):
                    st.warning(result["warning"])
                if st.button(f"Add This Interpretation as a Finding", key=f"add_f_{ri}"):
                    stmt = result.get("interpretation", f"{test}: p = {result.get('p_value','?')}")
                    if stmt not in st.session_state.key_findings:
                        st.session_state.key_findings.append(stmt)
                        st.success("Added to findings.")
                        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <strong>Research Transparency Reminder:</strong>
        Every finding should be traceable to verified computational results:
        <strong>Raw Data - Statistical Calculation - Verified Result - Interpretation - Conclusion</strong>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RESEARCH REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif "Report" in page:
    _require_data()
    st.markdown('<div class="section-title">📝 Research Report</div>', unsafe_allow_html=True)

    info_box("Generate an evidence-based research report. Every conclusion is traceable to "
             "computed statistical results. The report follows the chain: "
             "Raw Data → Calculation → Statistical Result → Interpretation → Conclusion.")

    from engine.reporter import generate_report_markdown, generate_report_docx

    profile   = st.session_state.profile or {}
    log       = st.session_state.cleaning_log
    analyses  = st.session_state.analysis_history
    findings  = st.session_state.key_findings

    # Add key findings
    st.markdown("### Add Key Findings")
    info_box("Add your findings here, or go to the **Findings** page to manage them in detail. "
             "Each finding will appear in the report.")
    with st.form("report_finding_form", clear_on_submit=True):
        new_finding = st.text_input("Add a key finding",
                                    placeholder="e.g. A significant positive correlation was found between X and Y (r = 0.72, p = 0.003).")
        if st.form_submit_button("Add Finding") and new_finding.strip():
            st.session_state.key_findings.append(new_finding.strip())
            st.rerun()

    if findings:
        st.markdown("**Current findings:**")
        for i, f in enumerate(findings):
            col_f, col_del = st.columns([9, 1])
            col_f.markdown(f"- {f}")
            if col_del.button("x", key=f"del_finding_r_{i}"):
                st.session_state.key_findings.pop(i)
                st.rerun()

    st.markdown("---")

    if st.button("📝 Generate Full Research Report", type="primary", key="gen_report"):
        report_md = generate_report_markdown(
            dataset_name=st.session_state.file_name,
            profile=profile,
            cleaning_log=log,
            analyses=analyses,
            key_findings=findings,
        )
        st.session_state["_last_report"] = report_md
        st.markdown(report_md)

        # DOCX download
        docx_bytes = generate_report_docx(
            dataset_name=st.session_state.file_name,
            profile=profile,
            cleaning_log=log,
            analyses=analyses,
            key_findings=findings,
        )
        if docx_bytes:
            st.download_button(
                "📥 Download Report (Word / DOCX)",
                data=docx_bytes,
                file_name="research_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        # Markdown download
        st.download_button(
            "📥 Download Report (Markdown)",
            data=report_md.encode(),
            file_name="research_report.md",
            mime="text/markdown",
        )

    elif "_last_report" in st.session_state and st.session_state["_last_report"]:
        st.markdown(st.session_state["_last_report"])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EXPORT
# ══════════════════════════════════════════════════════════════════════════════
elif "Export" in page:
    _require_data()
    st.markdown('<div class="section-title">💾 Export</div>', unsafe_allow_html=True)

    import os as _os
    _stem = _os.path.splitext(st.session_state.file_name)[0]   # strip extension once

    df_orig    = st.session_state.raw_df
    df_cleaned = st.session_state.working_df
    log        = st.session_state.cleaning_log
    analyses   = st.session_state.analysis_history

    st.markdown("### Available Downloads")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**📊 Data Files**")

        # Original
        buf = io.BytesIO()
        df_orig.to_excel(buf, index=False)
        st.download_button("📥 Original Dataset (Excel)",
                           data=buf.getvalue(),
                           file_name=f"original_{_stem}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

        # Cleaned
        buf2 = io.BytesIO()
        df_cleaned.to_excel(buf2, index=False)
        st.download_button("📥 Cleaned Dataset (Excel)",
                           data=buf2.getvalue(),
                           file_name=f"cleaned_{_stem}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           use_container_width=True)

        # Cleaned CSV
        csv_bytes = df_cleaned.to_csv(index=False).encode()
        st.download_button("📥 Cleaned Dataset (CSV)",
                           data=csv_bytes,
                           file_name=f"cleaned_{_stem}.csv",
                           mime="text/csv",
                           use_container_width=True)

    with col_b:
        st.markdown("**📋 Analysis Files**")

        # Cleaning log
        if log:
            buf3 = io.BytesIO()
            pd.DataFrame(log).to_excel(buf3, index=False)
            st.download_button("📥 Cleaning Log (Excel)",
                               data=buf3.getvalue(),
                               file_name="cleaning_log.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

        # Statistical results
        if analyses:
            safe_analyses = []
            for a in analyses:
                safe = {k: v for k, v in a.items()
                        if not isinstance(v, pd.DataFrame) and not hasattr(v, 'head')}
                safe_analyses.append(safe)
            buf4 = io.BytesIO()
            pd.DataFrame(safe_analyses).to_excel(buf4, index=False)
            st.download_button("📥 Statistical Results (Excel)",
                               data=buf4.getvalue(),
                               file_name="statistical_results.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               use_container_width=True)

        # Report
        from engine.reporter import generate_report_markdown, generate_report_docx
        profile = st.session_state.profile or {}
        findings = st.session_state.key_findings

        report_md = generate_report_markdown(_stem, profile, log, analyses, findings)
        st.download_button("📥 Research Report (Markdown)",
                           data=report_md.encode(),
                           file_name="research_report.md",
                           mime="text/markdown",
                           use_container_width=True)

        docx_bytes = generate_report_docx(_stem, profile, log, analyses, findings)
        if docx_bytes:
            st.download_button("📥 Research Report (Word/DOCX)",
                               data=docx_bytes,
                               file_name="research_report.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               use_container_width=True)

    # ── Part 20: Reproducibility Script Export ────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔁 Reproducibility Script")
    st.markdown(
        "Download a self-contained Python script that re-runs every analysis performed "
        "in this session. Open it in any Python environment (Jupyter, VS Code, etc.) "
        "and run it with your cleaned CSV file to reproduce all results."
    )

    def _build_repro_script(file_stem: str, log_entries: list, analysis_results: list) -> str:
        """Generate a standalone Python reproducibility script from the session state."""
        from datetime import datetime as _dt
        now_str = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f'"""',
            f'Reproducibility Script — AI Research Analyst',
            f'Generated: {now_str}',
            f'Dataset:   {file_stem}',
            f'',
            f'Instructions:',
            f'  1. Install dependencies:',
            f'       pip install pandas scipy statsmodels scikit-learn openpyxl',
            f'  2. Place your cleaned dataset CSV in the same folder as this script.',
            f'  3. Update DATA_FILE below to match the actual filename.',
            f'  4. Run:  python {file_stem}_reproducibility.py',
            f'"""',
            f'',
            f'import pandas as pd',
            f'import numpy as np',
            f'import warnings',
            f'warnings.filterwarnings("ignore")',
            f'',
            f'# ── Configuration ────────────────────────────────────────────────────────────',
            f'DATA_FILE = "cleaned_{file_stem}.csv"   # update if filename differs',
            f'',
            f'# ── Load Data ────────────────────────────────────────────────────────────────',
            f'df = pd.read_csv(DATA_FILE)',
            f'print(f"Loaded: {{len(df)}} rows × {{len(df.columns)}} columns")',
            f'print(f"Columns: {{list(df.columns)}}")',
            f'print()',
        ]

        # ── Data cleaning steps ────────────────────────────────────────────────
        if log_entries:
            lines += [
                f'',
                f'# ── Data Cleaning Steps ({len(log_entries)} operation(s)) ─────────────────────────────',
            ]
            for i, entry in enumerate(log_entries, 1):
                col     = entry.get("column", "?")
                op      = entry.get("operation", "?")
                method  = entry.get("method", "?")
                detail  = entry.get("detail", "")
                lines.append(f'# Step {i}: {op} — column "{col}" — method: {method}')
                if detail:
                    lines.append(f'#   Detail: {detail}')
                # Generate concrete pandas code per operation type
                op_lower = op.lower()
                meth_lower = method.lower()
                if "missing" in op_lower or "impute" in op_lower:
                    if "mean" in meth_lower:
                        lines.append(f'df["{col}"] = df["{col}"].fillna(df["{col}"].mean())')
                    elif "median" in meth_lower:
                        lines.append(f'df["{col}"] = df["{col}"].fillna(df["{col}"].median())')
                    elif "mode" in meth_lower:
                        lines.append(f'df["{col}"] = df["{col}"].fillna(df["{col}"].mode()[0])')
                    elif "knn" in meth_lower:
                        lines.append(f'# KNN imputation — install scikit-learn:')
                        lines.append(f'# from sklearn.impute import KNNImputer')
                        lines.append(f'# df["{col}"] = KNNImputer(n_neighbors=5).fit_transform(df[["{col}"]])')
                    else:
                        lines.append(f'df["{col}"].fillna(method="ffill", inplace=True)  # adjust method if needed')
                elif "outlier" in op_lower or "cap" in op_lower or "winsor" in op_lower:
                    lines.append(f'q1, q3 = df["{col}"].quantile([0.25, 0.75])')
                    lines.append(f'iqr = q3 - q1')
                    lines.append(f'df["{col}"] = df["{col}"].clip(lower=q1 - 1.5*iqr, upper=q3 + 1.5*iqr)')
                elif "log" in meth_lower or "log" in op_lower:
                    lines.append(f'df["{col}"] = np.log1p(df["{col}"])')
                elif "zscore" in meth_lower or "standardis" in meth_lower or "standardiz" in meth_lower:
                    lines.append(f'df["{col}"] = (df["{col}"] - df["{col}"].mean()) / df["{col}"].std()')
                elif "drop" in op_lower or "remov" in op_lower:
                    lines.append(f'df = df.dropna(subset=["{col}"])  # remove rows with missing values in this column')
                else:
                    lines.append(f'# (cleaning step recorded — update code as needed)')
                lines.append(f'print("Cleaning step {i} applied: {op} — {col}")')
                lines.append(f'')
        else:
            lines += [
                f'',
                f'# ── Data Cleaning ────────────────────────────────────────────────────────────',
                f'# No cleaning operations were recorded for this session.',
                f'',
            ]

        # ── Statistical analyses ──────────────────────────────────────────────
        if analysis_results:
            lines += [
                f'',
                f'# ── Statistical Analyses ({len(analysis_results)} test(s)) ──────────────────────────────────',
                f'from scipy import stats as sp',
                f'',
            ]
            for i, res in enumerate(analysis_results, 1):
                test = res.get("test", f"Analysis {i}")
                lines.append(f'# ── Analysis {i}: {test} ──────────────────────')

                col1 = res.get("col1") or res.get("cont_col") or res.get("num_col") or res.get("dependent", "col1")
                col2 = res.get("col2") or res.get("group_col") or res.get("binary_col") or ""
                n    = res.get("n", "N/A")
                p    = res.get("p_value", "N/A")

                test_lower = test.lower()

                if "pearson" in test_lower:
                    lines.append(f'x1 = df["{col1}"].dropna()')
                    lines.append(f'x2 = df["{col2}"].dropna()')
                    lines.append(f'paired = df[["{col1}", "{col2}"]].dropna()')
                    lines.append(f'r, p_val = sp.pearsonr(paired["{col1}"].astype(float), paired["{col2}"].astype(float))')
                    lines.append(f'print(f"Pearson r = {{r:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "spearman" in test_lower:
                    lines.append(f'paired = df[["{col1}", "{col2}"]].dropna()')
                    lines.append(f'r, p_val = sp.spearmanr(paired["{col1}"].astype(float), paired["{col2}"].astype(float))')
                    lines.append(f'print(f"Spearman rho = {{r:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "kendall" in test_lower:
                    lines.append(f'paired = df[["{col1}", "{col2}"]].dropna()')
                    lines.append(f'tau, p_val = sp.kendalltau(paired["{col1}"].astype(float), paired["{col2}"].astype(float))')
                    lines.append(f'print(f"Kendall tau = {{tau:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "point-biserial" in test_lower:
                    lines.append(f'paired = df[["{col1}", "{col2}"]].dropna()')
                    lines.append(f'binary_enc = (paired["{col2}"] == paired["{col2}"].unique()[1]).astype(int)')
                    lines.append(f'r_pb, p_val = sp.pointbiserialr(binary_enc.values, paired["{col1}"].astype(float).values)')
                    lines.append(f'print(f"Point-Biserial r = {{r_pb:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "independent samples t" in test_lower or "t-test" in test_lower:
                    grp1 = res.get("group1", "Group1")
                    grp2 = res.get("group2", "Group2")
                    lines.append(f'g1 = df.loc[df["{col2}"] == "{grp1}", "{col1}"].dropna().astype(float)')
                    lines.append(f'g2 = df.loc[df["{col2}"] == "{grp2}", "{col1}"].dropna().astype(float)')
                    lines.append(f't, p_val = sp.ttest_ind(g1, g2, equal_var=False)')
                    lines.append(f'print(f"t = {{t:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "paired" in test_lower and "t-test" in test_lower:
                    lines.append(f'paired = df[["{col1}", "{col2}"]].dropna().astype(float)')
                    lines.append(f't, p_val = sp.ttest_rel(paired["{col1}"], paired["{col2}"])')
                    lines.append(f'print(f"Paired t = {{t:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "chi-square test of independence" in test_lower or "chi-square independence" in test_lower:
                    from scipy.stats import chi2_contingency as _c2c
                    lines.append(f'ct = pd.crosstab(df["{col1}"], df["{col2}"])')
                    lines.append(f'chi2, p_val, dof, expected = sp.chi2_contingency(ct)')
                    lines.append(f'print(f"Chi2 = {{chi2:.4f}}, df = {{dof}}, p = {{p_val:.6f}}, n = {n}")')
                elif "goodness-of-fit" in test_lower:
                    lines.append(f'obs = df["{col1}"].value_counts().sort_index()')
                    lines.append(f'n_total = obs.sum()')
                    lines.append(f'expected_freq = [n_total / len(obs)] * len(obs)  # uniform; adjust if needed')
                    lines.append(f'chi2, p_val = sp.chisquare(obs.values, f_exp=expected_freq)')
                    lines.append(f'print(f"GoF Chi2 = {{chi2:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "fisher" in test_lower:
                    lines.append(f'ct = pd.crosstab(df["{col1}"], df["{col2}"])')
                    lines.append(f'if ct.shape == (2, 2):')
                    lines.append(f'    odds_ratio, p_val = sp.fisher_exact(ct.values, alternative="two-sided")')
                    lines.append(f'    print(f"Fisher OR = {{odds_ratio:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "one-way anova" in test_lower or "anova" in test_lower:
                    lines.append(f'groups = df["{col2}"].dropna().unique()')
                    lines.append(f'group_data = [df.loc[df["{col2}"] == g, "{col1}"].dropna().astype(float) for g in groups]')
                    lines.append(f'f_stat, p_val = sp.f_oneway(*group_data)')
                    lines.append(f'print(f"ANOVA F = {{f_stat:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "welch" in test_lower and "anova" in test_lower:
                    lines.append(f'# Welch ANOVA requires pingouin or statsmodels')
                    lines.append(f'try:')
                    lines.append(f'    import pingouin as pg')
                    lines.append(f'    result_w = pg.welch_anova(dv="{col1}", between="{col2}", data=df.dropna(subset=["{col1}","{col2}"]))')
                    lines.append(f'    print(result_w.to_string())')
                    lines.append(f'except ImportError:')
                    lines.append(f'    print("Install pingouin for Welch ANOVA:  pip install pingouin")')
                elif "friedman" in test_lower:
                    conds = res.get("conditions") or []
                    cond_str = repr(conds) if conds else 'list(df.select_dtypes("number").columns[:3])'
                    lines.append(f'cond_cols = {cond_str}')
                    lines.append(f'clean = df[cond_cols].dropna()')
                    lines.append(f'fr_stat, p_val = sp.friedmanchisquare(*[clean[c].values for c in cond_cols])')
                    lines.append(f'print(f"Friedman chi2 = {{fr_stat:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "kruskal" in test_lower:
                    lines.append(f'groups = df["{col2}"].dropna().unique()')
                    lines.append(f'group_data = [df.loc[df["{col2}"] == g, "{col1}"].dropna().astype(float).values for g in groups]')
                    lines.append(f'h_stat, p_val = sp.kruskal(*group_data)')
                    lines.append(f'print(f"Kruskal-Wallis H = {{h_stat:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "mann-whitney" in test_lower or "mann whitney" in test_lower:
                    grp1 = res.get("group1", "Group1")
                    grp2 = res.get("group2", "Group2")
                    lines.append(f'g1 = df.loc[df["{col2}"] == "{grp1}", "{col1}"].dropna().astype(float).values')
                    lines.append(f'g2 = df.loc[df["{col2}"] == "{grp2}", "{col1}"].dropna().astype(float).values')
                    lines.append(f'u_stat, p_val = sp.mannwhitneyu(g1, g2, alternative="two-sided")')
                    lines.append(f'print(f"Mann-Whitney U = {{u_stat:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "wilcoxon" in test_lower:
                    lines.append(f'paired = df[["{col1}", "{col2}"]].dropna().astype(float)')
                    lines.append(f'stat, p_val = sp.wilcoxon(paired["{col1}"].values, paired["{col2}"].values, alternative="two-sided")')
                    lines.append(f'print(f"Wilcoxon W = {{stat:.4f}}, p = {{p_val:.6f}}, n = {n}")')
                elif "linear regression" in test_lower:
                    indeps = res.get("independents", [col2] if col2 else [])
                    indeps_str = repr(indeps)
                    lines.append(f'import statsmodels.api as sm')
                    lines.append(f'data_r = df[["{col1}"] + {indeps_str}].dropna()')
                    lines.append(f'X = sm.add_constant(data_r[{indeps_str}].astype(float))')
                    lines.append(f'y = data_r["{col1}"].astype(float)')
                    lines.append(f'model = sm.OLS(y, X).fit()')
                    lines.append(f'print(f"R2 = {{model.rsquared:.4f}}, F = {{model.fvalue:.4f}}, p = {{model.f_pvalue:.6f}}, n = {n}")')
                elif "logistic regression" in test_lower:
                    indeps = res.get("independents", [col2] if col2 else [])
                    indeps_str = repr(indeps)
                    lines.append(f'import statsmodels.api as sm')
                    lines.append(f'data_r = df[["{col1}"] + {indeps_str}].dropna()')
                    lines.append(f'X = sm.add_constant(data_r[{indeps_str}].astype(float))')
                    lines.append(f'y = data_r["{col1}"].astype(int)')
                    lines.append(f'model = sm.Logit(y, X).fit(disp=0)')
                    lines.append(f'print(f"Pseudo-R2 = {{model.prsquared:.4f}}, AIC = {{model.aic:.2f}}, n = {n}")')
                elif "pca" in test_lower or "principal component" in test_lower:
                    pca_cols = res.get("columns") or []
                    pca_str  = repr(pca_cols) if pca_cols else 'list(df.select_dtypes("number").columns)'
                    lines.append(f'from sklearn.preprocessing import StandardScaler')
                    lines.append(f'from sklearn.decomposition import PCA')
                    lines.append(f'pca_cols = {pca_str}')
                    lines.append(f'clean_pca = df[pca_cols].dropna()')
                    lines.append(f'X_scaled = StandardScaler().fit_transform(clean_pca)')
                    lines.append(f'pca = PCA()')
                    lines.append(f'pca.fit(X_scaled)')
                    lines.append(f'print(f"Explained variance ratio: {{pca.explained_variance_ratio_.round(4)}}")')
                    lines.append(f'print(f"Kaiser components (eigenvalue >= 1): {{(pca.explained_variance_ >= 1).sum()}}")')
                elif "cronbach" in test_lower:
                    items = res.get("items", [])
                    items_str = repr(items) if items else '[]'
                    lines.append(f'items = {items_str}')
                    lines.append(f'data_c = df[items].apply(pd.to_numeric, errors="coerce").dropna()')
                    lines.append(f'k = len(items)')
                    lines.append(f'item_var = data_c.var(axis=0, ddof=1)')
                    lines.append(f'total_var = data_c.sum(axis=1).var(ddof=1)')
                    lines.append(f'alpha = (k / (k-1)) * (1 - item_var.sum() / total_var)')
                    lines.append(f'print(f"Cronbach alpha = {{alpha:.4f}}, n_items = {{k}}, n = {n}")')
                else:
                    lines.append(f'# Test: {test}')
                    lines.append(f'# p-value reported: {p}  |  n = {n}')
                    lines.append(f'# Re-run using the appropriate scipy/statsmodels function for this test.')
                    lines.append(f'print("Analysis {i} ({test}): p = {p}, n = {n}")')

                lines.append(f'print()')

        else:
            lines += [
                f'',
                f'# ── Statistical Analyses ─────────────────────────────────────────────────────',
                f'# No analyses were recorded for this session.',
            ]

        lines += [
            f'',
            f'# ── End of Reproducibility Script ────────────────────────────────────────────',
            f'print("="*60)',
            f'print("Reproducibility script completed.")',
            f'print("="*60)',
        ]

        return "\n".join(lines)

    _repro_script = _build_repro_script(_stem, log, analyses)
    _repro_filename = f"{_stem}_reproducibility.py"

    # Preview
    with st.expander("👁 Preview reproducibility script", expanded=False):
        st.code(_repro_script, language="python")

    st.download_button(
        label="📥 Download Reproducibility Script (.py)",
        data=_repro_script.encode("utf-8"),
        file_name=_repro_filename,
        mime="text/x-python",
        use_container_width=True,
        help="A standalone Python script that re-runs every analysis from this session.",
    )

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
        <strong>Data Privacy:</strong> Your data is processed entirely in memory during this session.
        Files are not stored on any server. When you close the browser tab or reset the app,
        all uploaded data is cleared from memory.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: TEACHING MODE
# ══════════════════════════════════════════════════════════════════════════════
elif "Teaching" in page:
    st.markdown('<div class="section-title">📚 Teaching Mode — Statistical Literacy & Method Guide</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <strong>Teaching Mode</strong> explains the statistical concepts, methods, and assumptions
        used throughout this platform. Use it to build your understanding before running analyses,
        to check assumptions for a method you are considering, or to prepare for your viva.
        If a dataset is loaded, the <strong>In Your Data</strong> tab connects concepts directly
        to your own variables.
    </div>
    """, unsafe_allow_html=True)

    # ── Tab layout ─────────────────────────────────────────────────────────────
    tab_concepts, tab_methods, tab_assumptions, tab_glossary, tab_quiz, tab_in_data = st.tabs([
        "📖  Concepts",
        "🔬  Method Selection",
        "✔  Assumptions",
        "📖  Glossary",
        "❓  Self-Quiz",
        "🔗  In Your Data",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — STATISTICAL CONCEPTS
    # ══════════════════════════════════════════════════════════════════════════
    with tab_concepts:
        st.markdown("### Core Statistical Concepts")
        st.markdown("Select a topic to expand it.")

        with st.expander("📏  Scales of Measurement", expanded=True):
            st.markdown("""
**Understanding the scale of each variable determines which statistical methods are valid.**

| Scale | Description | Example | Arithmetic allowed |
|-------|-------------|---------|-------------------|
| **Nominal** | Named categories with no order | Gender, Blood Group | Count only |
| **Ordinal** | Ordered categories; gaps between ranks are unequal | Likert scale (1–5), Education level | Rank, median |
| **Interval** | Equal intervals; no true zero | Temperature (°C), IQ score | Mean, SD |
| **Ratio** | Equal intervals with a true zero | Height, Income, Age | All operations |

**Why it matters:**
- Calculating a mean on a *nominal* variable (e.g., average gender) is meaningless.
- Using a parametric test (e.g., t-test) on a 3-point *ordinal* scale is debated — ordinal data with 5+ well-spaced levels is often treated as interval in social science research.
            """)

        with st.expander("📊  Descriptive vs. Inferential Statistics"):
            st.markdown("""
**Descriptive statistics** summarise the data you have collected.
**Inferential statistics** draw conclusions about a population from a sample.

**Descriptive — what you compute:**
- Measures of centre: Mean, Median, Mode
- Measures of spread: Standard Deviation, Variance, Range, IQR
- Shape: Skewness (symmetry), Kurtosis (tail weight)
- Frequencies and proportions for categorical variables

**Inferential — what you ask:**
- Is there a significant difference between groups? → t-test, ANOVA
- Is there a significant association between variables? → Correlation, Chi-square
- Can I predict one variable from another? → Regression
- Does my sample distribution fit a theoretical distribution? → Goodness-of-fit

**The p-value** is the probability of obtaining results at least as extreme as observed, *if the null hypothesis were true*.
A p-value < 0.05 means the result is statistically significant at the 5% level — it does not measure the size or importance of the effect.
            """)

        with st.expander("🎯  Effect Size — Why It Matters"):
            st.markdown("""
Statistical significance (p < 0.05) tells you an effect *exists* in your sample.
**Effect size** tells you how *large* or *practically important* that effect is.

| Measure | Used with | Small | Medium | Large |
|---------|-----------|-------|--------|-------|
| **Cohen's d** | t-tests | 0.2 | 0.5 | 0.8 |
| **r (correlation)** | Pearson / Spearman | 0.1 | 0.3 | 0.5 |
| **η² (eta squared)** | ANOVA | 0.01 | 0.06 | 0.14 |
| **Cramér's V** | Chi-square | 0.1 | 0.3 | 0.5 |
| **R² (R-squared)** | Regression | 0.02 | 0.13 | 0.26 |

**Example:** A drug study with 10,000 participants might show p < 0.001 for a drug that reduces blood pressure by 1 mmHg (negligible effect size). Always report effect size alongside p-values.
            """)

        with st.expander("🔔  Normal Distribution & Why It Matters"):
            st.markdown("""
Many parametric tests assume that the data (or the residuals) follow a **normal (Gaussian) distribution**.

**Properties of a normal distribution:**
- Symmetric bell curve centred on the mean
- Mean = Median = Mode
- ~68% of values fall within 1 SD; ~95% within 2 SD; ~99.7% within 3 SD
- Skewness ≈ 0, Kurtosis ≈ 3 (excess kurtosis ≈ 0)

**How to check normality:**
1. **Histogram** — does it look bell-shaped?
2. **Q-Q plot** — do points fall on the diagonal line?
3. **Shapiro-Wilk test** — p > 0.05 suggests normality (best for n < 50)
4. **Kolmogorov-Smirnov test** — p > 0.05 suggests normality (larger samples)

**Central Limit Theorem:** For n ≥ 30, the sampling distribution of the mean is approximately normal *regardless* of the underlying distribution. This is why parametric tests remain valid with large samples even when raw data are non-normal.
            """)

        with st.expander("🔗  Correlation vs. Causation"):
            st.markdown("""
**Correlation** measures the strength and direction of a *linear relationship* between two variables.
It does **not** imply that one variable *causes* the other.

**Why correlation ≠ causation:**
- A third variable (confounder) may drive both.
- The relationship may be coincidental (spurious correlation).
- Causation requires temporal precedence, controlled experiments, and mechanistic explanation.

**To establish causation you need:**
1. A controlled experiment (ideally randomised)
2. Temporal precedence (cause precedes effect)
3. Elimination of confounders
4. A plausible mechanism

**Example:** Ice cream sales and drowning rates are positively correlated — because both increase in summer (a confounding variable).
            """)

        with st.expander("📐  Null Hypothesis Significance Testing (NHST)"):
            st.markdown("""
**The standard framework for inferential statistics:**

1. **Null Hypothesis (H₀):** assumes no effect, no difference, no association.
   Example: "There is no difference in exam scores between Group A and Group B."

2. **Alternative Hypothesis (H₁ / Hₐ):** the effect/difference you are testing for.
   Example: "Group A scores differ from Group B scores."

3. **Test statistic:** a number calculated from your data (t, F, chi², r, etc.) that measures how far the data depart from H₀.

4. **p-value:** the probability of obtaining a test statistic as extreme as yours *if H₀ were true*.

5. **Decision rule:** if p < α (typically 0.05), reject H₀.

**Common misinterpretations of p-values:**
- ❌ "p = 0.03 means there is a 3% chance the null hypothesis is true." (Wrong)
- ❌ "p > 0.05 means there is no effect." (Wrong — it means insufficient evidence to reject H₀)
- ✅ "p = 0.03 means, if H₀ were true, there is a 3% probability of observing results this extreme by chance alone."
            """)

        with st.expander("📦  Parametric vs. Non-Parametric Tests"):
            st.markdown("""
| Feature | Parametric | Non-Parametric |
|---------|-----------|----------------|
| Assumption | Data normally distributed | No normality required |
| Data type | Interval or Ratio | Ordinal, Nominal, or non-normal |
| Power | Higher (more sensitive) | Lower (more conservative) |
| Sample size | Robust with n ≥ 30 | Suitable for small samples |

**When to choose non-parametric:**
- Ordinal data (Likert scales with few levels)
- Small sample (n < 30) with non-normal data
- Presence of extreme outliers that cannot be removed
- Distribution is heavily skewed

**Equivalent test pairs:**
| Parametric | Non-Parametric equivalent |
|-----------|--------------------------|
| Independent t-test | Mann-Whitney U |
| Paired t-test | Wilcoxon Signed-Rank |
| One-way ANOVA | Kruskal-Wallis H |
| Repeated-measures ANOVA | Friedman Test |
| Pearson r | Spearman rho / Kendall tau |
            """)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — METHOD SELECTION GUIDE
    # ══════════════════════════════════════════════════════════════════════════
    with tab_methods:
        st.markdown("### Which Statistical Method Should I Use?")
        st.markdown("Answer the questions below to find the right test for your research question.")

        _q1 = st.selectbox(
            "1. What is your research question type?",
            [
                "— select —",
                "Describe / summarise data (no hypothesis)",
                "Compare groups (is there a difference?)",
                "Test association / relationship between variables",
                "Predict one variable from others",
                "Reduce dimensionality / find underlying factors",
                "Test reliability / consistency of a scale",
            ],
            key="tm_q1",
        )

        if _q1 == "Describe / summarise data (no hypothesis)":
            st.markdown("""
<div class="result-box">
<strong>Recommended: Descriptive Statistics</strong><br><br>
Use <strong>mean, median, mode</strong> for centre; <strong>SD, IQR, range</strong> for spread.<br>
Use <strong>frequency tables and bar charts</strong> for categorical variables.<br>
Use <strong>histograms, box plots</strong> for distributions.<br><br>
No hypothesis testing needed — you are characterising your sample.
</div>
            """, unsafe_allow_html=True)

        elif _q1 == "Compare groups (is there a difference?)":
            _q2 = st.selectbox("2. How many groups are you comparing?",
                               ["— select —", "2 groups", "3 or more groups"], key="tm_q2a")
            if _q2 != "— select —":
                _q3 = st.selectbox("3. Are the groups independent or related (paired/repeated)?",
                                   ["— select —", "Independent (different participants)", "Related / paired / repeated measures"],
                                   key="tm_q3a")
                if _q3 != "— select —":
                    _q4 = st.selectbox("4. What is the scale of your outcome variable?",
                                       ["— select —", "Interval / Ratio (continuous)", "Ordinal (ranked)", "Nominal (categories)"],
                                       key="tm_q4a")
                    if _q4 != "— select —":
                        _rec = ""
                        if _q2 == "2 groups" and _q3 == "Independent (different participants)":
                            if "Interval" in _q4 or "Ratio" in _q4:
                                _rec = ("**Independent Samples t-test** (parametric) — assumes normality and homogeneity of variance.\n\n"
                                        "Non-normal data or small sample: **Mann-Whitney U test**.")
                            elif "Ordinal" in _q4:
                                _rec = "**Mann-Whitney U test** — compares medians/distributions for ordinal/non-normal data."
                            else:
                                _rec = "**Chi-square test of independence** — compares proportions of a nominal outcome across 2 groups."
                        elif _q2 == "2 groups" and "Related" in _q3:
                            if "Interval" in _q4 or "Ratio" in _q4:
                                _rec = ("**Paired Samples t-test** (parametric) — assumes differences are normally distributed.\n\n"
                                        "Non-normal differences: **Wilcoxon Signed-Rank test**.")
                            else:
                                _rec = "**Wilcoxon Signed-Rank test** — non-parametric paired comparison."
                        elif "3 or more" in _q2 and "Independent" in _q3:
                            if "Interval" in _q4 or "Ratio" in _q4:
                                _rec = ("**One-Way ANOVA** (parametric) — assumes normality and homogeneity of variance.\n\n"
                                        "Non-normal: **Kruskal-Wallis H test**.\n\n"
                                        "Unequal variances: **Welch's ANOVA**.")
                            else:
                                _rec = "**Kruskal-Wallis H test** — non-parametric equivalent of one-way ANOVA."
                        elif "3 or more" in _q2 and "Related" in _q3:
                            if "Interval" in _q4 or "Ratio" in _q4:
                                _rec = ("**Repeated-Measures ANOVA** — compares the same participants across 3+ conditions.\n\n"
                                        "Non-parametric: **Friedman Test**.")
                            else:
                                _rec = "**Friedman Test** — non-parametric repeated-measures comparison."
                        if _rec:
                            st.markdown(f'<div class="result-box"><strong>Recommended Method:</strong><br><br>{_rec}</div>',
                                        unsafe_allow_html=True)

        elif _q1 == "Test association / relationship between variables":
            _q2b = st.selectbox("2. What types of variables are involved?",
                                ["— select —",
                                 "Both continuous (interval/ratio)",
                                 "Both ordinal",
                                 "One continuous, one binary (0/1)",
                                 "Both categorical (nominal)"],
                                key="tm_q2b")
            if _q2b != "— select —":
                if "Both continuous" in _q2b:
                    st.markdown("""
<div class="result-box">
<strong>Recommended: Pearson Correlation</strong><br><br>
Measures linear association between two continuous variables.<br>
Produces a correlation coefficient r (−1 to +1).<br><br>
<strong>Assumptions:</strong> both variables approximately normal, linear relationship, no extreme outliers.<br><br>
<strong>Non-parametric alternative:</strong> Spearman rho (rank-based; robust to outliers and non-normality).
</div>
                    """, unsafe_allow_html=True)
                elif "Both ordinal" in _q2b:
                    st.markdown("""
<div class="result-box">
<strong>Recommended: Spearman Rank Correlation (or Kendall's Tau)</strong><br><br>
Spearman rho measures monotonic association between two ordinal or ranked variables.<br>
Kendall's tau is preferred when there are many tied ranks or small samples.
</div>
                    """, unsafe_allow_html=True)
                elif "binary" in _q2b:
                    st.markdown("""
<div class="result-box">
<strong>Recommended: Point-Biserial Correlation</strong><br><br>
Measures association between one continuous variable and one binary (0/1) variable.<br>
Mathematically equivalent to Pearson r when one variable is dichotomous.
</div>
                    """, unsafe_allow_html=True)
                elif "categorical" in _q2b:
                    st.markdown("""
<div class="result-box">
<strong>Recommended: Chi-Square Test of Independence</strong><br><br>
Tests whether two categorical variables are independent.<br>
Reports Cramér's V as the effect size measure.<br><br>
<strong>Assumption:</strong> Expected cell frequency ≥ 5 in 80% of cells.<br><br>
<strong>Small samples (2×2 table):</strong> Fisher's Exact Test.
</div>
                    """, unsafe_allow_html=True)

        elif _q1 == "Predict one variable from others":
            _q2c = st.selectbox("2. What is the scale of the outcome variable you want to predict?",
                                ["— select —",
                                 "Continuous (interval/ratio)",
                                 "Binary (yes/no, 0/1)"],
                                key="tm_q2c")
            if _q2c != "— select —":
                if "Continuous" in _q2c:
                    st.markdown("""
<div class="result-box">
<strong>Recommended: Linear Regression (OLS)</strong><br><br>
Predicts a continuous outcome from one or more predictor variables.<br>
Reports R² (proportion of variance explained), F-statistic, and beta coefficients.<br><br>
<strong>Key assumptions:</strong>
<ul>
<li>Linear relationship between predictors and outcome</li>
<li>Residuals are normally distributed</li>
<li>Homoscedasticity (constant variance of residuals)</li>
<li>Independence of observations</li>
<li>No severe multicollinearity among predictors</li>
</ul>
</div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
<div class="result-box">
<strong>Recommended: Logistic Regression</strong><br><br>
Predicts the probability of a binary outcome from one or more predictor variables.<br>
Reports pseudo-R² (McFadden), AIC, and odds ratios.<br><br>
<strong>Key assumptions:</strong>
<ul>
<li>Binary outcome variable (0 or 1)</li>
<li>Linear relationship between predictors and log-odds of the outcome</li>
<li>No severe multicollinearity</li>
<li>Large sample (rule of thumb: ≥ 10 events per predictor)</li>
</ul>
</div>
                    """, unsafe_allow_html=True)

        elif _q1 == "Reduce dimensionality / find underlying factors":
            st.markdown("""
<div class="result-box">
<strong>Recommended: Principal Component Analysis (PCA)</strong><br><br>
PCA reduces many correlated variables to a smaller set of uncorrelated components that capture most of the variance.<br><br>
<strong>Use PCA when you want to:</strong>
<ul>
<li>Reduce the number of variables before regression</li>
<li>Identify which variables cluster together</li>
<li>Remove multicollinearity</li>
</ul>
<strong>Key steps:</strong>
<ol>
<li>Standardise all variables (zero mean, unit variance)</li>
<li>Compute the covariance/correlation matrix</li>
<li>Extract eigenvalues and eigenvectors</li>
<li>Retain components with eigenvalue ≥ 1 (Kaiser criterion)</li>
<li>Examine cumulative explained variance (aim for ≥ 70%)</li>
</ol>
</div>
            """, unsafe_allow_html=True)

        elif _q1 == "Test reliability / consistency of a scale":
            st.markdown("""
<div class="result-box">
<strong>Recommended: Cronbach's Alpha</strong><br><br>
Measures the internal consistency of a multi-item scale (e.g., a 5-item Likert questionnaire).<br><br>
<strong>Interpretation:</strong>
<ul>
<li>α ≥ 0.9: Excellent</li>
<li>0.8 ≤ α < 0.9: Good</li>
<li>0.7 ≤ α < 0.8: Acceptable</li>
<li>0.6 ≤ α < 0.7: Questionable</li>
<li>α < 0.6: Poor (consider revising or removing items)</li>
</ul>
<strong>Assumptions:</strong> items measure the same construct; tau-equivalence (equal factor loadings).
</div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Quick Reference Decision Table")
        st.table(pd.DataFrame([
            {"Research Question": "Summarise data",      "Variables": "Any",                        "Recommended Test": "Descriptive Statistics"},
            {"Research Question": "Compare 2 indep. groups", "Variables": "1 continuous outcome",   "Recommended Test": "Independent t-test / Mann-Whitney U"},
            {"Research Question": "Compare 2 paired groups", "Variables": "1 continuous outcome",   "Recommended Test": "Paired t-test / Wilcoxon"},
            {"Research Question": "Compare 3+ indep. groups","Variables": "1 continuous outcome",   "Recommended Test": "One-way ANOVA / Kruskal-Wallis"},
            {"Research Question": "Compare 3+ conditions",   "Variables": "1 continuous outcome",   "Recommended Test": "Repeated-Measures ANOVA / Friedman"},
            {"Research Question": "Linear relationship",     "Variables": "2 continuous",            "Recommended Test": "Pearson r / Spearman rho"},
            {"Research Question": "Association",             "Variables": "2 categorical",           "Recommended Test": "Chi-square / Fisher's Exact"},
            {"Research Question": "Predict outcome",         "Variables": "Continuous outcome",      "Recommended Test": "Linear Regression"},
            {"Research Question": "Predict binary outcome",  "Variables": "Binary outcome",          "Recommended Test": "Logistic Regression"},
            {"Research Question": "Dimensionality reduction","Variables": "Multiple continuous",     "Recommended Test": "PCA"},
            {"Research Question": "Scale reliability",       "Variables": "Multiple Likert items",   "Recommended Test": "Cronbach's Alpha"},
        ]))

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — ASSUMPTIONS EXPLAINER
    # ══════════════════════════════════════════════════════════════════════════
    with tab_assumptions:
        st.markdown("### Test Assumptions — What to Check Before You Run")

        _assump_test = st.selectbox(
            "Select a statistical method to see its assumptions:",
            [
                "— select —",
                "Pearson Correlation",
                "Spearman Rank Correlation",
                "Independent Samples t-test",
                "Paired Samples t-test",
                "Mann-Whitney U Test",
                "One-Way ANOVA",
                "Kruskal-Wallis H Test",
                "Chi-Square Test of Independence",
                "Fisher's Exact Test",
                "Simple / Multiple Linear Regression",
                "Logistic Regression",
                "PCA (Principal Component Analysis)",
                "Cronbach's Alpha",
            ],
            key="tm_assump_sel",
        )

        _ASSUMPTIONS = {
            "Pearson Correlation": {
                "purpose": "Measures linear association between two continuous variables.",
                "assumptions": [
                    ("Both variables are continuous (interval or ratio scale)", "Scale of measurement"),
                    ("Linear relationship between X and Y (check scatter plot)", "Linearity"),
                    ("Both variables are approximately normally distributed (esp. for small n)", "Normality"),
                    ("No extreme outliers (outliers inflate or deflate r)", "Outliers"),
                    ("Each observation is independent", "Independence"),
                ],
                "how_to_check": [
                    "Scatter plot — look for a linear (not curved) pattern",
                    "Histogram / Q-Q plot for each variable",
                    "Shapiro-Wilk test if n < 50",
                    "Box plot to detect outliers",
                ],
                "violation": "Use Spearman rho if normality or linearity cannot be assured.",
            },
            "Spearman Rank Correlation": {
                "purpose": "Measures monotonic (not necessarily linear) association; rank-based.",
                "assumptions": [
                    ("Both variables are at least ordinal", "Scale of measurement"),
                    ("Monotonic relationship between X and Y", "Monotonicity"),
                    ("No excessive ties in ranks (use Kendall tau if many ties)", "Ties"),
                ],
                "how_to_check": [
                    "Scatter plot — look for a consistent direction (monotonic), not necessarily a straight line",
                    "Check for many tied values if ordinal data",
                ],
                "violation": "Very robust — violations are rare. Use Kendall tau for heavily tied ordinal data.",
            },
            "Independent Samples t-test": {
                "purpose": "Compares means of two independent groups on a continuous outcome.",
                "assumptions": [
                    ("Outcome variable is continuous (interval/ratio)", "Scale"),
                    ("Two independent groups (participants appear in only one group)", "Independence"),
                    ("Outcome is approximately normally distributed within each group", "Normality"),
                    ("Homogeneity of variance — both groups have similar variance (Levene's test)", "Equal variances"),
                ],
                "how_to_check": [
                    "Shapiro-Wilk per group (n < 50) or histogram",
                    "Levene's test: p > 0.05 → variances are equal → use pooled t-test; p < 0.05 → use Welch's t-test",
                    "Box plots per group for visual check",
                ],
                "violation": "Non-normal data or unequal variances → use Mann-Whitney U or Welch's t-test.",
            },
            "Paired Samples t-test": {
                "purpose": "Compares means of two related measurements (before/after, matched pairs).",
                "assumptions": [
                    ("The differences between pairs are continuous", "Scale"),
                    ("Observations are paired / matched (same participant or matched unit)", "Pairing"),
                    ("The *differences* (post − pre) are approximately normally distributed", "Normality of differences"),
                ],
                "how_to_check": [
                    "Compute difference scores (col2 − col1)",
                    "Shapiro-Wilk on the difference scores",
                    "Histogram of difference scores",
                ],
                "violation": "Non-normal differences → use Wilcoxon Signed-Rank test.",
            },
            "Mann-Whitney U Test": {
                "purpose": "Non-parametric comparison of two independent groups; compares distributions / medians.",
                "assumptions": [
                    ("Outcome variable is at least ordinal", "Scale"),
                    ("Two independent groups", "Independence"),
                    ("Similar shape of distribution in both groups (for median interpretation)", "Distribution shape"),
                ],
                "how_to_check": [
                    "Box plots or histograms per group — check whether shapes look similar",
                    "No normality check needed (this is a non-parametric test)",
                ],
                "violation": "Very robust. If distributions are very differently shaped, interpret as difference in distributions, not medians.",
            },
            "One-Way ANOVA": {
                "purpose": "Compares means of three or more independent groups on a continuous outcome.",
                "assumptions": [
                    ("Outcome variable is continuous (interval/ratio)", "Scale"),
                    ("Three or more independent groups", "Independence"),
                    ("Outcome is approximately normally distributed within each group", "Normality"),
                    ("Homogeneity of variance across groups (Levene's test)", "Equal variances"),
                ],
                "how_to_check": [
                    "Shapiro-Wilk per group (or use Central Limit Theorem for n ≥ 30 per group)",
                    "Levene's test: p > 0.05 → homogeneity satisfied",
                    "Box plots per group",
                ],
                "violation": ("Non-normal: Kruskal-Wallis H. "
                              "Unequal variances: Welch's ANOVA. "
                              "Significant ANOVA → run post-hoc tests (Tukey HSD, Bonferroni) to find which pairs differ."),
            },
            "Kruskal-Wallis H Test": {
                "purpose": "Non-parametric equivalent of one-way ANOVA; compares 3+ independent groups.",
                "assumptions": [
                    ("Outcome variable is at least ordinal", "Scale"),
                    ("Three or more independent groups", "Independence"),
                    ("Similar distribution shape across groups for median interpretation", "Shape"),
                ],
                "how_to_check": [
                    "Box plots per group",
                    "No normality assumption needed",
                ],
                "violation": "Very robust. Post-hoc pairwise Mann-Whitney U (with Bonferroni correction) for follow-up comparisons.",
            },
            "Chi-Square Test of Independence": {
                "purpose": "Tests whether two categorical variables are statistically independent.",
                "assumptions": [
                    ("Both variables are categorical (nominal or ordinal)", "Scale"),
                    ("Observations are independent", "Independence"),
                    ("Expected cell frequency ≥ 5 in at least 80% of cells", "Expected frequencies"),
                    ("Each observation contributes to only one cell", "Mutual exclusivity"),
                ],
                "how_to_check": [
                    "Print the contingency table and the expected frequency table",
                    "Count cells where expected < 5 — if more than 20%, use Fisher's Exact Test",
                ],
                "violation": "Expected frequencies < 5 in > 20% of cells → use Fisher's Exact Test (2×2 tables) or merge categories.",
            },
            "Fisher's Exact Test": {
                "purpose": "Exact test of independence for 2×2 contingency tables with small expected frequencies.",
                "assumptions": [
                    ("2×2 contingency table (two binary categorical variables)", "Table size"),
                    ("Observations are independent", "Independence"),
                    ("Fixed marginal totals (hypergeometric model)", "Marginals"),
                ],
                "how_to_check": [
                    "Verify table is 2×2",
                    "Preferred over chi-square when any expected cell frequency < 5",
                ],
                "violation": "For tables larger than 2×2 with small expected frequencies, consider collapsing categories.",
            },
            "Simple / Multiple Linear Regression": {
                "purpose": "Predicts a continuous outcome from one or more predictor variables.",
                "assumptions": [
                    ("Outcome variable is continuous (interval/ratio)", "Scale"),
                    ("Linear relationship between each predictor and the outcome", "Linearity"),
                    ("Residuals are approximately normally distributed", "Normality of residuals"),
                    ("Homoscedasticity — residuals have constant variance across fitted values", "Homoscedasticity"),
                    ("Independence of observations (residuals are uncorrelated)", "Independence"),
                    ("No severe multicollinearity (for multiple regression: VIF < 10)", "No multicollinearity"),
                ],
                "how_to_check": [
                    "Residuals vs. Fitted plot — look for random scatter (homoscedasticity & linearity)",
                    "Q-Q plot of residuals — points should follow the diagonal line (normality)",
                    "Scale-Location plot — points should be randomly spread (homoscedasticity)",
                    "VIF (Variance Inflation Factor) — VIF > 10 signals multicollinearity",
                ],
                "violation": ("Non-linear: add polynomial terms or use non-linear regression. "
                              "Non-normal residuals: try transforming the outcome (log, sqrt). "
                              "Heteroscedasticity: use robust standard errors. "
                              "High VIF: remove or combine correlated predictors."),
            },
            "Logistic Regression": {
                "purpose": "Predicts the probability of a binary outcome (0/1) from predictor variables.",
                "assumptions": [
                    ("Binary outcome variable (0 or 1)", "Scale"),
                    ("Linear relationship between predictors and log-odds of the outcome", "Linearity of log-odds"),
                    ("Independence of observations", "Independence"),
                    ("No severe multicollinearity (VIF < 10)", "No multicollinearity"),
                    ("Adequate sample size (rule of thumb: ≥ 10 events per predictor)", "Sample size"),
                ],
                "how_to_check": [
                    "Box-Tidwell test for linearity of log-odds",
                    "Check VIF for multicollinearity",
                    "Count the number of outcome events per predictor",
                    "Hosmer-Lemeshow test for overall model fit",
                ],
                "violation": ("Non-linear log-odds: add interaction terms or splines. "
                              "High VIF: remove collinear predictors. "
                              "Too few events: use penalised regression (LASSO) or reduce number of predictors."),
            },
            "PCA (Principal Component Analysis)": {
                "purpose": "Reduces many correlated variables into fewer uncorrelated components.",
                "assumptions": [
                    ("Variables are continuous (interval/ratio)", "Scale"),
                    ("Variables are standardised (zero mean, unit variance) before PCA", "Standardisation"),
                    ("Correlations among variables exist (otherwise PCA is uninformative)", "Correlations"),
                    ("Linear relationships among variables (PCA captures linear structure only)", "Linearity"),
                ],
                "how_to_check": [
                    "Correlation matrix heatmap — check that many correlations are moderate to strong",
                    "Kaiser-Meyer-Olkin (KMO) measure ≥ 0.6 indicates suitability",
                    "Bartlett's test of sphericity — p < 0.05 confirms correlations are non-zero",
                ],
                "violation": ("Uncorrelated variables: PCA will not yield interpretable components. "
                              "Non-linear relationships: consider non-linear dimensionality reduction (t-SNE, UMAP)."),
            },
            "Cronbach's Alpha": {
                "purpose": "Measures internal consistency / reliability of a multi-item scale.",
                "assumptions": [
                    ("All items measure the same underlying construct (unidimensionality)", "Construct validity"),
                    ("Items are scored on the same numeric scale", "Scale"),
                    ("Tau-equivalence — items contribute equally to the total score", "Tau-equivalence"),
                ],
                "how_to_check": [
                    "Factor analysis (or PCA) to verify all items load on a single factor",
                    "Inter-item correlation matrix — values should generally be > 0.2",
                    "Alpha if item deleted — if removing an item increases alpha substantially, consider revising that item",
                ],
                "violation": ("Low alpha (< 0.7): items may not measure the same construct, or some items may be poorly worded. "
                              "Check inter-item correlations and consider revising problematic items."),
            },
        }

        if _assump_test != "— select —" and _assump_test in _ASSUMPTIONS:
            _a = _ASSUMPTIONS[_assump_test]
            st.markdown(f"#### {_assump_test}")
            st.markdown(f"**Purpose:** {_a['purpose']}")
            st.markdown("**Assumptions to verify:**")
            _rows = [{"#": i+1, "Assumption": row[0], "Category": row[1]}
                     for i, row in enumerate(_a["assumptions"])]
            st.table(pd.DataFrame(_rows))
            st.markdown("**How to check:**")
            for item in _a["how_to_check"]:
                st.markdown(f"- {item}")
            st.markdown(f"""
<div class="warn-box"><strong>What to do if assumptions are violated:</strong><br>{_a['violation']}</div>
            """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — GLOSSARY
    # ══════════════════════════════════════════════════════════════════════════
    with tab_glossary:
        st.markdown("### Research & Statistics Glossary")

        _glossary_search = st.text_input("🔍 Search glossary", placeholder="e.g. p-value, effect size, regression...",
                                         key="tm_glossary_search")

        _GLOSSARY = [
            ("Alpha (α)", "The significance threshold, typically 0.05. If p < α, reject the null hypothesis."),
            ("Alternative Hypothesis (H₁)", "The hypothesis that an effect, difference, or association exists; what you are testing for."),
            ("ANOVA", "Analysis of Variance. Tests for differences in means across three or more groups. Produces an F-statistic and p-value."),
            ("Central Limit Theorem", "States that the distribution of sample means approaches normality as sample size increases (n ≥ 30), regardless of the population distribution."),
            ("Chi-Square (χ²)", "A test statistic used to assess association between categorical variables or goodness of fit."),
            ("Cohen's d", "Effect size measure for t-tests. d = 0.2 small, 0.5 medium, 0.8 large."),
            ("Confidence Interval (CI)", "A range of values that contains the true population parameter with a stated level of confidence (e.g., 95%)."),
            ("Confounding Variable", "A variable that is related to both the predictor and the outcome, creating a spurious association."),
            ("Correlation Coefficient (r)", "A number from −1 to +1 measuring the strength and direction of a linear relationship."),
            ("Cramér's V", "Effect size for chi-square test. V = 0.1 small, 0.3 medium, 0.5 large."),
            ("Cronbach's Alpha (α)", "Measure of internal consistency / reliability of a multi-item scale. Ranges from 0 to 1."),
            ("Degrees of Freedom (df)", "The number of independent values that can vary in a calculation. Related to sample size and number of parameters estimated."),
            ("Dependent Variable", "The outcome variable you are trying to explain or predict."),
            ("Descriptive Statistics", "Methods that summarise and describe the features of a dataset (mean, SD, frequencies, etc.)."),
            ("Effect Size", "A quantitative measure of the magnitude of a statistical effect, independent of sample size."),
            ("Eta Squared (η²)", "Effect size for ANOVA. η² = 0.01 small, 0.06 medium, 0.14 large."),
            ("F-statistic", "A test statistic used in ANOVA and regression; ratio of explained to unexplained variance."),
            ("Fisher's Exact Test", "Exact test of independence for 2×2 contingency tables, especially with small expected frequencies."),
            ("Friedman Test", "Non-parametric equivalent of repeated-measures ANOVA; compares 3+ related groups."),
            ("Homoscedasticity", "The assumption that residuals (or group variances) have constant variance."),
            ("Hypothesis", "A testable prediction or claim about a population parameter."),
            ("Independent Variable", "A predictor or explanatory variable used to explain or predict the dependent variable."),
            ("Inferential Statistics", "Methods that generalise from a sample to a population, making probabilistic statements."),
            ("Inter-Quartile Range (IQR)", "The range of the middle 50% of data; Q3 − Q1. Robust to outliers."),
            ("Interval Scale", "Scale with equal intervals between values but no true zero (e.g., temperature in °C)."),
            ("Kendall's Tau (τ)", "Non-parametric correlation measure; preferred over Spearman when there are many tied ranks."),
            ("Kruskal-Wallis H Test", "Non-parametric equivalent of one-way ANOVA for three or more independent groups."),
            ("Kurtosis", "Measure of the 'tail heaviness' of a distribution. Normal distribution has kurtosis = 3 (excess kurtosis = 0)."),
            ("Levene's Test", "Tests homogeneity of variance across groups. p > 0.05 indicates equal variances."),
            ("Linear Regression", "Models the linear relationship between a continuous outcome and one or more predictors."),
            ("Logistic Regression", "Models the probability of a binary outcome from predictor variables."),
            ("Mann-Whitney U Test", "Non-parametric test comparing two independent groups; equivalent to independent t-test."),
            ("Mean", "The arithmetic average of a set of values."),
            ("Median", "The middle value when data are ordered; robust to outliers."),
            ("Mode", "The most frequently occurring value."),
            ("Multicollinearity", "High correlation among predictor variables in regression, making coefficients unstable."),
            ("Nominal Scale", "Scale with named categories having no natural order (e.g., blood type, gender)."),
            ("Normal Distribution", "A symmetric, bell-shaped probability distribution defined by mean and standard deviation."),
            ("Null Hypothesis (H₀)", "The hypothesis of no effect, no difference, or no association; what you are testing against."),
            ("Ordinal Scale", "Scale with ordered categories but unequal intervals (e.g., Likert scale, ranking)."),
            ("Outlier", "An observation that lies unusually far from the rest of the data. Can distort means and correlation."),
            ("p-value", "The probability of obtaining a result as extreme as observed, assuming H₀ is true. Not the probability that H₀ is true."),
            ("Paired t-test", "Compares means of two related measurements from the same participants (e.g., before vs. after)."),
            ("Parametric Test", "A test that assumes the data follow a known distribution (usually normal)."),
            ("PCA", "Principal Component Analysis. Reduces multiple correlated variables into fewer uncorrelated components."),
            ("Pearson r", "Parametric correlation coefficient measuring linear association between two continuous variables."),
            ("Point-Biserial Correlation", "Correlation between one continuous and one binary (0/1) variable."),
            ("Population", "The entire group of individuals or observations about which you want to draw conclusions."),
            ("Post-hoc Test", "Tests conducted after a significant ANOVA to identify which specific pairs of groups differ (e.g., Tukey HSD)."),
            ("Power (Statistical Power)", "The probability of correctly rejecting a false null hypothesis. Power = 1 − β."),
            ("R² (R-squared)", "Proportion of variance in the outcome explained by the regression model. Ranges from 0 to 1."),
            ("Ratio Scale", "Scale with equal intervals and a true zero (e.g., height, weight, income). All arithmetic operations valid."),
            ("Regression", "A statistical method for predicting or explaining a dependent variable using one or more independent variables."),
            ("Reliability", "The consistency or repeatability of a measurement instrument."),
            ("Residual", "The difference between an observed value and the value predicted by the model."),
            ("Sample", "A subset of the population selected for measurement."),
            ("Sampling Distribution", "The probability distribution of a statistic (e.g., mean) over many repeated samples."),
            ("Shapiro-Wilk Test", "Tests whether a variable is normally distributed. p > 0.05 suggests normality. Best for n < 50."),
            ("Skewness", "Measure of asymmetry of a distribution. Positive skew = tail to the right; negative = tail to the left."),
            ("Spearman Rho (ρ)", "Non-parametric rank-based correlation; robust to outliers and non-normality."),
            ("Standard Deviation (SD)", "The average distance of data points from the mean; square root of variance."),
            ("Standard Error (SE)", "The standard deviation of the sampling distribution of a statistic (e.g., SE of the mean = SD/√n)."),
            ("Statistical Significance", "A result is statistically significant when p < α. Does not imply practical importance."),
            ("t-statistic", "A test statistic comparing the difference between group means relative to variability."),
            ("t-test (Independent)", "Parametric test comparing means of two independent groups."),
            ("Type I Error (α)", "Rejecting a true null hypothesis (false positive). Controlled by setting α = 0.05."),
            ("Type II Error (β)", "Failing to reject a false null hypothesis (false negative). Reduced by increasing power/sample size."),
            ("Validity", "Whether a test or measurement actually measures what it claims to measure."),
            ("Variance", "The average of the squared deviations from the mean. SD² = Variance."),
            ("VIF (Variance Inflation Factor)", "Measures multicollinearity in regression. VIF > 10 is problematic."),
            ("Welch's t-test", "A variant of the independent t-test that does not assume equal variances."),
            ("Wilcoxon Signed-Rank Test", "Non-parametric equivalent of the paired t-test."),
        ]

        _filtered = [
            (term, defn) for term, defn in _GLOSSARY
            if not _glossary_search.strip()
            or _glossary_search.strip().lower() in term.lower()
            or _glossary_search.strip().lower() in defn.lower()
        ]

        if not _filtered:
            st.info("No matching terms found. Try a different search term.")
        else:
            st.markdown(f"*Showing {len(_filtered)} of {len(_GLOSSARY)} terms.*")
            for term, defn in _filtered:
                with st.expander(f"**{term}**", expanded=False):
                    st.markdown(defn)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5 — SELF-QUIZ
    # ══════════════════════════════════════════════════════════════════════════
    with tab_quiz:
        st.markdown("### Self-Assessment Quiz")
        st.markdown("Test your statistical knowledge. Select an answer for each question then click **Check Answers**.")

        _QUIZ = [
            {
                "q":       "A researcher measures anxiety on a 5-point Likert scale (1 = Not anxious, 5 = Extremely anxious). What is the scale of measurement?",
                "options": ["Nominal", "Ordinal", "Interval", "Ratio"],
                "answer":  "Ordinal",
                "exp":     "Likert scales produce ordered categories, but the intervals between points are not guaranteed to be equal. The correct scale is **Ordinal**.",
            },
            {
                "q":       "A p-value of 0.03 means which of the following?",
                "options": [
                    "There is a 3% chance the null hypothesis is true.",
                    "There is a 97% chance the alternative hypothesis is true.",
                    "If the null hypothesis were true, there is a 3% probability of observing results this extreme by chance.",
                    "The effect size is small.",
                ],
                "answer":  "If the null hypothesis were true, there is a 3% probability of observing results this extreme by chance.",
                "exp":     "The p-value is the probability of the data given H₀ — not the probability that H₀ is true.",
            },
            {
                "q":       "You want to compare exam scores between three independent teaching methods. Which test is most appropriate (data are normally distributed)?",
                "options": ["Independent t-test", "Paired t-test", "One-Way ANOVA", "Chi-Square Test"],
                "answer":  "One-Way ANOVA",
                "exp":     "With 3+ independent groups and a continuous, normally distributed outcome, **One-Way ANOVA** is the correct parametric test.",
            },
            {
                "q":       "Pearson r = −0.72 between study time and anxiety. How do you interpret this?",
                "options": [
                    "Students who study more have higher anxiety.",
                    "There is a strong positive relationship.",
                    "There is a strong negative linear relationship — students who study more tend to report lower anxiety.",
                    "The result is not statistically significant.",
                ],
                "answer":  "There is a strong negative linear relationship — students who study more tend to report lower anxiety.",
                "exp":     "r = −0.72 indicates a **strong negative** linear association (|r| > 0.5 is large by Cohen's benchmarks).",
            },
            {
                "q":       "An ANOVA produces F(2, 57) = 8.43, p = 0.001. What does this tell you?",
                "options": [
                    "All three group means are significantly different from each other.",
                    "At least one group mean differs significantly from the others.",
                    "The effect size is large.",
                    "The data are normally distributed.",
                ],
                "answer":  "At least one group mean differs significantly from the others.",
                "exp":     "A significant ANOVA only tells you that *at least one* group mean differs. Run a **post-hoc test** (e.g., Tukey HSD) to find *which* pairs differ.",
            },
            {
                "q":       "R² = 0.64 in a linear regression. What does this mean?",
                "options": [
                    "The regression slope is 0.64.",
                    "64% of the variance in the outcome is explained by the predictors.",
                    "The p-value is 0.64.",
                    "There are 64 observations.",
                ],
                "answer":  "64% of the variance in the outcome is explained by the predictors.",
                "exp":     "R² (coefficient of determination) represents the proportion of variance in the outcome variable explained by the model.",
            },
            {
                "q":       "Your data have a significant Shapiro-Wilk test (p = 0.01, n = 20) and you want to compare two groups. What should you do?",
                "options": [
                    "Proceed with independent t-test — it is robust.",
                    "Use Mann-Whitney U test instead.",
                    "Increase the sample size to 30.",
                    "Use chi-square test.",
                ],
                "answer":  "Use Mann-Whitney U test instead.",
                "exp":     "With n = 20 and evidence of non-normality, the **Mann-Whitney U test** is the appropriate non-parametric alternative.",
            },
            {
                "q":       "Cronbach's alpha = 0.54 for your 6-item questionnaire. What does this indicate?",
                "options": [
                    "Excellent internal consistency.",
                    "Good internal consistency.",
                    "Poor internal consistency — the items may not measure the same construct.",
                    "The scale has exactly 54% missing data.",
                ],
                "answer":  "Poor internal consistency — the items may not measure the same construct.",
                "exp":     "α < 0.6 is considered **Poor**. Consider examining inter-item correlations and revising or removing items that do not align with the others.",
            },
        ]

        # Initialise quiz state
        if "tm_quiz_submitted" not in st.session_state:
            st.session_state["tm_quiz_submitted"] = False
        if "tm_quiz_answers" not in st.session_state:
            st.session_state["tm_quiz_answers"] = {}

        for qi, question in enumerate(_QUIZ, 1):
            st.markdown(f"**Q{qi}.** {question['q']}")
            selected = st.radio(
                f"Q{qi}",
                options=question["options"],
                key=f"tm_quiz_q{qi}",
                label_visibility="collapsed",
                index=None,
            )
            st.session_state["tm_quiz_answers"][qi] = selected
            st.markdown("")

        col_submit, _ = st.columns([1, 4])
        if col_submit.button("✅ Check Answers", type="primary", key="tm_quiz_submit"):
            st.session_state["tm_quiz_submitted"] = True

        if st.session_state["tm_quiz_submitted"]:
            score = 0
            st.markdown("---")
            st.markdown("### Results")
            for qi, question in enumerate(_QUIZ, 1):
                selected = st.session_state["tm_quiz_answers"].get(qi)
                correct = question["answer"]
                is_correct = selected == correct
                if is_correct:
                    score += 1
                    st.markdown(f"**Q{qi}: ✅ Correct**")
                else:
                    st.markdown(f"**Q{qi}: ❌ Incorrect** — Correct answer: *{correct}*")
                st.markdown(f"> {question['exp']}")
                st.markdown("")

            pct = round(score / len(_QUIZ) * 100)
            if pct == 100:
                st.success(f"🎉 Perfect score! {score}/{len(_QUIZ)} ({pct}%)")
            elif pct >= 75:
                st.success(f"✅ Good score: {score}/{len(_QUIZ)} ({pct}%). Review any incorrect answers above.")
            elif pct >= 50:
                st.warning(f"⚠️ Moderate score: {score}/{len(_QUIZ)} ({pct}%). Study the Concepts and Assumptions tabs.")
            else:
                st.error(f"❌ Score: {score}/{len(_QUIZ)} ({pct}%). Revisit the Concepts tab and read through each method's assumptions.")

            if st.button("🔄 Retake Quiz", key="tm_quiz_reset"):
                st.session_state["tm_quiz_submitted"] = False
                st.session_state["tm_quiz_answers"] = {}
                for qi in range(1, len(_QUIZ) + 1):
                    if f"tm_quiz_q{qi}" in st.session_state:
                        del st.session_state[f"tm_quiz_q{qi}"]
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6 — IN YOUR DATA
    # ══════════════════════════════════════════════════════════════════════════
    with tab_in_data:
        st.markdown("### Teaching Mode — Applied to Your Dataset")

        if st.session_state.raw_df is None:
            st.markdown("""
<div class="warn-box">
No dataset is currently loaded. Upload a file via the <strong>Upload Data</strong> page
to see teaching content connected to your own variables.
</div>
            """, unsafe_allow_html=True)
        else:
            df_td   = _get_df()
            profile_td = st.session_state.profile or _profile(df_td)
            classifications_td = profile_td.get("classifications", {})
            overrides_td = st.session_state.get("user_role_overrides", {})

            # Merge overrides
            _cls = {}
            for col in df_td.columns:
                cl = dict(classifications_td.get(col, {}))
                if col in overrides_td:
                    if "analytical_role" in overrides_td[col]:
                        cl["analytical_role"] = overrides_td[col]["analytical_role"]
                    if "measurement_level" in overrides_td[col]:
                        cl["measurement_level"] = overrides_td[col]["measurement_level"]
                _cls[col] = cl

            st.markdown(f"**Dataset:** `{st.session_state.file_name}` — "
                        f"{len(df_td):,} rows × {len(df_td.columns)} columns")

            # ── Variable type summary ──────────────────────────────────────────
            st.markdown("#### Variable Classification Summary")
            _role_counts = {}
            for col, cl in _cls.items():
                role = cl.get("analytical_role", "unknown")
                _role_counts[role] = _role_counts.get(role, 0) + 1

            _role_rows = [{"Analytical Role": role, "Count": count,
                           "Example Variables": ", ".join(
                               [c for c, cl in _cls.items() if cl.get("analytical_role") == role][:3]
                           )}
                          for role, count in sorted(_role_counts.items())]
            st.table(pd.DataFrame(_role_rows))

            st.markdown("""
<div class="info-box">
<strong>How roles affect which tests are available:</strong><br>
<ul>
<li><strong>substantive_numerical</strong> — eligible for t-tests, ANOVA, Pearson correlation, regression.</li>
<li><strong>categorical / boolean</strong> — eligible for chi-square, frequency tables, group comparisons.</li>
<li><strong>ordinal</strong> — eligible for Spearman, Mann-Whitney, Kruskal-Wallis.</li>
<li><strong>identifier / serial_number / constant</strong> — <em>excluded</em> from all statistical tests (no analytical meaning).</li>
</ul>
If a variable is misclassified, correct it in the <strong>Variable Classification</strong> page.
</div>
            """, unsafe_allow_html=True)

            # ── Method suggestions based on dataset ───────────────────────────
            st.markdown("#### Suggested Analyses for Your Dataset")

            _num_cols = [c for c, cl in _cls.items() if cl.get("analytical_role") == "substantive_numerical"]
            _cat_cols = [c for c, cl in _cls.items() if cl.get("analytical_role") in ("categorical", "boolean")]
            _ord_cols = [c for c, cl in _cls.items() if cl.get("analytical_role") == "ordinal"]

            _suggestions = []
            if len(_num_cols) >= 2:
                _suggestions.append(
                    f"**Pearson Correlation** — you have {len(_num_cols)} continuous variables "
                    f"({', '.join(_num_cols[:3])}{'...' if len(_num_cols) > 3 else ''}). "
                    "Use the Statistical Analysis page to test pairwise linear relationships."
                )
            if _cat_cols and _num_cols:
                _suggestions.append(
                    f"**Independent t-test or ANOVA** — you have categorical group variables "
                    f"({', '.join(_cat_cols[:2])}) and continuous outcome variables. "
                    "Compare means across groups."
                )
            if len(_cat_cols) >= 2:
                _suggestions.append(
                    f"**Chi-Square Test of Independence** — you have multiple categorical variables "
                    f"({', '.join(_cat_cols[:3])}). Test whether they are associated."
                )
            if _num_cols:
                _suggestions.append(
                    f"**Linear Regression** — predict one continuous variable from others. "
                    f"Available continuous columns: {', '.join(_num_cols[:4])}{'...' if len(_num_cols) > 4 else ''}."
                )
            if _ord_cols and len(_num_cols) >= 1:
                _suggestions.append(
                    f"**Spearman Rank Correlation** — you have ordinal variable(s) "
                    f"({', '.join(_ord_cols[:2])}). Use Spearman rather than Pearson for ranked data."
                )
            if len(_num_cols) >= 3:
                _suggestions.append(
                    f"**PCA** — with {len(_num_cols)} continuous variables, PCA can reveal underlying structure "
                    "and reduce dimensionality."
                )

            if _suggestions:
                for sug in _suggestions:
                    st.markdown(f'<div class="step-card">💡 {sug}</div>', unsafe_allow_html=True)
            else:
                st.info("Variable classification is still being processed or no suitable variable combinations were found. "
                        "Try profiling or re-classifying variables on the Variable Classification page.")

            # ── Assumption pre-check for Pearson ──────────────────────────────
            if len(_num_cols) >= 2:
                st.markdown("---")
                st.markdown("#### Quick Normality Check for Continuous Variables")
                st.markdown(
                    "The following table shows skewness and kurtosis for your continuous variables. "
                    "Skewness near 0 and excess kurtosis near 0 suggest approximate normality."
                )
                _norm_rows = []
                for col in _num_cols[:10]:    # limit to first 10 for performance
                    try:
                        series = pd.to_numeric(df_td[col], errors="coerce").dropna()
                        if len(series) < 3:
                            continue
                        sk  = float(series.skew())
                        ku  = float(series.kurt())   # excess kurtosis
                        n   = len(series)
                        flag = ""
                        if abs(sk) > 2:
                            flag += " ⚠️ High skew"
                        if abs(ku) > 7:
                            flag += " ⚠️ High kurtosis"
                        _norm_rows.append({
                            "Variable":       col,
                            "n":              n,
                            "Skewness":       round(sk, 3),
                            "Excess Kurtosis":round(ku, 3),
                            "Note":           flag.strip() or "✅ Likely normal",
                        })
                    except Exception:
                        pass
                if _norm_rows:
                    st.table(pd.DataFrame(_norm_rows))
                    st.caption(
                        "Rule of thumb: |skewness| > 2 or |excess kurtosis| > 7 suggests meaningful non-normality. "
                        "For small samples (n < 50), also run the Shapiro-Wilk test in the Statistical Analysis page."
                    )
