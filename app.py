import streamlit as st
from PIL import Image
import numpy as np
import hashlib
import hmac
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

import config
from ui_animations import animated_heading, animated_result_banner, animated_loading_card
from components.antigravity import antigravity_hero
from database import init_db, get_prediction_feedback, save_prediction_feedback
from registry import (
    get_all_vehicles,
    register_vehicle,
    update_vehicle,
    remove_vehicle,
    normalize_plate,
    lookup_plate,
    is_valid_email
)
from parking_detector import (
    get_vehicle_model,
    get_overhead_vehicle_model,
    detect_vehicles_and_violations
)
from plate_reader import get_plate_detector, get_ocr_reader, read_plate_from_vehicle_crop
from notifications import send_violation_notice, get_recent_audit_logs, is_duplicate_event

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Smart Parking AI",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Premium Glassmorphism Design System & React Bits Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

    :root {
        --sp-bg-base: #070b14;
        --sp-bg-surface: rgba(13, 21, 39, 0.7);
        --sp-bg-surface-elevated: rgba(20, 31, 57, 0.75);
        --sp-bg-surface-hover: rgba(28, 43, 76, 0.8);
        --sp-bg-glass: rgba(15, 23, 42, 0.65);
        --sp-glass-border: rgba(148, 163, 184, 0.14);
        --sp-glass-border-hover: rgba(56, 189, 248, 0.4);
        --sp-glass-blur: 18px;
        --sp-text-primary: #f8fafc;
        --sp-text-secondary: #94a3b8;
        --sp-text-muted: #64748b;
        --sp-accent-cyan: #06b6d4;
        --sp-accent-sky: #38bdf8;
        --sp-accent-blue: #3b82f6;
        --sp-accent-glow: rgba(56, 189, 248, 0.22);
        --sp-danger: #f43f5e;
        --sp-danger-bg: rgba(244, 63, 94, 0.14);
        --sp-danger-border: rgba(244, 63, 94, 0.4);
        --sp-warning: #f59e0b;
        --sp-warning-bg: rgba(245, 158, 11, 0.14);
        --sp-warning-border: rgba(245, 158, 11, 0.4);
        --sp-success: #10b981;
        --sp-success-bg: rgba(16, 185, 129, 0.14);
        --sp-success-border: rgba(16, 185, 129, 0.4);
        --sp-radius-sm: 8px;
        --sp-radius-md: 12px;
        --sp-radius-lg: 16px;
        --sp-radius-xl: 20px;
        --sp-radius-full: 9999px;
        --sp-shadow-glass: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
        --sp-shadow-glow: 0 0 25px rgba(56, 189, 248, 0.22);
        --sp-transition-snappy: 160ms cubic-bezier(0.16, 1, 0.3, 1);
        --sp-transition-smooth: 260ms cubic-bezier(0.2, 0.8, 0.2, 1);
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--sp-text-primary) !important;
    }

    *, *::before, *::after {
        box-sizing: border-box;
    }

    /* Subtle layered ambient aurora canvas background */
    .stApp {
        background-color: #070b14 !important;
        background-image: 
            radial-gradient(ellipse 70% 45% at 20% -5%, rgba(14, 165, 233, 0.12), transparent),
            radial-gradient(ellipse 60% 40% at 85% 15%, rgba(99, 102, 241, 0.10), transparent),
            radial-gradient(circle 500px at 50% 95%, rgba(6, 182, 212, 0.07), transparent),
            radial-gradient(circle 350px at 15% 75%, rgba(59, 130, 246, 0.05), transparent) !important;
        background-attachment: fixed !important;
    }

    .main .block-container {
        width: 100%;
        max-width: 1360px;
        padding: 1.75rem clamp(1rem, 3vw, 2.5rem) 3.5rem;
    }

    /* Clean Streamlit horizontal layout */
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch;
        gap: clamp(0.75rem, 2vw, 1.25rem);
    }
    div[data-testid="column"] {
        min-width: 0 !important;
    }

    /* ------------------------------------------------------------- */
    /* Sidebar: Frosted Glass Panel & Compact Brand Identity         */
    /* ------------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: rgba(9, 14, 26, 0.78) !important;
        backdrop-filter: blur(24px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(190%) !important;
        border-right: 1px solid var(--sp-glass-border) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.45) !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 2rem !important;
    }

    .sp-sidebar-header {
        display: flex;
        flex-direction: column;
        gap: 0.65rem;
        padding-bottom: 1rem;
        margin-bottom: 1.25rem;
        border-bottom: 1px solid var(--sp-glass-border);
    }
    .sp-brand-badge {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .sp-brand-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.25rem;
        height: 2.25rem;
        border-radius: var(--sp-radius-md);
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.25), rgba(37, 99, 235, 0.25));
        border: 1px solid rgba(56, 189, 248, 0.4);
        font-size: 1.15rem;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.25);
    }
    .sp-brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
    }
    .sp-brand-subtitle {
        font-size: 0.76rem;
        color: var(--sp-text-secondary);
        letter-spacing: 0.01em;
    }
    .sp-status-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.22rem 0.65rem;
        border-radius: var(--sp-radius-full);
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6ee7b7;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        width: fit-content;
    }
    .sp-chip-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: sp-dot-pulse 2s infinite ease-in-out;
    }

    /* Distinct Sign out button in Sidebar */
    button[data-testid="baseButton-secondary"]:has(p:contains("Sign out")),
    button[key="btn_sign_out"] {
        background: rgba(244, 63, 94, 0.08) !important;
        border: 1px solid rgba(244, 63, 94, 0.3) !important;
        color: #fda4af !important;
        font-weight: 600 !important;
        transition: all var(--sp-transition-smooth) !important;
    }
    button[data-testid="baseButton-secondary"]:has(p:contains("Sign out")):hover,
    button[key="btn_sign_out"]:hover {
        background: rgba(244, 63, 94, 0.18) !important;
        border-color: #f43f5e !important;
        color: #ffffff !important;
        box-shadow: 0 0 15px rgba(244, 63, 94, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar DataFrame clean display */
    section[data-testid="stSidebar"] div[data-testid="stDataFrame"] {
        border-radius: var(--sp-radius-md) !important;
        border: 1px solid var(--sp-glass-border) !important;
        background: rgba(13, 21, 39, 0.6) !important;
        overflow-x: auto !important;
    }

    /* ------------------------------------------------------------- */
    /* Frosted Glass Panels & Containers                             */
    /* ------------------------------------------------------------- */
    div[data-testid="stVerticalBlockBorderWrapper"],
    .glass-card {
        border-radius: var(--sp-radius-lg) !important;
        background: var(--sp-bg-surface) !important;
        backdrop-filter: blur(var(--sp-glass-blur)) saturate(180%) !important;
        -webkit-backdrop-filter: blur(var(--sp-glass-blur)) saturate(180%) !important;
        border: 1px solid var(--sp-glass-border) !important;
        box-shadow: var(--sp-shadow-glass), inset 0 1px 1px 0 rgba(255, 255, 255, 0.06) !important;
        transition: transform var(--sp-transition-smooth), border-color var(--sp-transition-smooth), box-shadow var(--sp-transition-smooth) !important;
        animation: sp-card-reveal 420ms cubic-bezier(0.16, 1, 0.3, 1) both;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover,
    .glass-card:hover {
        border-color: var(--sp-glass-border-hover) !important;
        transform: translateY(-2px);
        box-shadow: 0 12px 36px 0 rgba(0, 0, 0, 0.55), var(--sp-shadow-glow), inset 0 1px 1px 0 rgba(255, 255, 255, 0.1) !important;
    }

    @keyframes sp-card-reveal {
        from {
            opacity: 0;
            transform: translateY(12px);
            filter: blur(3px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
            filter: blur(0);
        }
    }

    /* Frosted Expanders */
    div[data-testid="stExpander"] {
        background: rgba(13, 21, 39, 0.55) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid var(--sp-glass-border) !important;
        border-radius: var(--sp-radius-md) !important;
        overflow: hidden !important;
        margin-bottom: 0.85rem !important;
        transition: border-color var(--sp-transition-smooth) !important;
    }
    div[data-testid="stExpander"]:hover {
        border-color: rgba(56, 189, 248, 0.3) !important;
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #f1f5f9 !important;
        padding: 0.7rem 0.95rem !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stExpander"] summary:hover {
        background: rgba(56, 189, 248, 0.05) !important;
        color: #38bdf8 !important;
    }
    div[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        padding: 0.95rem 1.15rem !important;
        border-top: 1px solid rgba(148, 163, 184, 0.1) !important;
    }

    /* ------------------------------------------------------------- */
    /* Interactive Controls, Inputs & React Bits Buttons             */
    /* ------------------------------------------------------------- */
    button, [role="button"], input, textarea, [data-baseweb="select"] {
        border-radius: var(--sp-radius-md) !important;
    }
    button:focus-visible, input:focus-visible, textarea:focus-visible,
    [role="button"]:focus-visible {
        outline: 2px solid var(--sp-accent-sky) !important;
        outline-offset: 2px !important;
    }

    button {
        min-height: 2.65rem;
        transition: transform var(--sp-transition-snappy), box-shadow var(--sp-transition-snappy), filter var(--sp-transition-snappy), border-color var(--sp-transition-snappy) !important;
        position: relative;
        overflow: hidden;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
    }
    button:hover {
        filter: brightness(1.05);
        transform: translateY(-1px);
    }
    button:active {
        transform: translateY(0.5px);
        filter: brightness(0.95);
    }

    /* Primary Buttons (Analyze, Sign In, Dispatch) with React Bits Metallic Sheen */
    button[data-testid="baseButton-primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 55%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.35), inset 0 1px 1px rgba(255, 255, 255, 0.25) !important;
    }
    button[data-testid="baseButton-primary"]::after,
    button[kind="primary"]::after {
        content: '';
        position: absolute;
        top: 0;
        left: -140%;
        width: 60%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.28), transparent);
        transform: skewX(-22deg);
        pointer-events: none;
    }
    button[data-testid="baseButton-primary"]:hover::after,
    button[kind="primary"]:hover::after {
        left: 210%;
        transition: left 650ms cubic-bezier(0.16, 1, 0.3, 1);
    }
    button[data-testid="baseButton-primary"]:hover,
    button[kind="primary"]:hover {
        transform: translateY(-1.5px) !important;
        box-shadow: 0 6px 24px rgba(2, 132, 199, 0.5), 0 0 20px rgba(56, 189, 248, 0.35) !important;
    }

    /* Secondary Buttons */
    button[data-testid="baseButton-secondary"],
    button[kind="secondary"] {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid var(--sp-glass-border) !important;
        color: #e2e8f0 !important;
        backdrop-filter: blur(12px) !important;
    }
    button[data-testid="baseButton-secondary"]:hover,
    button[kind="secondary"]:hover {
        border-color: rgba(56, 189, 248, 0.45) !important;
        background: rgba(30, 41, 59, 0.8) !important;
        color: #ffffff !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.25) !important;
    }

    /* ------------------------------------------------------------- */
    /* CRITICAL FIX: Human Review Feedback Buttons (✓ / ✗)          */
    /* Must NEVER truncate to "✓ Co..." or "✗ ..."                   */
    /* ------------------------------------------------------------- */
    .cat-badge-container {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 0.6rem;
        min-height: 2.2rem;
        padding-top: 0.15rem;
        margin-bottom: 0.2rem;
    }
    .cat-title {
        font-weight: 600;
        font-size: 0.96rem;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }

    /* Ensure feedback buttons inside category assessment have ample width and clear text */
    div[data-testid="stExpander"] div[data-testid="stButton"] button {
        min-height: 2.3rem !important;
        min-width: 95px !important;
        padding: 0.3rem 0.65rem !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        letter-spacing: 0.02em !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: var(--sp-radius-sm) !important;
    }
    div[data-testid="stExpander"] div[data-testid="stButton"] button p {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Distinct Emerald Accent for ✓ Correct */
    button[data-testid*="feedback_correct"],
    button[key*="feedback_correct"] {
        background: rgba(16, 185, 129, 0.14) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        color: #34d399 !important;
    }
    button[data-testid*="feedback_correct"]:hover,
    button[key*="feedback_correct"]:hover {
        background: rgba(16, 185, 129, 0.26) !important;
        border-color: #10b981 !important;
        color: #ffffff !important;
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.38) !important;
        transform: translateY(-1px) !important;
    }

    /* Distinct Rose Accent for ✗ Incorrect */
    button[data-testid*="feedback_wrong"],
    button[key*="feedback_wrong"] {
        background: rgba(244, 63, 94, 0.14) !important;
        border: 1px solid rgba(244, 63, 94, 0.4) !important;
        color: #fb7185 !important;
    }
    button[data-testid*="feedback_wrong"]:hover,
    button[key*="feedback_wrong"]:hover {
        background: rgba(244, 63, 94, 0.26) !important;
        border-color: #f43f5e !important;
        color: #ffffff !important;
        box-shadow: 0 0 14px rgba(244, 63, 94, 0.38) !important;
        transform: translateY(-1px) !important;
    }

    /* ------------------------------------------------------------- */
    /* CRITICAL FIX: Image Preview Max-Height Constraint             */
    /* Prevents vertical monopolization of the viewport              */
    /* ------------------------------------------------------------- */
    div[data-testid="stImage"] img {
        display: block;
        max-width: 100%;
        max-height: 440px !important;
        width: auto !important;
        margin: 0.35rem auto 0.75rem !important;
        object-fit: contain !important;
        border-radius: var(--sp-radius-md) !important;
        border: 1px solid var(--sp-glass-border) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5) !important;
    }

    .sp-preview-meta {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        padding: 0.55rem 1rem;
        margin: 0.5rem 0 0.85rem 0;
        border-radius: var(--sp-radius-md);
        background: rgba(13, 21, 39, 0.65);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        font-size: 0.84rem;
        color: var(--sp-text-secondary);
    }

    /* ------------------------------------------------------------- */
    /* CRITICAL FIX: Analyzing Status - ZERO Scrollbars              */
    /* ------------------------------------------------------------- */
    .sp-analysis-status {
        position: relative;
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.35rem;
        margin: 0.85rem 0 1.35rem 0;
        border-radius: var(--sp-radius-md);
        border: 1px solid rgba(56, 189, 248, 0.4);
        background: rgba(13, 21, 39, 0.88);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: #f8fafc;
        box-sizing: border-box;
        width: 100%;
        overflow: hidden !important;
        scrollbar-width: none !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5), 0 0 25px rgba(56, 189, 248, 0.2);
    }
    .sp-analysis-status::-webkit-scrollbar {
        display: none !important;
    }
    .sp-analysis-wave-top {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #06b6d4 0%, #38bdf8 45%, #818cf8 80%, #06b6d4 100%);
        background-size: 200% 100%;
        animation: sp-wave-travel 1.8s linear infinite;
    }
    .sp-analysis-spinner {
        width: 1.35rem;
        height: 1.35rem;
        border: 2px solid rgba(56, 189, 248, 0.25);
        border-top-color: #38bdf8;
        border-radius: 50%;
        animation: sp-spin 750ms linear infinite;
        flex-shrink: 0;
    }
    .sp-analysis-text {
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
        min-width: 0;
    }
    .sp-analysis-title {
        font-weight: 600;
        font-size: 0.98rem;
        color: #f8fafc;
    }
    .sp-analysis-sub {
        font-size: 0.84rem;
        color: #94a3b8;
    }

    @keyframes sp-wave-travel {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }
    @keyframes sp-spin {
        to { transform: rotate(360deg); }
    }
    @keyframes sp-dot-pulse {
        0%, 100% { opacity: 0.55; transform: scale(0.9); }
        50% { opacity: 1; transform: scale(1.3); }
    }

    /* Antigravity Hero Iframe & Container */
    iframe[title*="antigravity_hero"] {
        border: none !important;
        border-radius: var(--sp-radius-lg) !important;
        overflow: hidden !important;
        display: block !important;
        width: 100% !important;
        height: 400px !important;
        margin-bottom: 0.75rem !important;
    }
    div[data-testid="stCustomComponentV1"] {
        overflow: hidden !important;
        border-radius: var(--sp-radius-lg) !important;
    }

    /* ------------------------------------------------------------- */
    /* Header, Panels & Badges                                       */
    /* ------------------------------------------------------------- */
    .app-subtitle {
        font-size: 0.96rem;
        color: var(--sp-text-secondary);
        margin: -0.2rem 0 0.85rem 0;
        line-height: 1.5;
    }
    .app-header-rule {
        border-bottom: 1px solid var(--sp-glass-border);
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }

    .sp-upload-panel,
    .sp-controls-card {
        padding: 1.15rem 1.35rem;
        margin-bottom: 1.25rem;
        border-radius: var(--sp-radius-lg);
        background: var(--sp-bg-surface);
        backdrop-filter: blur(var(--sp-glass-blur));
        -webkit-backdrop-filter: blur(var(--sp-glass-blur));
        border: 1px solid var(--sp-glass-border);
        box-shadow: var(--sp-shadow-glass);
    }

    /* Badges */
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.24rem 0.65rem;
        border-radius: var(--sp-radius-full);
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    .badge-red {
        background-color: var(--sp-danger-bg);
        color: #fda4af;
        border: 1px solid var(--sp-danger-border);
        box-shadow: 0 0 12px rgba(244, 63, 94, 0.2);
    }
    .badge-green {
        background-color: var(--sp-success-bg);
        color: #6ee7b7;
        border: 1px solid var(--sp-success-border);
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
    }
    .badge-amber {
        background-color: var(--sp-warning-bg);
        color: #fcd34d;
        border: 1px solid var(--sp-warning-border);
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.2);
    }
    .badge-gray {
        background-color: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.25);
    }

    /* Vehicle Details Styling */
    .sp-vehicle-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.85rem;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid var(--sp-glass-border);
    }
    .sp-vh-title {
        display: flex;
        align-items: baseline;
        gap: 0.65rem;
    }
    .sp-vh-id {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
    }
    .sp-vh-class {
        font-size: 0.9rem;
        font-weight: 500;
        color: var(--sp-accent-sky);
        padding: 0.15rem 0.5rem;
        border-radius: var(--sp-radius-sm);
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.25);
    }
    .sp-vh-badges {
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    /* Monospace License Plate Pill */
    .sp-plate-pill {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 0.22rem 0.65rem;
        border-radius: var(--sp-radius-sm);
        margin-right: 0.35rem;
    }
    .sp-plate-pill.read {
        background: rgba(16, 185, 129, 0.16);
        border: 1px solid rgba(16, 185, 129, 0.45);
        color: #6ee7b7;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
    }
    .sp-plate-pill.uncertain {
        background: rgba(245, 158, 11, 0.16);
        border: 1px solid rgba(245, 158, 11, 0.45);
        color: #fcd34d;
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.2);
    }
    .sp-plate-pill.unreadable {
        background: rgba(148, 163, 184, 0.14);
        border: 1px solid rgba(148, 163, 184, 0.3);
        color: #cbd5e1;
    }
    .sp-plate-sub {
        font-size: 0.8rem;
        color: var(--sp-text-secondary);
    }

    /* Action Container */
    .action-panel {
        background-color: rgba(13, 21, 39, 0.7);
        backdrop-filter: blur(14px);
        border: 1px solid var(--sp-glass-border);
        border-radius: var(--sp-radius-md);
        padding: 1.1rem 1.25rem;
        margin-top: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }

    /* Login Card */
    .sp-login-card {
        text-align: center;
        padding: 2.25rem 2rem 1.25rem 2rem;
        margin: 2.5rem auto 1.5rem auto;
        border-radius: var(--sp-radius-xl);
        background: var(--sp-bg-surface);
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid rgba(56, 189, 248, 0.35);
        box-shadow: var(--sp-shadow-glass), 0 0 35px rgba(56, 189, 248, 0.15);
    }
    .sp-login-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .sp-login-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0 0 0.35rem 0;
        letter-spacing: -0.025em;
    }
    .sp-login-sub {
        font-size: 0.92rem;
        color: var(--sp-text-secondary);
        margin: 0 0 1rem 0;
    }
    .sp-login-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.24rem 0.75rem;
        border-radius: var(--sp-radius-full);
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: var(--sp-accent-sky);
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1.25rem;
    }

    /* Footer */
    .app-footer {
        border-top: 1px solid var(--sp-glass-border);
        margin-top: 3.5rem;
        padding-top: 1.25rem;
        text-align: center;
        font-size: 0.84rem;
        color: var(--sp-text-muted);
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
        }
        div[data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Security & Access Control
# ---------------------------------------------------------
if not config.APP_PASSWORD:
    st.error("App access is disabled. Configure APP_PASSWORD in the local .env file and restart the app.")
    st.stop()
if len(config.APP_PASSWORD) < 8:
    st.error("APP_PASSWORD must contain at least 8 characters. Update the local .env file and restart the app.")
    st.stop()

now = time.monotonic()
if st.session_state.get("authenticated", False):
    last_activity = st.session_state.get("auth_last_activity", now)
    if now - last_activity > config.AUTH_IDLE_TIMEOUT_MINUTES * 60:
        st.session_state.clear()

if not st.session_state.get("authenticated", False):
    _, col_login, _ = st.columns([1, 1.4, 1])
    with col_login:
        st.markdown("""
            <div class="sp-login-card">
                <div class="sp-login-icon">🅿️</div>
                <h2 class="sp-login-title">Smart Parking AI</h2>
                <p class="sp-login-sub">Deep Learning Vision & Campus Enforcement</p>
                <div class="sp-login-badge">
                    <span class="sp-chip-dot"></span>
                    <span>RESTRICTED ACCESS PORTAL</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        with st.form("app_login"):
            entered_password = st.text_input("Security Passcode", type="password", placeholder="Enter authorization key")
            login_clicked = st.form_submit_button("Authorize & Launch", type="primary", width="stretch")
        if login_clicked:
            if hmac.compare_digest(entered_password, config.APP_PASSWORD):
                st.session_state["authenticated"] = True
                st.session_state["auth_last_activity"] = time.monotonic()
                st.rerun()
            st.error("Access denied: Invalid credentials.")
    st.stop()

# Initialize Database on app load
init_db()

# Refresh the idle timer for authenticated sessions
st.session_state["auth_last_activity"] = time.monotonic()

# ---------------------------------------------------------
# Cached Model Loaders
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO vehicle detection model...")
def load_cached_vehicle_model():
    return get_vehicle_model()

@st.cache_resource(show_spinner="Loading overhead vehicle detection model...")
def load_cached_overhead_model():
    return get_overhead_vehicle_model()

@st.cache_resource(show_spinner="Loading license plate detection model...")
def load_cached_plate_model():
    try:
        return get_plate_detector()
    except FileNotFoundError:
        return None

@st.cache_resource(show_spinner="Loading EasyOCR character recognition engine...")
def load_cached_ocr_reader():
    return get_ocr_reader()

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None
if "notification_feedback" not in st.session_state:
    st.session_state["notification_feedback"] = {}

# ---------------------------------------------------------
# Sidebar: Registry Management & Audit History
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sp-sidebar-header">
            <div class="sp-brand-badge">
                <span class="sp-brand-icon">🅿️</span>
                <div>
                    <div class="sp-brand-title">Smart Parking AI</div>
                    <div class="sp-brand-subtitle">Campus CV Lab & Enforcement</div>
                </div>
            </div>
            <div class="sp-status-chip">
                <span class="sp-chip-dot"></span>
                <span>System Online</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🚗 Registered Vehicles")
    st.caption("Persistent SQLite vehicle database & email dispatch authorization.")

    vehicles = get_all_vehicles()
    if vehicles:
        table_rows = []
        for v in vehicles:
            table_rows.append({
                "Plate": v["plate_number"],
                "Role": (v.get("role") or "general").title(),
                "Consent": "Yes" if v.get("consent_status", 1) == 1 else "No",
                "Email": v["email"],
            })
        st.dataframe(
            table_rows,
            column_config={
                "Plate": st.column_config.TextColumn("Plate", width=75),
                "Role": st.column_config.TextColumn("Role", width=70),
                "Consent": st.column_config.TextColumn("Consent", width=65),
                "Email": st.column_config.TextColumn("Email", width=120),
            },
            width="stretch",
            hide_index=True
        )
    else:
        st.caption("No registered vehicles.")

    with st.expander("➕ Add / Manage Vehicle", expanded=False):
        tab_add, tab_del = st.tabs(["Add", "Remove"])
        with tab_add:
            with st.form("sidebar_add_vehicle", clear_on_submit=True):
                new_plate = st.text_input("Number Plate", placeholder="e.g. KA01AB1234")
                new_role = st.selectbox("Campus Role", ["student", "faculty", "staff", "principal", "visitor", "vendor", "general"])
                new_email = st.text_input("Registered Email", placeholder="owner@example.com")
                new_consent = st.checkbox("Notification Consent", value=True)
                submit_add = st.form_submit_button("Save Vehicle", type="primary", width="stretch")

                if submit_add:
                    success, msg = register_vehicle(
                        plate_number=new_plate,
                        email=new_email,
                        role=new_role,
                        consent_status=1 if new_consent else 0
                    )
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_del:
            if vehicles:
                del_opts = {f"{v['plate_number']} ({v['email']})": v["id"] for v in vehicles}
                selected_del = st.selectbox("Select Vehicle", list(del_opts.keys()))
                if st.button("Delete Registration", width="stretch"):
                    if remove_vehicle(del_opts[selected_del]):
                        st.success("Vehicle removed.")
                        st.rerun()
                    else:
                        st.error("Could not remove vehicle.")

    with st.expander("📜 Notification Audit Log", expanded=False):
        logs = get_recent_audit_logs(limit=10)
        if logs:
            st.dataframe(
                [
                    {
                        "Plate": l["plate_number"],
                        "Status": l["status"],
                        "Time": l["timestamp"][:19]
                    }
                    for l in logs
                ],
                width="stretch",
                hide_index=True
            )
        else:
            st.caption("No notifications sent yet.")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("🚪 Sign out", key="btn_sign_out", width="stretch"):
        st.session_state.clear()
        st.rerun()

# ---------------------------------------------------------
# 1. Main Header: Official React Bits Antigravity Particle Background
# ---------------------------------------------------------
antigravity_hero(
    title="Smart Parking Violation Detection",
    subtitle="AI-powered parking scene analysis • Autonomous CV System",
    badge="Autonomous CV Lab",
    key="sp-antigravity-hero"
)
st.markdown('<div class="app-header-rule"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Image Upload Section & Test Scenes
# ---------------------------------------------------------
sample_options = {
    "Upload custom image...": None,
    "Overhead Parking Lot (~15 cars)": "data/test_images/sample_parking_1.jpg",
    "Category 1: Red curb / Fire Lane potential violation": "data/test_images/scene_clearly_marked_restricted_area.png",
    "Category 1: Vehicle near No-Parking sign (Travel lane, compliant)": "data/test_images/scene_near_no_parking_sign.png",
    "Category 2: Motorcycle parked on pedestrian footpath (Violation)": "data/test_images/scene_cat2_footpath_parking.png",
    "Category 2: Red pickup truck mounted on roadside footpath (Real Scene)": "data/test_images/red_pickup_footpath.jpg",
    "Category 2: Car mounted on footpath / sidewalk (Real Scene)": "data/test_images/car_parked_footpath_real.jpg",
    "Category 2: Vehicle beside footpath on road (Compliant)": "data/test_images/scene_cat2_footpath_compliant.png",
    "Category 3: Vehicle blocking building entrance gate (Violation)": "data/test_images/scene_cat3_blocking_entrance.png",
    "Category 3: Vehicle beside entrance clear corridor (Compliant)": "data/test_images/scene_cat3_entrance_clear.png",
    "Category 4: Car parked on campus lawn / turf (Violation)": "data/test_images/scene_cat4_lawn_parking.png",
    "Category 4: White car parked on landscaped green space (Real Scene)": "data/test_images/white_car_grass_parking.jpg",
    "Category 4: Car in paved parking bay beside lawn (Compliant)": "data/test_images/scene_cat4_paved_parking.png",
    "Category 5: Vehicle straddling divider lines (Improper Alignment)": "data/test_images/scene_cat5_improper_alignment.png",
    "Category 5: Vehicle centered within stall lines (Proper Alignment)": "data/test_images/scene_cat5_proper_alignment.png",
    "Category 6: Student car in Principal/Faculty slot (Wrong Slot)": "data/test_images/scene_cat6_wrong_slot_unauthorized.png",
    "Category 6: Principal car in Principal slot (Authorized)": "data/test_images/scene_cat6_correct_slot_authorized.png",
    "Standard: Compliant stall parking": "data/test_images/scene_ordinary_parking_area.png",
    "Plate Test: Car with legible plate (P3RVP)": "data/test_images/car_sample_2.png",
    "Plate Test: Blurry / unreadable number plate": "data/test_images/scene_unreadable_blurry_plate.png",
    "Empty scene: No vehicles": "data/test_images/scene_no_vehicles.png"
}

st.markdown('<div class="sp-upload-panel">', unsafe_allow_html=True)
col_upload, col_sample = st.columns([3, 2])
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help=f"Upload JPG or PNG up to {config.MAX_IMAGE_UPLOAD_MB} MB and {config.MAX_IMAGE_PIXELS:,} pixels."
    )
with col_sample:
    selected_sample = st.selectbox(
        "Or select a test scene",
        list(sample_options.keys()),
        index=0,
        label_visibility="collapsed"
    )
st.markdown('</div>', unsafe_allow_html=True)

active_image: Optional[Image.Image] = None
active_image_name: Optional[str] = None

if uploaded_file is not None:
    try:
        if uploaded_file.size > config.MAX_IMAGE_UPLOAD_MB * 1024 * 1024:
            raise ValueError(f"Image exceeds the {config.MAX_IMAGE_UPLOAD_MB} MB upload limit.")
        opened_image = Image.open(uploaded_file)
        if opened_image.width * opened_image.height > config.MAX_IMAGE_PIXELS:
            raise ValueError(f"Image exceeds the {config.MAX_IMAGE_PIXELS:,}-pixel limit.")
        active_image = opened_image.convert("RGB")
        active_image_name = uploaded_file.name
    except Exception as e:
        active_image = None
        st.error(f"Unable to read image: {e}")
elif selected_sample != "Upload custom image..." and sample_options[selected_sample] is not None:
    sample_path = config.BASE_DIR / sample_options[selected_sample]
    if sample_path.exists():
        active_image = Image.open(sample_path).convert("RGB")
        active_image_name = sample_path.name

# Show the selected source frame with metadata & constrained preview
if active_image is not None:
    st.markdown(f"""
        <div class="sp-preview-meta">
            <span>📷 <b>Source Frame:</b> {active_image_name}</span>
            <span>📐 {active_image.width} × {active_image.height} px</span>
            <span>⚡ Ready for Analysis</span>
        </div>
    """, unsafe_allow_html=True)
    st.image(
        active_image,
        caption=f"Image preview: {active_image_name} ({active_image.width} × {active_image.height})",
        width="stretch",
    )

# Detection Settings (Camera Perspective & Confidence Threshold)
st.markdown('<div class="sp-controls-card">', unsafe_allow_html=True)
col_mode, col_conf = st.columns([3, 2])
with col_mode:
    model_mode = st.selectbox(
        "Camera Perspective / Model",
        [
            "Auto (Adaptive Ground & Overhead)",
            "Overhead / Aerial (VisDrone YOLOv8n)",
            "Ground-Level (COCO YOLOv8n)"
        ],
        index=0,
        help="Select 'Overhead / Aerial' for top-down parking views, 'Ground-Level' for street cameras, or 'Auto' for automatic detection."
    )
with col_conf:
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.90,
        value=float(config.DEFAULT_VEHICLE_CONF),
        step=0.05,
        help="Minimum confidence threshold required to localize vehicles."
    )
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# 3. Primary Action Button: Analyze Image
# ---------------------------------------------------------
if active_image is not None:
    analyze_btn = st.button("🚀 Analyze Parking Scene", type="primary", width="stretch")

    if analyze_btn:
        loading_slot = st.empty()
        with loading_slot:
            st.markdown("""
                <div class="sp-analysis-status" role="status" aria-live="polite">
                    <div class="sp-analysis-wave-top"></div>
                    <span class="sp-analysis-spinner" aria-hidden="true"></span>
                    <div class="sp-analysis-text">
                        <span class="sp-analysis-title">Analyzing parking scene across 6 campus priority categories...</span>
                        <span class="sp-analysis-sub">Evaluating YOLOv8 vehicle detections, pavement markings & ALPR plates</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        try:
            p_model = load_cached_plate_model()
            ocr_reader = load_cached_ocr_reader()

            if model_mode == "Ground-Level (COCO YOLOv8n)":
                v_model = load_cached_vehicle_model()
                allow_fallback = False
            elif model_mode == "Overhead / Aerial (VisDrone YOLOv8n)":
                v_model = load_cached_overhead_model()
                allow_fallback = False
            else:
                v_model = None
                allow_fallback = True

            # Automatic detection and scene understanding across all 6 campus priority categories
            output = detect_vehicles_and_violations(
                image_input=active_image,
                model=v_model,
                conf_threshold=conf_threshold,
                zones=None,
                ocr_reader=ocr_reader,
                plate_model=p_model,
                allow_overhead_fallback=allow_fallback
            )

            # Ensure all vehicles have complete plate and registration metadata
            processed_vehicles = []
            for v in output["vehicles"]:
                v_record = dict(v)
                if "normalized_plate" not in v_record or not v_record.get("plate_detected"):
                    plate_data = read_plate_from_vehicle_crop(
                        vehicle_crop_rgb=v["vehicle_crop"],
                        plate_model=p_model,
                        ocr_reader=ocr_reader,
                        conf_threshold=config.DEFAULT_PLATE_CONF
                    )
                    v_record.update(plate_data)
                processed_vehicles.append(v_record)

            output["vehicles"] = processed_vehicles
            output["image_name"] = active_image_name
            output["image_hash"] = hashlib.sha256(active_image.tobytes()).hexdigest()
            # Keep the unannotated source frame with this result so feedback
            # always saves the exact image that produced the shown prediction.
            output["feedback_image_rgb"] = np.asarray(active_image, dtype=np.uint8).copy()
            output["analysis_id"] = uuid.uuid4().hex
            st.session_state["analysis_results"] = output
            st.session_state["notification_feedback"] = {}
        finally:
            loading_slot.empty()

# ---------------------------------------------------------
# 4 & 5. Annotated Image & Concise List of Potential Violations
# ---------------------------------------------------------
results = st.session_state.get("analysis_results")

if results is not None:
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    total_v = results["total_vehicles"]
    total_viol = results["total_violations"]
    total_reviews = results.get("total_reviews", sum(
        1 for vehicle in results.get("vehicles", [])
        if not vehicle.get("is_violation") and vehicle.get("violation_status") == "Review required"
    ))
    zone_status = results.get("zone_status", "")
    feedback_records = get_prediction_feedback(results.get("analysis_id", ""))

    # The Fade Content-inspired banner reveals the completed Python analysis summary.
    if total_v == 0:
        result_tone = "neutral"
        result_message = "No vehicles detected in the uploaded image."
    elif total_viol > 0:
        result_tone = "danger"
        result_message = (
            f"{total_viol} potential parking violation(s) detected — review required. "
            f"{f'{total_reviews} additional vehicle(s) need review; no violation is confirmed for those vehicles.' if total_reviews else ''}"
        )
    elif total_reviews > 0:
        result_tone = "warning"
        result_message = (
            f"{total_reviews} vehicle(s) need review — possible entrance, grass, or restricted-area obstruction; "
            "no violation has been confirmed."
        )
    elif zone_status == "Restricted zone could not be determined reliably":
        result_tone = "warning"
        result_message = "Restricted zone could not be determined reliably — no violations flagged."
    else:
        result_tone = "success"
        result_message = f"No violations detected — all {total_v} vehicle(s) are compliant with visible markings."

    animated_result_banner(
        result_tone,
        result_message,
        key=f"analysis-result-{results.get('analysis_id', 'latest')}",
    )

    # 4. Annotated image with detection results
    st.image(
        results["annotated_image"],
        caption=f"Analysis Result ({total_v} vehicle(s) • {total_viol} potential violation(s) • {total_reviews} review required)",
        width="stretch"
    )

    # 5. Concise list of potential violations & notification status
    st.markdown("### Results & Review")

    if total_v == 0:
        diag = results.get("diagnostics", {})
        used_model_name = Path(diag.get("model_name", "YOLO")).name
        raw_count = diag.get("raw_detection_count", 0)
        st.info(
            f"ℹ️ **No vehicles were localized in this image with the current detection settings.**\n\n"
            f"This does not necessarily mean the scene contains no vehicles. Troubleshooting guidance:\n\n"
            f"• **Camera Perspective:** If analyzing an overhead, aerial, or drone top-down view, ensure **'Auto (Adaptive)'** or **'Overhead / Aerial (VisDrone)'** is selected. Standard COCO models frequently fail to recognize cars from orthogonal top-down angles.\n"
            f"• **Confidence Threshold:** Try lowering the detection confidence threshold (current: `{diag.get('conf_threshold', conf_threshold)}`) in the settings above.\n"
            f"• **Diagnostics:** Expand the **Detection Diagnostics** below to review raw candidate detections ({raw_count} raw candidates were localized before vehicle class filtering)."
        )
    else:
        vehicles_list = results["vehicles"]

        for idx, v in enumerate(vehicles_list):
            img_hash = results.get("image_hash", "IMG")
            analysis_id = results.get("analysis_id", img_hash)
            case_id = f"case_{analysis_id}_{v['vehicle_id']}"
            is_violation = v["is_violation"]
            v_status = v.get("violation_status", "No violation detected")

            # Determine plate display
            norm_plate = v.get("normalized_plate", "").strip()
            ocr_reading_status = v.get("reading_status", "")
            manual_plate = st.session_state.get(f"verified_plate_{case_id}")

            if manual_plate:
                active_plate = manual_plate
                is_plate_readable = True
            elif ocr_reading_status == "Recognized" and len(norm_plate) >= 2:
                active_plate = norm_plate
                is_plate_readable = True
            else:
                active_plate = norm_plate
                is_plate_readable = False

            # Check database for registration
            reg_rec = lookup_plate(active_plate) if (is_plate_readable and active_plate) else None
            is_registered = reg_rec is not None
            reg_email = reg_rec["email"] if reg_rec else None
            consent_status = int(reg_rec.get("consent_status", 1)) if reg_rec else 0

            # Idempotency event check
            event_id = f"EVT-{analysis_id}-{v['vehicle_id']}-{active_plate if active_plate else 'NOPLATE'}"
            already_notified = is_duplicate_event(event_id)

            # Card badge style
            if is_violation:
                status_badge = '<span class="badge badge-red">Potential violation — review required</span>'
            elif v_status in ("Restricted zone could not be determined reliably", "Review required"):
                badge_label = "Review required" if v_status == "Review required" else "Zone undetermined"
                status_badge = f'<span class="badge badge-amber">{badge_label}</span>'
            else:
                status_badge = '<span class="badge badge-green">No violation detected</span>'

            with st.container(border=True):
                notice_badge = '<span class="badge badge-green" style="margin-left:0.4rem;">Notice Sent</span>' if already_notified else ''
                st.markdown(
                    f'<div class="sp-vehicle-header">'
                    f'<div class="sp-vh-title">'
                    f'<span class="sp-vh-id">Vehicle #{v["vehicle_id"]}</span>'
                    f'<span class="sp-vh-class">{v["class_name"].title()}</span>'
                    f'</div>'
                    f'<div class="sp-vh-badges">{status_badge}{notice_badge}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                col_thumb, col_details = st.columns([1, 2], gap="medium")

                with col_thumb:
                    st.image(v["vehicle_crop"], caption=f"Vehicle #{v['vehicle_id']}", width="stretch")
                    if v.get("plate_crop") is not None:
                        st.image(v["plate_crop"], caption="Plate Crop", width="stretch")

                with col_details:
                    # Plate display
                    if is_plate_readable:
                        st.markdown(f"**License Plate:** <span class='sp-plate-pill read'>{active_plate}</span> &nbsp; <span class='sp-plate-sub'>Verified OCR</span>", unsafe_allow_html=True)
                    elif active_plate:
                        st.markdown(f"**License Plate:** <span class='sp-plate-pill uncertain'>{active_plate}</span> &nbsp; <span class='sp-plate-sub'>Uncertain OCR — Manual Verification Recommended</span>", unsafe_allow_html=True)
                    else:
                        st.markdown("**License Plate:** <span class='sp-plate-pill unreadable'>Unreadable / No Plate Localized</span>", unsafe_allow_html=True)

                    st.caption(f"Vehicle detection confidence: {float(v.get('confidence', 0.0)):.0%}")

                    with st.expander("✏️ Correct / Verify Number Plate", expanded=False):
                        with st.form(f"manual_plate_form_{case_id}"):
                            entered_p = st.text_input("Enter Plate Number", value=active_plate if active_plate else "", key=f"input_manual_plate_{case_id}")
                            submit_p = st.form_submit_button("Update Plate")
                            if submit_p:
                                norm_p = normalize_plate(entered_p)
                                if len(norm_p) >= 2:
                                    st.session_state[f"verified_plate_{case_id}"] = norm_p
                                    st.success(f"Plate updated to `{norm_p}`.")
                                    st.rerun()
                                else:
                                    st.error("Please enter a valid license plate.")

                    # Location / Zone info
                    if is_violation:
                        st.markdown(f"**Restricted Zone:** {v['zone_name']}")
                    elif v_status == "Restricted zone could not be determined reliably":
                        st.markdown(f"**Scene Assessment:** `Restricted zone could not be determined reliably`  \n{v.get('review_reason', '')}")
                    elif v_status == "Review required":
                        st.markdown(f"**Parking Status:** `Review required` — {v.get('zone_name') or 'Campus rule'}  \n{v.get('review_reason', '')}")
                    else:
                        st.markdown("**Parking Status:** Compliant with visible roadway markings")

                    # Registration info
                    if is_registered:
                        consent_tag = "Enabled" if consent_status == 1 else "Disabled"
                        owner_role = (reg_rec.get("role") or "General").title()
                        st.markdown(f"**Registered Owner:** `{reg_email}` &nbsp; *(Role: **{owner_role}** • Consent: {consent_tag})*")
                    elif is_violation:
                        st.markdown("**Registration Status:** `Unregistered vehicle` (No permit on record)")

                    # 6 Priority Campus Categories Breakdown
                    campus_cats = v.get("campus_evaluations", {}).get("categories", {})
                    if campus_cats:
                        with st.expander("📊 Six Campus Priority Violation Categories Assessment", expanded=is_violation or v_status == "Review required"):
                            st.caption("Was each category prediction correct? Feedback is saved locally with the source image for later review; it does not retrain the model immediately.")
                            feedback_options = {
                                "No parking issue / none": "none",
                                "Unauthorized / restricted zone": "cat1_unauthorized_zone",
                                "Footpath parking": "cat2_footpath_parking",
                                "Blocking entrance / emergency access": "cat3_blocking_entrance",
                                "Parking on grass / green space": "cat4_lawn_parking",
                                "Improper parking alignment": "cat5_improper_alignment",
                                "Wrong / reserved parking slot": "cat6_wrong_parking_slot",
                                "Unclear / other": "other",
                            }
                            feedback_image = results.get("feedback_image_rgb")
                            model_name = results.get("diagnostics", {}).get("model_name", "unknown")
                            for cat_key, cat_val in campus_cats.items():
                                c_status = cat_val.get("status", "Not detected")
                                if c_status == "Potential violation detected":
                                    badge_html = '<span class="badge badge-red">Potential violation detected</span>'
                                elif c_status == "Review required":
                                    badge_html = '<span class="badge badge-amber">Review required</span>'
                                else:
                                    badge_html = '<span class="badge badge-green">Not detected</span>'

                                feedback_id = f"{analysis_id}_{v['vehicle_id']}_{cat_key}"
                                show_correction_key = f"show_feedback_correction_{feedback_id}"
                                label_key = f"feedback_correct_label_{feedback_id}"
                                saved_feedback = feedback_records.get((int(v["vehicle_id"]), cat_key))

                                # Responsive layout ensuring buttons never truncate
                                info_col, correct_col, wrong_col = st.columns([3.2, 1.4, 1.4], gap="small")
                                with info_col:
                                    st.markdown(
                                        f'<div class="cat-badge-container">'
                                        f'<span class="cat-title">{cat_val["category_name"]}</span>'
                                        f'{badge_html}'
                                        f'</div>',
                                        unsafe_allow_html=True
                                    )
                                with correct_col:
                                    correct_clicked = st.button("✓ Correct", key=f"feedback_correct_{feedback_id}", width="stretch", help="Confirm this category result")
                                with wrong_col:
                                    wrong_clicked = st.button("✗ Incorrect", key=f"feedback_wrong_{feedback_id}", width="stretch", help="Mark this category result wrong and provide the correct category")

                                st.caption(f'{cat_val["explanation"]}  _{cat_val.get("evidence", "")}_')

                                if saved_feedback:
                                    saved_label = saved_feedback.get("corrected_label", "")
                                    st.caption(f'Saved feedback: {saved_feedback["feedback_vote"].title()} · label: {saved_label}')

                                if correct_clicked:
                                    try:
                                        save_prediction_feedback(
                                            analysis_id=analysis_id,
                                            image_rgb=feedback_image,
                                            vehicle_id=v["vehicle_id"],
                                            category_key=cat_key,
                                            predicted_status=c_status,
                                            feedback_vote="correct",
                                            corrected_label=c_status if c_status != "Not detected" else "none",
                                            bbox=list(v["bbox"]),
                                            context={
                                                "category_name": cat_val["category_name"],
                                                "explanation": cat_val.get("explanation", ""),
                                                "evidence": cat_val.get("evidence", ""),
                                                "vehicle_class": v.get("class_name", ""),
                                                "vehicle_confidence": v.get("confidence"),
                                            },
                                            model_name=model_name,
                                        )
                                        st.success("Feedback recorded locally for future model evaluation.")
                                    except Exception as exc:
                                        st.error(f"Could not save feedback: {exc}")

                                if wrong_clicked:
                                    st.session_state[show_correction_key] = True

                                if st.session_state.get(show_correction_key, False):
                                    selected_feedback_label = st.selectbox(
                                        "Choose the correct result",
                                        list(feedback_options.keys()),
                                        key=label_key,
                                    )
                                    if st.button("Save correction", key=f"save_feedback_{feedback_id}", type="primary"):
                                        try:
                                            save_prediction_feedback(
                                                analysis_id=analysis_id,
                                                image_rgb=feedback_image,
                                                vehicle_id=v["vehicle_id"],
                                                category_key=cat_key,
                                                predicted_status=c_status,
                                                feedback_vote="wrong",
                                                corrected_label=feedback_options[selected_feedback_label],
                                                bbox=list(v["bbox"]),
                                                context={
                                                    "category_name": cat_val["category_name"],
                                                    "explanation": cat_val.get("explanation", ""),
                                                    "evidence": cat_val.get("evidence", ""),
                                                    "vehicle_class": v.get("class_name", ""),
                                                    "vehicle_confidence": v.get("confidence"),
                                                },
                                                model_name=model_name,
                                            )
                                            st.session_state[show_correction_key] = False
                                            st.success("Correction saved locally for later model review.")
                                        except Exception as exc:
                                            st.error(f"Could not save correction: {exc}")

                    # Action 1: Verified Consenting Violation -> Send Email Notice
                    if is_violation and is_registered and consent_status == 1:
                        st.markdown("""<div class="action-panel">""", unsafe_allow_html=True)
                        st.markdown("**Review & Advisory Notice Dispatch**")

                        if already_notified:
                            st.info("✅ Notice has already been sent for this verified event.")
                        else:
                            confirm_check = st.checkbox(
                                "I confirm that I have reviewed the vehicle, restricted area, and recipient email.",
                                key=f"confirm_{case_id}"
                            )
                            send_btn = st.button(
                                "Send Email Notice",
                                key=f"send_btn_{case_id}",
                                disabled=not confirm_check,
                                type="primary"
                            )

                            if send_btn:
                                with st.spinner("Dispatching notice..."):
                                    res = send_violation_notice(
                                        event_id=event_id,
                                        plate_number=active_plate,
                                        recipient_email=reg_email,
                                        zone_name=v["zone_name"]
                                    )
                                    st.session_state["notification_feedback"][case_id] = res
                                    if res["success"]:
                                        st.success(res["message"])
                                        st.rerun()
                                    else:
                                        st.error(res["message"])

                        st.markdown("</div>", unsafe_allow_html=True)

                    # Action 2: Unknown Vehicle with Potential Violation -> Optional Inline Registration
                    elif is_violation and not is_registered:
                        st.markdown("""<div class="action-panel">""", unsafe_allow_html=True)
                        st.markdown("**Vehicle Not Registered**")
                        st.caption("Optional: Register owner's email and role with explicit consent to receive notices.")

                        show_reg = st.session_state.get(f"show_reg_{case_id}", False)
                        if not show_reg:
                            if st.button("Register Vehicle", key=f"btn_open_reg_{case_id}"):
                                st.session_state[f"show_reg_{case_id}"] = True
                                st.rerun()
                        else:
                            with st.form(f"reg_form_{case_id}"):
                                reg_in_plate = st.text_input("Plate Number", value=active_plate if active_plate else "")
                                reg_in_role = st.selectbox(
                                    "Campus Role",
                                    ["student", "faculty", "staff", "principal", "visitor", "vendor", "general"],
                                    key=f"role_{case_id}"
                                )
                                reg_in_email = st.text_input("Owner Email Address", placeholder="user@example.com")
                                reg_in_consent = st.checkbox(
                                    "I confirm that the owner has agreed to receive parking notifications at this email.",
                                    value=False
                                )
                                col_s, col_c = st.columns([1, 1])
                                with col_s:
                                    submit_save = st.form_submit_button("Save Registration", type="primary", width="stretch")
                                with col_c:
                                    submit_cancel = st.form_submit_button("Cancel", width="stretch")

                                if submit_cancel:
                                    st.session_state[f"show_reg_{case_id}"] = False
                                    st.rerun()

                                if submit_save:
                                    norm_save = normalize_plate(reg_in_plate)
                                    if len(norm_save) < 2:
                                        st.error("Please enter a valid license plate.")
                                    elif not is_valid_email(reg_in_email):
                                        st.error("Please enter a valid email address.")
                                    elif not reg_in_consent:
                                        st.error("Consent confirmation is required to enable notifications.")
                                    else:
                                        ok, msg = register_vehicle(
                                            plate_number=norm_save,
                                            email=reg_in_email.strip(),
                                            role=reg_in_role,
                                            consent_status=1,
                                            require_consent=True
                                        )
                                        if ok:
                                            st.session_state[f"verified_plate_{case_id}"] = norm_save
                                            st.session_state[f"show_reg_{case_id}"] = False
                                            st.success(f"Vehicle '{norm_save}' registered successfully.")
                                            st.rerun()
                                        else:
                                            st.error(msg)

    # ---------------------------------------------------------
    # Detection Diagnostics
    # ---------------------------------------------------------
    with st.expander("🔍 Detection Diagnostics", expanded=False):
        diag = results.get("diagnostics", {})
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"**Model Name:** `{Path(diag.get('model_name', 'yolov8n.pt')).name}`")
            st.markdown(f"**Confidence Threshold:** `{diag.get('conf_threshold', config.DEFAULT_VEHICLE_CONF)}`")
            w, h = diag.get("image_size", (0, 0))
            st.markdown(f"**Image Dimensions:** `{w} × {h}` px")
        with col_d2:
            st.markdown(f"**Raw Detection Count:** `{diag.get('raw_detection_count', 0)}` candidate(s)")
            st.markdown(f"**Vehicle Detections (Class Filtered):** `{diag.get('vehicle_detection_count', total_v)}` vehicle(s)")
            cls_labels = diag.get("class_labels", {})
            st.markdown(f"**Target Vehicle Classes:** `{list(cls_labels.values())}`")

        raw_breakdown = diag.get("raw_class_breakdown", {})
        if raw_breakdown:
            st.caption(f"Raw detection breakdown before filtering: {', '.join([f'{k}: {cnt}' for k, cnt in raw_breakdown.items()])}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<div class="app-footer">
    <b>Smart Parking AI</b> • Autonomous Deep Learning Computer Vision System • Built with YOLOv8 & SegFormer
</div>
""", unsafe_allow_html=True)
