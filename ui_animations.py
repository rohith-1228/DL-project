"""React Bits motion components for Streamlit.

Implements selected React Bits (https://reactbits.dev/) animation patterns:
- SplitText: Staggered letter/word reveal with blur and spring-like easing.
- ShinyText: Subtle continuous metallic light-sweep sheen on badges, accents, and buttons.
- FadeContent: Smooth staggered entrance for prediction banners and cards.
- Loading Wave: Lightweight, multi-stage animated preloader for inference pipeline.
- SpotlightCard & Ambient Glass: Restrained glassmorphic depth with glow accents.

All heavy model inference and database operations remain strictly in Python.
Full prefers-reduced-motion accessibility support is included.
"""

import streamlit as st

# -----------------------------------------------------------------------------
# 1. Page Heading: React Bits SplitText + ShinyText Badge
# -----------------------------------------------------------------------------
_HEADING_CSS = """
html, body {
    margin: 0;
    padding: 0;
    overflow: hidden !important;
    background: transparent !important;
}
.sp-heading-container {
    width: 100%;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 0.85rem;
    padding: 0.25rem 0 0.5rem 0;
    overflow: visible;
    box-sizing: border-box;
}
.sp-heading-a11y {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
}
.sp-heading-title-group {
    display: flex;
    align-items: baseline;
    gap: 0.85rem;
    flex-wrap: wrap;
}
.sp-heading-visual {
    margin: 0;
    color: #f8fafc;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: clamp(1.85rem, 3.2vw, 2.35rem);
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.03em;
    display: inline-flex;
    flex-wrap: wrap;
}
.sp-split-char {
    display: inline-block;
    opacity: 0;
    filter: blur(6px);
    transform: translateY(10px);
    animation: sp-split-enter 450ms cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
.sp-split-space {
    display: inline-block;
    width: 0.35em;
}

/* React Bits ShinyText Badge */
.sp-shiny-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.32rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #38bdf8;
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.35);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    background-image: linear-gradient(
        115deg,
        transparent 20%,
        rgba(56, 189, 248, 0.35) 48%,
        transparent 70%
    );
    background-size: 200% 100%;
    animation: sp-shiny-sweep 3.2s infinite linear;
    user-select: none;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.15);
}
.sp-shiny-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background-color: #38bdf8;
    box-shadow: 0 0 10px #38bdf8;
    animation: sp-dot-pulse 2s infinite ease-in-out;
}

@keyframes sp-split-enter {
    to {
        opacity: 1;
        filter: blur(0);
        transform: translateY(0);
    }
}
@keyframes sp-shiny-sweep {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
@keyframes sp-dot-pulse {
    0%, 100% { opacity: 0.55; transform: scale(0.9); }
    50% { opacity: 1; transform: scale(1.3); }
}

@media (prefers-reduced-motion: reduce) {
    .sp-split-char {
        animation: none !important;
        opacity: 1 !important;
        filter: none !important;
        transform: none !important;
    }
    .sp-shiny-badge, .sp-shiny-dot {
        animation: none !important;
    }
}
"""

_HEADING_JS = """
export default function({ parentElement, data }) {
    const root = parentElement.querySelector('.sp-heading-container');
    const visual = root?.querySelector('.sp-heading-visual');
    const accessible = root?.querySelector('.sp-heading-a11y');
    const badgeText = root?.querySelector('.sp-badge-text');
    if (!root || !visual || !accessible) return;

    const titleText = String(data?.text || 'Smart Parking');
    const badgeLabel = String(data?.badge || 'Autonomous CV Lab');

    if (root.dataset.headingText === titleText && root.dataset.badgeText === badgeLabel) return;
    root.dataset.headingText = titleText;
    root.dataset.badgeText = badgeLabel;

    accessible.textContent = `${titleText} — ${badgeLabel}`;
    if (badgeText) badgeText.textContent = badgeLabel;

    visual.replaceChildren();
    const chars = Array.from(titleText);
    const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

    chars.forEach((char, index) => {
        if (char === ' ') {
            const space = document.createElement('span');
            space.className = 'sp-split-space';
            space.innerHTML = '&nbsp;';
            visual.appendChild(space);
            return;
        }
        const span = document.createElement('span');
        span.className = 'sp-split-char';
        span.textContent = char;
        span.setAttribute('aria-hidden', 'true');
        span.style.animationDelay = reduceMotion ? '0ms' : `${Math.min(index * 26, 650)}ms`;
        visual.appendChild(span);
    });
}
"""

# -----------------------------------------------------------------------------
# 2. Results Banner: React Bits FadeContent + Spotlight Glass Accent
# -----------------------------------------------------------------------------
_RESULT_CSS = """
html, body {
    margin: 0;
    padding: 0;
    overflow: hidden !important;
    background: transparent !important;
}
.sp-result-root {
    width: 100%;
    color: #f8fafc;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    margin: 0.35rem 0 0.85rem 0;
    box-sizing: border-box;
}
.sp-result-banner {
    box-sizing: border-box;
    display: flex;
    align-items: flex-start;
    gap: 0.95rem;
    width: 100%;
    min-height: 4rem;
    padding: 0.95rem 1.25rem;
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: rgba(13, 21, 39, 0.7);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.08);
    font-size: 0.96rem;
    font-weight: 500;
    line-height: 1.5;
    position: relative;
    overflow: hidden;
    animation: sp-fade-content 420ms cubic-bezier(0.16, 1, 0.3, 1) both;
}
.sp-result-banner::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: currentColor;
    opacity: 0.85;
}
.sp-result-banner[data-tone="danger"] {
    border-color: rgba(244, 63, 94, 0.45);
    background: rgba(244, 63, 94, 0.14);
    color: #fecdd3;
    box-shadow: 0 8px 32px rgba(244, 63, 94, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
}
.sp-result-banner[data-tone="danger"]::before {
    background: linear-gradient(90deg, #f43f5e, #fb7185, #f43f5e);
}
.sp-result-banner[data-tone="warning"] {
    border-color: rgba(245, 158, 11, 0.45);
    background: rgba(245, 158, 11, 0.14);
    color: #fef3c7;
    box-shadow: 0 8px 32px rgba(245, 158, 11, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
}
.sp-result-banner[data-tone="warning"]::before {
    background: linear-gradient(90deg, #f59e0b, #fcd34d, #f59e0b);
}
.sp-result-banner[data-tone="success"] {
    border-color: rgba(16, 185, 129, 0.45);
    background: rgba(16, 185, 129, 0.14);
    color: #d1fae5;
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
}
.sp-result-banner[data-tone="success"]::before {
    background: linear-gradient(90deg, #10b981, #6ee7b7, #10b981);
}
.sp-result-banner[data-tone="neutral"] {
    border-color: rgba(56, 189, 248, 0.4);
    background: rgba(56, 189, 248, 0.12);
    color: #e0f2fe;
    box-shadow: 0 8px 32px rgba(56, 189, 248, 0.15), inset 0 1px 1px rgba(255, 255, 255, 0.1);
}
.sp-result-banner[data-tone="neutral"]::before {
    background: linear-gradient(90deg, #0284c7, #38bdf8, #818cf8);
}
.sp-result-mark {
    display: inline-flex;
    flex: 0 0 1.6rem;
    align-items: center;
    justify-content: center;
    width: 1.6rem;
    height: 1.6rem;
    margin-top: 0.05rem;
    border-radius: 9999px;
    border: 1px solid currentColor;
    font-size: 0.9rem;
    font-weight: 700;
    background: rgba(255, 255, 255, 0.06);
}
.sp-result-content {
    flex: 1;
    min-width: 0;
}
.sp-result-title {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 0.2rem;
    letter-spacing: -0.01em;
}
.sp-result-message {
    color: rgba(248, 250, 252, 0.9);
    font-size: 0.92rem;
    line-height: 1.45;
}

@keyframes sp-fade-content {
    from {
        opacity: 0;
        transform: translateY(12px);
        filter: blur(4px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
        filter: blur(0);
    }
}

@media (max-width: 640px) {
    .sp-result-banner {
        padding: 0.85rem 1rem;
        font-size: 0.9rem;
    }
}
@media (prefers-reduced-motion: reduce) {
    .sp-result-banner {
        animation: none !important;
    }
}
"""

_RESULT_JS = """
export default function({ parentElement, data }) {
    const banner = parentElement.querySelector('.sp-result-banner');
    const title = banner?.querySelector('.sp-result-title');
    const message = banner?.querySelector('.sp-result-message');
    const mark = banner?.querySelector('.sp-result-mark');
    if (!banner || !message || !mark || !title) return;

    const text = String(data?.message || 'Analysis complete.');
    const tones = new Set(['danger', 'warning', 'success', 'neutral']);
    const tone = tones.has(data?.tone) ? data.tone : 'neutral';
    banner.dataset.tone = tone;

    if (tone === 'danger') {
        title.textContent = 'Potential Violation Detected';
        mark.textContent = '!';
    } else if (tone === 'warning') {
        title.textContent = 'Human Review Required';
        mark.textContent = '⚠';
    } else if (tone === 'success') {
        title.textContent = 'Scene Compliant';
        mark.textContent = '✓';
    } else {
        title.textContent = 'Scene Analysis';
        mark.textContent = 'i';
    }
    message.textContent = text;
}
"""

# -----------------------------------------------------------------------------
# 3. Loading Indicator: React Bits Wave Preloader (Zero Scrollbar)
# -----------------------------------------------------------------------------
_LOADING_CSS = """
html, body {
    margin: 0;
    padding: 0;
    overflow: hidden !important;
    background: transparent !important;
}
.sp-loading-root {
    box-sizing: border-box;
    width: 100%;
    padding: 0.95rem 1.25rem;
    margin: 0;
    border-radius: 14px;
    background: rgba(13, 21, 39, 0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(56, 189, 248, 0.35);
    color: #f8fafc;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45), 0 0 20px rgba(56, 189, 248, 0.15);
    position: relative;
    overflow: hidden !important;
    animation: sp-fade-content 300ms ease forwards;
}
.sp-loading-wave-bar {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #06b6d4 0%, #38bdf8 45%, #818cf8 80%, #06b6d4 100%);
    background-size: 200% 100%;
    animation: sp-wave-travel 1.8s linear infinite;
}
.sp-loading-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.45rem;
}
.sp-loading-pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #38bdf8;
    box-shadow: 0 0 12px #38bdf8;
    animation: sp-dot-pulse 1.4s infinite ease-in-out;
}
.sp-loading-title {
    font-size: 0.98rem;
    font-weight: 600;
    color: #f8fafc;
    letter-spacing: -0.01em;
}
.sp-loading-subtitle {
    font-size: 0.83rem;
    color: #94a3b8;
    line-height: 1.45;
    margin-left: 1.25rem;
}

@keyframes sp-wave-travel {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

@media (prefers-reduced-motion: reduce) {
    .sp-loading-wave-bar, .sp-loading-pulse-dot {
        animation: none !important;
    }
}
"""

_LOADING_JS = """
export default function({ parentElement, data }) {
    const root = parentElement.querySelector('.sp-loading-root');
    const title = root?.querySelector('.sp-loading-title');
    const subtitle = root?.querySelector('.sp-loading-subtitle');
    if (!root || !title) return;

    if (data?.title) title.textContent = String(data.title);
    if (data?.subtitle && subtitle) subtitle.textContent = String(data.subtitle);
}
"""

# -----------------------------------------------------------------------------
# Component Registrations via Streamlit Components v2
# -----------------------------------------------------------------------------
_heading_component = st.components.v2.component(
    name="smart_parking.reactbits_heading",
    html=(
        '<div class="sp-heading-container" role="heading" aria-level="1">'
        '  <span class="sp-heading-a11y">Smart Parking Violation Detection — Autonomous CV Lab</span>'
        '  <div class="sp-heading-title-group">'
        '    <p class="sp-heading-visual" aria-hidden="true"></p>'
        '  </div>'
        '  <div class="sp-shiny-badge" aria-hidden="true">'
        '    <span class="sp-shiny-dot"></span>'
        '    <span class="sp-badge-text">Autonomous CV Lab</span>'
        '  </div>'
        '</div>'
    ),
    css=_HEADING_CSS,
    js=_HEADING_JS,
)

_result_component = st.components.v2.component(
    name="smart_parking.reactbits_result_banner",
    html=(
        '<div class="sp-result-root">'
        '  <div class="sp-result-banner" role="status" aria-live="polite">'
        '    <span class="sp-result-mark" aria-hidden="true">✓</span>'
        '    <div class="sp-result-content">'
        '      <div class="sp-result-title">Analysis Complete</div>'
        '      <div class="sp-result-message">Processing finished.</div>'
        '    </div>'
        '  </div>'
        '</div>'
    ),
    css=_RESULT_CSS,
    js=_RESULT_JS,
)

_loading_component = st.components.v2.component(
    name="smart_parking.reactbits_loading",
    html=(
        '<div class="sp-loading-root" role="status" aria-live="polite">'
        '  <div class="sp-loading-wave-bar"></div>'
        '  <div class="sp-loading-header">'
        '    <span class="sp-loading-pulse-dot"></span>'
        '    <span class="sp-loading-title">Evaluating Scene Evidence...</span>'
        '  </div>'
        '  <div class="sp-loading-subtitle">'
        '    YOLOv8 Detection • SegFormer Pavement Reasoning • ALPR Plate OCR'
        '  </div>'
        '</div>'
    ),
    css=_LOADING_CSS,
    js=_LOADING_JS,
)


def animated_heading(
    text: str = "Smart Parking",
    *,
    badge: str = "Autonomous CV Lab",
    key: str = "smart-parking-heading"
) -> None:
    """Renders the React Bits SplitText animated heading with a ShinyText pill badge."""
    _heading_component(data={"text": text, "badge": badge}, width="stretch", height=68, key=key)


def animated_result_banner(tone: str, message: str, *, key: str) -> None:
    """Renders the React Bits FadeContent result banner with frosted glass tone accents."""
    _result_component(
        data={"tone": tone, "message": message},
        width="stretch",
        height=110,
        key=key,
    )


def animated_loading_card(
    title: str = "Analyzing parking scene across 6 priority categories...",
    subtitle: str = "YOLOv8 Vehicle Detection • Pavement & Sidewalk Segmentation • ALPR Plate Recognition",
    *,
    key: str = "smart-parking-loading"
) -> None:
    """Renders the React Bits Wave Preloader loading card during inference without scrollbars."""
    _loading_component(
        data={"title": title, "subtitle": subtitle},
        width="stretch",
        height=90,
        key=key,
    )
