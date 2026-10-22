import os
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# Release mode uses the built frontend in dist/
_RELEASE = True

_COMPONENT_DIR = Path(__file__).resolve().parent
_BUILD_DIR = _COMPONENT_DIR / "frontend" / "dist"

_antigravity_component = None


def _get_component():
    global _antigravity_component
    if _antigravity_component is not None:
        return _antigravity_component

    if not _RELEASE:
        _antigravity_component = components.declare_component(
            "antigravity_hero",
            url="http://localhost:5173",
        )
    else:
        if _BUILD_DIR.exists():
            _antigravity_component = components.declare_component(
                "antigravity_hero",
                path=str(_BUILD_DIR),
            )
    return _antigravity_component


def antigravity_hero(
    title: str = "Smart Parking Violation Detection",
    subtitle: str = "AI-powered parking scene analysis",
    badge: str = "Autonomous CV Lab",
    *,
    key: str = "antigravity-hero"
) -> None:
    """Renders the official React Bits Antigravity WebGL2 particle background behind the header.

    Configuration:
    - colors: ['#5227FF', '#FF9FFC', '#B497CF']
    - count: 1500
    - magnetRadius: 9.5
    - ringRadius: 4
    - particleSize: 0.9
    - autoAnimate: false
    - particleShape: 'sphere'
    - pulseSpeed: 1.7
    - height: 400px
    """
    comp = _get_component()
    if comp is not None:
        comp(
            title=title,
            subtitle=subtitle,
            badge=badge,
            key=key,
            default=None,
        )
    else:
        from ui_animations import animated_heading
        animated_heading(title, badge=badge, key=key)
        st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)
