"""Script analyser: extracts characters, emotion, and scene intent from natural language.

This module provides two implementations:
  1. A rule-based fallback analyser that works without an API key.
  2. An LLM-backed analyser using the OpenAI API (GPT-4o).

REQ-01: Character State Machine bootstrapping.
REQ-04: Semantic-Parameter Mapping (emotion detection).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from .models import (
    Character,
    CharacterFacing,
    EMOTION_MATRIX,
    Position,
    SceneContext,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword lists for the rule-based fallback
# ---------------------------------------------------------------------------

_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "angry":    ["angry", "anger", "rage", "furious", "shout", "yell", "slam"],
    "furious":  ["furious", "explode", "burst", "erupts"],
    "sad":      ["sad", "cry", "tear", "grieve", "sob", "weep", "sorrowful"],
    "happy":    ["happy", "smile", "laugh", "joy", "cheerful", "delight"],
    "joyful":   ["joyful", "ecstatic", "elated", "thrilled"],
    "tense":    ["tense", "nervous", "anxious", "stare", "glare", "silence"],
    "scared":   ["scared", "fear", "terrified", "trembles", "shiver"],
    "calm":     ["calm", "quiet", "peaceful", "serene", "tranquil"],
    "romantic": ["romantic", "love", "kiss", "embrace", "gaze", "intimate"],
    "neutral":  [],
}

# Two-character scene indicators
_TWO_PERSON_PATTERNS = [
    r"\btwo\b", r"\bboth\b", r"\beach other\b", r"\bfacing\b",
    r"\bopposite\b", r"\bacross\b", r"\bconversation\b",
    r"\bdialogue\b", r"\bthey\b",
]


def _detect_emotion(text: str) -> str:
    """Return the dominant emotion keyword found in the text."""
    lower = text.lower()
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        if emotion == "neutral":
            continue
        for kw in keywords:
            if kw in lower:
                return emotion
    return "neutral"


def _detect_character_count(text: str) -> int:
    """Heuristically detect whether the scene involves 1 or 2 characters."""
    lower = text.lower()
    for pattern in _TWO_PERSON_PATTERNS:
        if re.search(pattern, lower):
            return 2
    # Count explicit mentions of person/character names (capitalised words)
    caps = re.findall(r"\b[A-Z][a-z]+\b", text)
    unique_caps = set(caps)
    if len(unique_caps) >= 2:
        return 2
    return 1


def _extract_visual_anchors(text: str, num_chars: int) -> list[str]:
    """Extract or synthesise visual anchor descriptions for characters.

    This rule-based version looks for adjective-noun descriptors.
    The LLM version will do this much more accurately.
    """
    anchors: list[str] = []

    # Look for clothing / appearance descriptors
    patterns = [
        r"(\b(?:man|woman|person|figure)\b(?:\s+\w+){0,3})",
        r"(\b(?:wearing|dressed in)\s+[^,\.]+)",
        r"(\b[A-Z][a-z]+\b)",  # Named characters
    ]

    found: list[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, text, re.IGNORECASE))

    # De-duplicate and take first num_chars
    seen: set[str] = set()
    for f in found:
        clean = f.strip()
        if clean.lower() not in seen and len(clean) > 2:
            seen.add(clean.lower())
            anchors.append(clean)
        if len(anchors) >= num_chars:
            break

    # Pad with generic descriptors if needed
    defaults = ["person A", "person B"]
    while len(anchors) < num_chars:
        anchors.append(defaults[len(anchors)])

    return anchors[:num_chars]


# ---------------------------------------------------------------------------
# Rule-based analyser (no API required)
# ---------------------------------------------------------------------------


def analyse_script_rule_based(description: str, scene_id: str = "SCENE_01") -> SceneContext:
    """Parse a scene description using rule-based heuristics.

    This is the fallback path used when no OpenAI API key is available.
    """
    emotion = _detect_emotion(description)
    num_chars = _detect_character_count(description)
    anchors = _extract_visual_anchors(description, num_chars)

    characters: list[Character] = []
    for i, anchor in enumerate(anchors):
        char_id = f"CHAR_{'A' if i == 0 else 'B'}"
        # Initial positions: A left, B right (will be updated by spatial_solver)
        pos = Position(x=0.25 if i == 0 else 0.75, y=0.5)
        facing = CharacterFacing.RIGHT if i == 0 else CharacterFacing.LEFT
        characters.append(
            Character(char_id=char_id, visual_anchor=anchor, pos=pos, facing=facing)
        )

    return SceneContext(
        scene_id=scene_id,
        description=description,
        characters=characters,
        detected_emotion=emotion,
    )


# ---------------------------------------------------------------------------
# LLM-backed analyser (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a professional film director and script analyst.
Given a scene description, extract:
1. The number of characters (1 or 2).
2. A concise visual anchor description for each character
   (hair colour, clothing, distinguishing features — suitable as a Stable Diffusion prompt fragment).
3. The dominant emotion of the scene.

Respond ONLY with valid JSON in this exact format:
{
  "num_characters": <1 or 2>,
  "characters": [
    {"char_id": "CHAR_A", "visual_anchor": "<description>"},
    {"char_id": "CHAR_B", "visual_anchor": "<description>"}
  ],
  "emotion": "<one of: angry, furious, sad, happy, joyful, tense, scared, calm, romantic, neutral>"
}
If there is only 1 character, omit CHAR_B from the list.
"""


def analyse_script_llm(
    description: str,
    scene_id: str = "SCENE_01",
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
) -> SceneContext:
    """Parse a scene description using GPT-4o.

    Falls back to rule-based analysis if the API call fails.

    Args:
        description: Raw scene text from the user.
        scene_id:    Identifier for this scene.
        api_key:     OpenAI API key.  Reads OPENAI_API_KEY env var if not provided.
        model:       OpenAI model name.

    Returns:
        Populated SceneContext ready for the Director filter.
    """
    try:
        import openai  # noqa: PLC0415
    except ImportError:
        logger.warning("openai package not installed; falling back to rule-based analysis.")
        return analyse_script_rule_based(description, scene_id)

    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_key:
        logger.warning("OPENAI_API_KEY not set; falling back to rule-based analysis.")
        return analyse_script_rule_based(description, scene_id)

    client = openai.OpenAI(api_key=resolved_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": description},
            ],
            temperature=0.2,
            max_tokens=512,
        )
        raw = response.choices[0].message.content or ""
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM analysis failed (%s); falling back to rule-based.", exc)
        return analyse_script_rule_based(description, scene_id)

    # Build characters
    characters: list[Character] = []
    valid_emotions = set(EMOTION_MATRIX.keys())
    emotion = data.get("emotion", "neutral")
    if emotion not in valid_emotions:
        emotion = "neutral"

    for i, char_data in enumerate(data.get("characters", [])):
        pos = Position(x=0.25 if i == 0 else 0.75, y=0.5)
        facing = CharacterFacing.RIGHT if i == 0 else CharacterFacing.LEFT
        characters.append(
            Character(
                char_id=char_data.get("char_id", f"CHAR_{'A' if i == 0 else 'B'}"),
                visual_anchor=char_data.get("visual_anchor", f"person {i + 1}"),
                pos=pos,
                facing=facing,
            )
        )

    return SceneContext(
        scene_id=scene_id,
        description=description,
        characters=characters,
        detected_emotion=emotion,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def analyse_script(
    description: str,
    scene_id: str = "SCENE_01",
    use_llm: bool = True,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
) -> SceneContext:
    """Analyse a scene description, using LLM if available else rule-based fallback.

    Args:
        description: Raw scene description text.
        scene_id:    Scene identifier string.
        use_llm:     If True, attempt LLM analysis first.
        api_key:     Optional OpenAI API key override.
        model:       OpenAI model to use.

    Returns:
        SceneContext populated with characters and emotion.
    """
    if use_llm:
        return analyse_script_llm(description, scene_id, api_key, model)
    return analyse_script_rule_based(description, scene_id)
