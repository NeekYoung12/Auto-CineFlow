"""Script analyser: extracts characters, emotion, dialogue, and scene intent.

This module provides two implementations:
  1. A rule-based fallback analyser that works without an API key.
  2. An LLM-backed analyser using an OpenAI-compatible API.

REQ-01: Character State Machine bootstrapping.
REQ-04: Semantic-Parameter Mapping (emotion detection).
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from typing import Optional

from .config_loader import resolve_openai_settings
from .models import (
    Character,
    CharacterFacing,
    DialogueLine,
    EMOTION_MATRIX,
    Position,
    SceneContext,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Keyword lists for the rule-based fallback
# ---------------------------------------------------------------------------

_EMOTION_KEYWORDS: dict[str, list[str]] = {
    "angry": ["angry", "anger", "rage", "yell", "slam", "furious", "愤怒", "怒", "拍桌", "怒吼", "爆发"],
    "furious": ["furious", "erupts", "explode", "explodes", "burst", "暴怒", "失控", "咆哮"],
    "sad": ["sad", "cry", "tear", "grieve", "sob", "weep", "伤心", "哭", "流泪", "悲伤"],
    "happy": ["happy", "smile", "laugh", "cheerful", "delight", "开心", "笑", "高兴"],
    "joyful": ["joyful", "ecstatic", "elated", "thrilled", "欢呼", "喜悦", "雀跃"],
    "tense": ["tense", "nervous", "anxious", "glare", "silence", "tension", "紧张", "压迫", "对峙", "沉默"],
    "scared": ["scared", "fear", "terrified", "shiver", "afraid", "害怕", "恐惧", "颤抖"],
    "calm": ["calm", "quiet", "peaceful", "serene", "tranquil", "平静", "安静", "宁静"],
    "romantic": ["romantic", "love", "kiss", "embrace", "intimate", "浪漫", "暧昧", "亲吻", "拥抱"],
    "neutral": [],
}

_TWO_PERSON_PATTERNS = [
    r"\btwo\b",
    r"\bboth\b",
    r"\beach other\b",
    r"\bfacing\b",
    r"\bopposite\b",
    r"\bacross\b",
    r"\bconversation\b",
    r"\bdialogue\b",
    r"\bthey\b",
    r"两人",
    r"两个人",
    r"彼此",
    r"对坐",
    r"对视",
    r"二人",
]

_ENGLISH_ROLE_PATTERN = re.compile(
    r"\b(?:a|an|the)?\s*"
    r"(?:(?:young|old|middle-aged|tall|short|slim|broad-shouldered|bearded|scarred|"
    r"elegant|tired|nervous|stern|grizzled|hooded|masked|blonde|dark-haired|red-haired|"
    r"grey-haired|gray-haired|well-dressed|disheveled|wounded|smiling|tearful|drunk|dusty)\s+){0,4}"
    r"(?:man|woman|person|figure|bartender|waitress|waiter|stranger|wanderer|soldier|"
    r"detective|informant|doctor|nurse|teacher|girl|boy|child|guard|officer|chef|musician)\b"
    r"(?:\s+(?:with|in|wearing|dressed in|holding)\s+"
    r"(?:(?!\b(?:faces?|facing|across|opposite|beside|next|while|and)\b)[A-Za-z-]+\s*){1,6})?",
    re.IGNORECASE,
)
_CHINESE_ROLE_PATTERN = re.compile(
    r"(?:穿着[^，。；]{1,12}的|穿[^，。；]{1,12}的|身穿[^，。；]{1,12}的|留着[^，。；]{1,10}的|戴着[^，。；]{1,10}的)?"
    r"(?:高大的|瘦削的|疲惫的|年轻的|年老的|沉默的|愤怒的|紧张的|冷静的|红裙|黑色大衣|白衬衫|风衣)?"
    r"(?:男人|女人|男孩|女孩|酒保|旅人|士兵|警探|医生|老师|陌生人|流浪者)",
)
_QUOTE_PATTERN = re.compile(r'"([^"]+)"|“([^”]+)”|「([^」]+)」')
_SPEAKER_PATTERN = re.compile(
    r"(?:(?P<speaker_en>[A-Z][a-z]+)\s+(?:said|asked|whispered|shouted)|"
    r"(?P<speaker_zh>[\u4e00-\u9fff]{1,4})(?:低声说|轻声说|说|问|喊道|怒吼))",
)

_NAME_STOPWORDS = {
    "A",
    "An",
    "The",
    "Two",
    "One",
    "In",
    "On",
    "At",
    "Across",
    "Facing",
    "When",
    "After",
    "Before",
    "As",
}

_LOCATION_KEYWORDS: dict[str, list[str]] = {
    "tavern": ["tavern", "pub", "bar", "inn", "酒馆", "酒吧", "客栈"],
    "field": ["field", "meadow", "plain", "田野", "原野"],
    "alley": ["alley", "backstreet", "巷子", "小巷"],
    "office": ["office", "meeting room", "办公室", "会议室"],
    "street": ["street", "road", "crosswalk", "街道", "马路"],
    "forest": ["forest", "woods", "树林", "森林"],
    "room": ["room", "bedroom", "living room", "房间", "卧室", "客厅"],
}

_SCENE_TAG_KEYWORDS: dict[str, list[str]] = {
    "night": ["night", "midnight", "夜", "深夜"],
    "rain": ["rain", "storm", "雨", "暴雨"],
    "candlelight": ["candle", "candles", "烛光"],
    "smoke": ["smoke", "fog", "mist", "烟", "雾"],
    "table": ["table", "desk", "桌", "桌子"],
}


def _normalise_spaces(value: str) -> str:
    """Collapse repeated whitespace."""

    return re.sub(r"\s+", " ", value).strip(" ,.;:，。；：")


def _trim_anchor_tail(candidate: str) -> str:
    """Remove relationship/action tails that should not become visual anchors."""

    trimmed = re.split(
        r"\b(?:faces?|facing|across|opposite|beside|next to|while|sits?|stands?|looks?|stares?|glances?)\b",
        candidate,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    trimmed = re.split(r"(?:对坐|对视|站在|看着|望着|旁边)", trimmed, maxsplit=1)[0]
    trimmed = re.split(
        r"\b(?:in|at|under|inside|outside)\b\s+(?:a|an|the)?\s*(?:(?:\w+)\s+){0,2}"
        r"(?:tavern|pub|bar|inn|alley|field|meadow|plain|street|road|crosswalk|"
        r"forest|woods|office|meeting room|room|bedroom|living room|night|midnight)\b.*",
        trimmed,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    trimmed = re.sub(r"^(?:a|an|the)\s+", "", trimmed, flags=re.IGNORECASE)
    return _normalise_spaces(trimmed)


def _detect_emotion(text: str) -> str:
    """Return the dominant emotion keyword found in the text."""

    lower = text.lower()
    scored = {emotion: 0 for emotion in EMOTION_MATRIX}
    for emotion, keywords in _EMOTION_KEYWORDS.items():
        if emotion == "neutral":
            continue
        for keyword in keywords:
            if keyword.lower() in lower or keyword in text:
                scored[emotion] += 1

    ranked = sorted(
        ((emotion, score) for emotion, score in scored.items() if score > 0),
        key=lambda item: (item[1], EMOTION_MATRIX[item[0]]["motion"]),
        reverse=True,
    )
    return ranked[0][0] if ranked else "neutral"


def _extract_names(text: str) -> list[str]:
    """Extract likely English proper names."""

    names = []
    for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b", text):
        candidate = match.group(0).strip()
        if candidate in _NAME_STOPWORDS:
            continue
        names.append(candidate)
    return names


def _detect_character_count(text: str) -> int:
    """Heuristically detect whether the scene involves 1 or 2 characters."""

    lower = text.lower()
    for pattern in _TWO_PERSON_PATTERNS:
        if re.search(pattern, lower) or re.search(pattern, text):
            return 2

    if len(set(_extract_names(text))) >= 2:
        return 2

    role_hits = list(_ENGLISH_ROLE_PATTERN.finditer(text)) + list(_CHINESE_ROLE_PATTERN.finditer(text))
    if len(role_hits) >= 2:
        return 2
    return 1


def _is_useful_anchor(candidate: str) -> bool:
    """Reject placeholders that are too generic to help prompting."""

    cleaned = _normalise_spaces(candidate)
    lowered = cleaned.lower()
    if len(cleaned) < 3:
        return False
    if lowered in {"two people", "people", "two", "one", "person", "figure", "两人", "两个人"}:
        return False
    if lowered.startswith("two ") and lowered.endswith(" people"):
        return False
    return True


def _extract_visual_anchors(text: str, num_chars: int) -> list[str]:
    """Extract or synthesise visual anchor descriptions for characters."""

    anchor_candidates: list[str] = []
    cleaned = _QUOTE_PATTERN.sub("", text)
    segments = [cleaned]
    segments.extend(part for part in re.split(r"[与和及跟]", cleaned) if part)

    for segment in segments:
        for match in _ENGLISH_ROLE_PATTERN.finditer(segment):
            candidate = _trim_anchor_tail(match.group(0))
            if _is_useful_anchor(candidate):
                anchor_candidates.append(candidate)

        for match in _CHINESE_ROLE_PATTERN.finditer(segment):
            candidate = _trim_anchor_tail(match.group(0))
            chinese_parts = [part for part in re.split(r"[与和及跟]", candidate) if part]
            if not chinese_parts:
                chinese_parts = [candidate]
            for part in chinese_parts:
                normalised = _normalise_spaces(part)
                if _is_useful_anchor(normalised):
                    anchor_candidates.append(normalised)

    for name in _extract_names(cleaned):
        if _is_useful_anchor(name):
            anchor_candidates.append(name)

    deduped = list(OrderedDict.fromkeys(anchor_candidates))
    defaults = ["person A", "person B"]
    while len(deduped) < num_chars:
        deduped.append(defaults[len(deduped)])
    return deduped[:num_chars]


def _extract_dialogue(text: str) -> list[DialogueLine]:
    """Extract quoted dialogue and best-effort speaker hints."""

    dialogue: list[DialogueLine] = []
    for match in _QUOTE_PATTERN.finditer(text):
        content = next(group for group in match.groups() if group)
        prefix = text[max(0, match.start() - 32): match.start()]
        suffix = text[match.end(): min(len(text), match.end() + 32)]
        speaker_match = _SPEAKER_PATTERN.search(prefix)
        if speaker_match is None:
            speaker_match = _SPEAKER_PATTERN.search(suffix)
        speaker_hint = None
        if speaker_match:
            speaker_hint = speaker_match.group("speaker_en") or speaker_match.group("speaker_zh")
        if speaker_hint:
            speaker_hint = re.sub(r"(低声|轻声)$", "", speaker_hint)

        dialogue.append(DialogueLine(text=_normalise_spaces(content), speaker_hint=speaker_hint))
    return dialogue


def _detect_scene_location(text: str) -> str:
    """Return a best-effort scene location tag."""

    lower = text.lower()
    for location, keywords in _LOCATION_KEYWORDS.items():
        if any(keyword.lower() in lower or keyword in text for keyword in keywords):
            return location
    return ""


def _build_scene_tags(text: str, scene_location: str, emotion: str) -> list[str]:
    """Extract environment and mood tags that help prompt assembly."""

    lower = text.lower()
    tags: list[str] = []
    if scene_location:
        tags.append(scene_location)
    if emotion != "neutral":
        tags.append(emotion)
    for tag, keywords in _SCENE_TAG_KEYWORDS.items():
        if any(keyword.lower() in lower or keyword in text for keyword in keywords):
            tags.append(tag)
    return list(OrderedDict.fromkeys(tags))


def _infer_primary_focus_char_id(description: str, num_chars: int) -> str | None:
    """Pick a best-effort focus character for high-emotion beats."""

    if num_chars == 0:
        return None
    if num_chars == 1:
        return "CHAR_A"
    if any(token in description.lower() for token in ("one of them", "first", "he ")) or "其中一人" in description:
        return "CHAR_A"
    return "CHAR_A"


def _build_character_state(anchors: list[str]) -> list[Character]:
    """Bootstrap characters with stable IDs and initial screen positions."""

    characters: list[Character] = []
    for i, anchor in enumerate(anchors):
        char_id = f"CHAR_{'A' if i == 0 else 'B'}"
        pos = Position(x=0.25 if i == 0 else 0.75, y=0.5)
        facing = CharacterFacing.RIGHT if i == 0 else CharacterFacing.LEFT
        characters.append(
            Character(
                char_id=char_id,
                visual_anchor=anchor,
                pos=pos,
                facing=facing,
            )
        )
    return characters


def _parse_json_response(raw: str) -> dict:
    """Parse a JSON response, accepting fenced markdown wrappers."""

    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return json.loads(stripped)


def _extract_chat_completion_text(response: object) -> str:
    """Extract text content from multiple OpenAI-compatible response shapes."""

    if isinstance(response, str):
        if response.lstrip().lower().startswith("<!doctype html"):
            raise ValueError("Gateway returned HTML instead of an API completion payload.")
        return response

    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            return str(choices[0].get("message", {}).get("content", "") or "")
        if "output_text" in response:
            return str(response["output_text"])

    if hasattr(response, "choices") and response.choices:
        message = response.choices[0].message
        return str(getattr(message, "content", "") or "")

    if hasattr(response, "output_text"):
        return str(response.output_text or "")

    raise ValueError(f"Unsupported completion payload type: {type(response)!r}")


def _is_placeholder_anchor(anchor: str) -> bool:
    """Return True when an anchor is too generic for production use."""

    return _normalise_spaces(anchor).lower() in {"person a", "person b", "person 1", "person 2"}


def _merge_llm_with_rule_based(
    llm_data: dict,
    fallback_context: SceneContext,
) -> dict:
    """Blend LLM output with rule-based extraction to improve production robustness."""

    merged = dict(llm_data)
    fallback_characters = fallback_context.characters
    llm_characters = list(merged.get("characters", []) or [])

    target_count = max(len(llm_characters), len(fallback_characters))
    if target_count == 0:
        target_count = 1

    merged_characters: list[dict] = []
    for index in range(target_count):
        llm_char = llm_characters[index] if index < len(llm_characters) else {}
        fallback_char = fallback_characters[index] if index < len(fallback_characters) else None

        anchor = str(llm_char.get("visual_anchor", "")).strip()
        if (not anchor or _is_placeholder_anchor(anchor)) and fallback_char is not None:
            anchor = fallback_char.visual_anchor

        char_id = str(llm_char.get("char_id", "")).strip()
        if not char_id:
            char_id = fallback_char.char_id if fallback_char is not None else f"CHAR_{'A' if index == 0 else 'B'}"

        merged_characters.append({"char_id": char_id, "visual_anchor": anchor})

    merged["characters"] = merged_characters
    merged["num_characters"] = len(merged_characters)

    if merged.get("emotion") not in EMOTION_MATRIX:
        merged["emotion"] = fallback_context.detected_emotion
    elif merged.get("emotion") == "neutral" and fallback_context.detected_emotion != "neutral":
        merged["emotion"] = fallback_context.detected_emotion

    if not merged.get("scene_location"):
        merged["scene_location"] = fallback_context.scene_location

    llm_tags = [str(tag).strip().lower() for tag in merged.get("scene_tags", []) if str(tag).strip()]
    merged["scene_tags"] = list(OrderedDict.fromkeys(llm_tags + list(fallback_context.scene_tags)))

    dialogue = merged.get("dialogue", [])
    if not dialogue:
        merged["dialogue"] = [
            {"text": line.text, "speaker_hint": line.speaker_hint}
            for line in fallback_context.dialogue
        ]

    if not merged.get("primary_focus_char_id"):
        merged["primary_focus_char_id"] = fallback_context.primary_focus_char_id

    return merged


# ---------------------------------------------------------------------------
# Rule-based analyser (no API required)
# ---------------------------------------------------------------------------


def analyse_script_rule_based(description: str, scene_id: str = "SCENE_01") -> SceneContext:
    """Parse a scene description using rule-based heuristics."""

    emotion = _detect_emotion(description)
    num_chars = _detect_character_count(description)
    anchors = _extract_visual_anchors(description, num_chars)
    dialogue = _extract_dialogue(description)
    scene_location = _detect_scene_location(description)
    scene_tags = _build_scene_tags(description, scene_location, emotion)
    primary_focus_char_id = _infer_primary_focus_char_id(description, num_chars)
    characters = _build_character_state(anchors)

    return SceneContext(
        scene_id=scene_id,
        description=description,
        characters=characters,
        detected_emotion=emotion,
        dialogue=dialogue,
        scene_location=scene_location,
        scene_tags=scene_tags,
        primary_focus_char_id=primary_focus_char_id,
        analysis_source="rule_based",
    )


# ---------------------------------------------------------------------------
# LLM-backed analyser (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a professional film director and script analyst.
Given a scene description, extract:
1. The number of characters (1 or 2).
2. A concise visual anchor description for each character
   (hair colour, clothing, distinguishing features suitable as a Stable Diffusion prompt fragment).
3. The dominant emotion of the scene.
4. The scene location as a short lowercase tag.
5. A list of environment tags useful for prompting.
6. The primary focus character ID if one performer is the emotional centre.
7. Any quoted dialogue fragments with optional speaker hints.

Respond ONLY with valid JSON in this exact format:
{
  "num_characters": 2,
  "characters": [
    {"char_id": "CHAR_A", "visual_anchor": "man in black coat"},
    {"char_id": "CHAR_B", "visual_anchor": "woman in red dress"}
  ],
  "emotion": "tense",
  "scene_location": "tavern",
  "scene_tags": ["tavern", "night", "tense"],
  "primary_focus_char_id": "CHAR_A",
  "dialogue": [
    {"text": "We finish this tonight.", "speaker_hint": "CHAR_A"}
  ]
}
If there is only 1 character, omit CHAR_B from the list.
The emotion must be one of: angry, furious, sad, happy, joyful, tense, scared, calm, romantic, neutral.
"""


def analyse_script_llm(
    description: str,
    scene_id: str = "SCENE_01",
    api_key: Optional[str] = None,
    model: str = "gpt-5",
    base_url: str | None = None,
    config_path: str | None = None,
) -> SceneContext:
    """Parse a scene description using an OpenAI-compatible LLM."""

    try:
        import openai  # noqa: PLC0415
    except ImportError:
        logger.warning("openai package not installed; falling back to rule-based analysis.")
        return analyse_script_rule_based(description, scene_id)

    resolved_key, resolved_base_url = resolve_openai_settings(
        api_key=api_key,
        base_url=base_url,
        config_path=config_path,
    )
    if not resolved_key:
        logger.warning("No OpenAI-compatible API key found; falling back to rule-based analysis.")
        return analyse_script_rule_based(description, scene_id)

    client_kwargs = {"api_key": resolved_key}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    client = openai.OpenAI(**client_kwargs)

    fallback_context = analyse_script_rule_based(description, scene_id)
    last_error: Exception | None = None
    model_candidates = list(OrderedDict.fromkeys([model, "gpt-5", "gpt-4o"]))
    data: dict | None = None

    for candidate in model_candidates:
        try:
            response = client.chat.completions.create(
                model=candidate,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": description},
                ],
                temperature=0.1,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            raw = _extract_chat_completion_text(response)
            data = _parse_json_response(raw)
            if data:
                break
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    if not data:
        logger.warning("LLM analysis failed (%s); falling back to rule-based.", last_error)
        return fallback_context

    data = _merge_llm_with_rule_based(data, fallback_context)

    valid_emotions = set(EMOTION_MATRIX.keys())
    emotion = data.get("emotion", "neutral")
    if emotion not in valid_emotions:
        emotion = "neutral"

    characters_payload = data.get("characters", [])
    characters = _build_character_state(
        [entry.get("visual_anchor", f"person {index + 1}") for index, entry in enumerate(characters_payload)]
    )
    for index, payload in enumerate(characters_payload):
        if index >= len(characters):
            break
        characters[index] = characters[index].model_copy(
            update={"char_id": payload.get("char_id", characters[index].char_id)}
        )

    dialogue = [
        DialogueLine(
            text=_normalise_spaces(item.get("text", "")),
            speaker_hint=item.get("speaker_hint"),
        )
        for item in data.get("dialogue", [])
        if item.get("text")
    ]

    scene_location = _normalise_spaces(str(data.get("scene_location", ""))).lower()
    scene_tags = [str(tag).strip().lower() for tag in data.get("scene_tags", []) if str(tag).strip()]
    if scene_location and scene_location not in scene_tags:
        scene_tags.insert(0, scene_location)

    return SceneContext(
        scene_id=scene_id,
        description=description,
        characters=characters,
        detected_emotion=emotion,
        dialogue=dialogue,
        scene_location=scene_location,
        scene_tags=list(OrderedDict.fromkeys(scene_tags)),
        primary_focus_char_id=data.get("primary_focus_char_id"),
        analysis_source="llm",
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def analyse_script(
    description: str,
    scene_id: str = "SCENE_01",
    use_llm: bool = True,
    api_key: Optional[str] = None,
    model: str = "gpt-5",
    base_url: str | None = None,
    config_path: str | None = None,
) -> SceneContext:
    """Analyse a scene description, using LLM if available else rule-based fallback."""

    if use_llm:
        return analyse_script_llm(
            description=description,
            scene_id=scene_id,
            api_key=api_key,
            model=model,
            base_url=base_url,
            config_path=config_path,
        )
    return analyse_script_rule_based(description, scene_id)
