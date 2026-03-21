"""Tests for the enhanced rule-based script analyser."""

from autocineflow.script_analyzer import analyse_script_rule_based


class TestRuleBasedScriptAnalysis:
    def test_extracts_visual_anchors_and_dialogue_from_english_description(self):
        ctx = analyse_script_rule_based(
            'A man in black coat faces a woman in red dress across a tavern table. '
            '"We end this tonight," Alice said.'
        )

        assert len(ctx.characters) == 2
        assert ctx.characters[0].visual_anchor == "man in black coat"
        assert ctx.characters[1].visual_anchor == "woman in red dress"
        assert ctx.scene_location == "tavern"
        assert ctx.dialogue[0].text == "We end this tonight"
        assert ctx.dialogue[0].speaker_hint == "Alice"

    def test_supports_chinese_scene_descriptions(self):
        ctx = analyse_script_rule_based(
            "穿黑色大衣的男人与红裙女人在酒馆对视。女人低声说：“别再撒谎了。”"
        )

        assert len(ctx.characters) == 2
        assert ctx.characters[0].visual_anchor == "穿黑色大衣的男人"
        assert ctx.characters[1].visual_anchor == "红裙女人"
        assert ctx.scene_location == "tavern"
        assert ctx.dialogue[0].text == "别再撒谎了"
        assert ctx.dialogue[0].speaker_hint == "女人"

    def test_detects_two_character_angry_scene_in_chinese(self):
        ctx = analyse_script_rule_based("两人在酒馆对坐，空气越来越紧张，其中一人突然愤怒地拍桌。")

        assert len(ctx.characters) == 2
        assert ctx.detected_emotion == "angry"
        assert ctx.primary_focus_char_id == "CHAR_A"
