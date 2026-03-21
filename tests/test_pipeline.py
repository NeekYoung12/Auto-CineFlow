"""Integration tests for the full CineFlowPipeline.

These tests verify end-to-end acceptance criteria from the PRD.
"""

import pytest

from autocineflow.pipeline import CineFlowPipeline


@pytest.fixture
def pipeline():
    return CineFlowPipeline()


@pytest.fixture
def tavern_context(pipeline):
    """Two people sitting across from each other in a tavern."""
    return pipeline.run(
        description="Two people sit facing each other across a tavern table, tension rising.",
        num_shots=5,
        scene_id="TAVERN_SCENE",
        use_llm=False,
        emotion_override="tense",
    )


class TestAxisConsistency:
    def test_axis_side_constant_across_five_shots(self, tavern_context):
        """Logic Pass: axis_side must remain constant across all 5 shots."""
        sides = [sb.camera_angle.axis_side for sb in tavern_context.shot_blocks]
        assert len(set(sides)) == 1, f"Axis changed: {sides}"

    def test_pipeline_validate_axis(self, pipeline, tavern_context):
        """Pipeline utility method must confirm axis consistency."""
        assert pipeline.validate_axis_consistency(tavern_context) is True


class TestGazeLogic:
    def test_char_a_left_faces_right(self, tavern_context):
        """Logic Pass: if CHAR_A is on the left, their facing must be RIGHT."""
        for shot in tavern_context.shot_blocks:
            for char in shot.characters:
                if char.char_id == "CHAR_A" and char.pos.x < 0.5:
                    assert char.facing.value == "RIGHT", (
                        f"CHAR_A on left must face RIGHT, got {char.facing} in shot {shot.shot_index}"
                    )

    def test_char_b_right_faces_left(self, tavern_context):
        """Logic Pass: if CHAR_B is on the right, their facing must be LEFT."""
        for shot in tavern_context.shot_blocks:
            for char in shot.characters:
                if char.char_id == "CHAR_B" and char.pos.x > 0.5:
                    assert char.facing.value == "LEFT", (
                        f"CHAR_B on right must face LEFT, got {char.facing} in shot {shot.shot_index}"
                    )


class TestDataCompleteness:
    def test_all_shots_have_required_fields(self, tavern_context):
        """Data Pass: every ShotBlock must have framing, camera_angle, lighting, motion."""
        for shot in tavern_context.shot_blocks:
            assert shot.framing is not None, f"Shot {shot.shot_index} missing framing"
            assert shot.camera_angle is not None, f"Shot {shot.shot_index} missing camera_angle"
            assert shot.lighting is not None, f"Shot {shot.shot_index} missing lighting"
            assert shot.motion_instruction is not None, f"Shot {shot.shot_index} missing motion"

    def test_visual_anchor_consistent(self, pipeline, tavern_context):
        """Data Pass: CHAR_A visual_anchor must be 100% identical across all shots."""
        assert pipeline.validate_visual_anchor_consistency(tavern_context) is True

    def test_char_id_format(self, tavern_context):
        """Character IDs must follow the CHAR_X format."""
        for shot in tavern_context.shot_blocks:
            for char in shot.characters:
                assert char.char_id.startswith("CHAR_"), f"Unexpected char_id: {char.char_id}"


class TestRenderCompatibility:
    def test_sd_prompts_populated(self, tavern_context):
        """Interface Pass: every shot must have a non-empty SD prompt."""
        for shot in tavern_context.shot_blocks:
            assert shot.sd_prompt, f"Shot {shot.shot_index} has empty sd_prompt"

    def test_negative_prompts_populated(self, tavern_context):
        """Interface Pass: every shot must have a negative prompt."""
        for shot in tavern_context.shot_blocks:
            assert shot.negative_prompt, f"Shot {shot.shot_index} has empty negative_prompt"

    def test_controlnet_coords_in_range(self, pipeline, tavern_context):
        """Interface Pass: ControlNet coordinates must be in [0, 1]."""
        coords = pipeline.controlnet_coords(tavern_context)
        for coord in coords:
            assert 0.0 <= coord["x"] <= 1.0
            assert 0.0 <= coord["y"] <= 1.0

    def test_first_shot_is_master(self, tavern_context):
        """Scene A: first shot must be MASTER_SHOT to establish spatial relationships."""
        from autocineflow.models import ShotType
        assert tavern_context.shot_blocks[0].framing.shot_type == ShotType.MASTER_SHOT


class TestJsonOutput:
    def test_json_serialisable(self, pipeline, tavern_context):
        """The full output must be valid JSON."""
        import json
        json_str = pipeline.to_json(tavern_context)
        data = json.loads(json_str)
        assert "shot_blocks" in data
        assert len(data["shot_blocks"]) == 5

    def test_json_contains_all_required_keys(self, pipeline, tavern_context):
        """Each shot block in the JSON must contain all required top-level keys."""
        import json
        data = json.loads(pipeline.to_json(tavern_context))
        required = {"framing", "camera_angle", "lighting", "motion_instruction"}
        for shot in data["shot_blocks"]:
            for key in required:
                assert key in shot, f"Missing key '{key}' in shot {shot.get('shot_index')}"


class TestSingleCharacterScene:
    def test_single_char_pipeline(self, pipeline):
        ctx = pipeline.run(
            description="A lone wanderer walks through an empty field.",
            num_shots=3,
            use_llm=False,
        )
        assert len(ctx.shot_blocks) == 3
        for shot in ctx.shot_blocks:
            assert shot.framing is not None
