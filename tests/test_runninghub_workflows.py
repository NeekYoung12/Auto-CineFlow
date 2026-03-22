"""Tests for curated RunningHub workflow planning."""

import json

from autocineflow.pipeline import CineFlowPipeline
from autocineflow.runninghub_workflows import (
    build_runninghub_video_bundle,
    recommended_runninghub_workflows,
    sanitize_runninghub_prompt,
    select_runninghub_video_profile,
    strengthen_runninghub_negative_prompt,
)


def _build_package():
    pipeline = CineFlowPipeline()
    context = pipeline.run(
        description="A detective in a rain-soaked trench coat confronts a wounded informant in a neon alley at night.",
        num_shots=5,
        scene_id="RUNNINGHUB_SCENE",
        use_llm=False,
        emotion_override="tense",
    )
    package = pipeline.build_storyboard_package(context, project_name="RunningHub Project")
    return pipeline, package


def test_recommended_runninghub_workflows_include_curated_video_replacements():
    profiles = {profile.workflow_key: profile for profile in recommended_runninghub_workflows()}

    assert "rh_shot_i2v_wan22_full_v1" in profiles
    assert "rh_shot_i2v_wan21_hq_v1" in profiles
    assert "rh_shot_i2v_framepack_fast_v1" in profiles
    assert profiles["rh_shot_i2v_wan22_full_v1"].page_url.startswith("https://www.runninghub.cn/post/")


def test_select_runninghub_video_profile_uses_quality_for_closeups_and_fast_when_requested():
    assert select_runninghub_video_profile("CLOSE_UP", mode="auto").workflow_key == "rh_shot_i2v_wan21_hq_v1"
    assert select_runninghub_video_profile("MASTER_SHOT", mode="auto").workflow_key == "rh_shot_i2v_wan21_hq_v1"
    assert select_runninghub_video_profile("MASTER_SHOT", mode="fast").workflow_key == "rh_shot_i2v_framepack_fast_v1"
    assert select_runninghub_video_profile("OVER_SHOULDER", mode="auto", primary_subject_count=2).workflow_key == "rh_shot_i2v_wan22_full_v1"


def test_runninghub_video_bundle_json_exports_auto_quality_and_fast_profiles():
    pipeline, package = _build_package()

    auto_bundle = json.loads(pipeline.runninghub_video_bundle_json(package, mode="auto", indent=2))
    quality_bundle = json.loads(pipeline.runninghub_video_bundle_json(package, mode="quality", indent=2))
    fast_bundle = json.loads(pipeline.runninghub_video_bundle_json(package, mode="fast", indent=2))

    assert len(auto_bundle) == len(package.video_segments)
    assert len(quality_bundle) == len(package.video_segments)
    assert len(fast_bundle) == len(package.video_segments)
    assert quality_bundle[0]["workflow_key"] == "rh_shot_i2v_wan21_hq_v1"
    assert fast_bundle[0]["workflow_key"] == "rh_shot_i2v_framepack_fast_v1"

    auto_keys = {item["workflow_key"] for item in auto_bundle}
    assert "rh_shot_i2v_wan21_hq_v1" in auto_keys


def test_build_runninghub_video_bundle_includes_request_contract_and_env_mapping():
    _, package = _build_package()
    bundle = build_runninghub_video_bundle(package, mode="auto")

    assert bundle[0].workflow_id_env.startswith("RUNNINGHUB_WORKFLOW_")
    assert "prompt" in bundle[0].request_contract
    assert "duration_seconds" in bundle[0].request_contract


def test_sanitize_runninghub_prompt_and_negative_prompt_strengthening():
    prompt = "cinematic photography, 8k resolution, detective, 涓昏, rain-soaked alley, film grain, detective"
    negative = "blurry, deformed"

    cleaned = sanitize_runninghub_prompt(prompt)
    strengthened = strengthen_runninghub_negative_prompt(negative)

    assert "cinematic photography" not in cleaned
    assert "8k resolution" not in cleaned
    assert "detective" in cleaned
    assert "涓昏" not in cleaned
    assert "text" in strengthened
    assert "subtitles" in strengthened
    assert "low resolution" in strengthened
