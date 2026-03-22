"""RunningHub workflow registry and shot-to-workflow planning."""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .delivery import StoryboardPackage


class RunningHubWorkflowProfile(BaseModel):
    """Curated RunningHub workflow profile used by the previs pipeline."""

    workflow_key: str
    category: str
    title: str
    page_url: str
    recommended_role: str
    mode: str = ""
    required_inputs: list[str] = Field(default_factory=list)
    optional_inputs: list[str] = Field(default_factory=list)
    notes: str = ""
    workflow_id_env: str = ""


class RunningHubVideoPlanItem(BaseModel):
    """One planned RunningHub video task mapped from a generated segment."""

    segment_id: str
    shot_id: str
    workflow_key: str
    workflow_title: str
    workflow_page_url: str
    mode: str
    rationale: str
    workflow_id_env: str
    request_contract: dict[str, object] = Field(default_factory=dict)


class RunningHubWorkflowSuite(BaseModel):
    """Curated RunningHub workflow suite for industrial previs."""

    generated_at: str
    profiles: list[RunningHubWorkflowProfile] = Field(default_factory=list)


def recommended_runninghub_workflows() -> list[RunningHubWorkflowProfile]:
    """Return the curated workflow set recommended for this pipeline."""

    return [
        RunningHubWorkflowProfile(
            workflow_key="rh_char_identity_forge_v1",
            category="character",
            title="〖Consistency〗Generate character material set",
            page_url="https://www.runninghub.cn/post/1859173707481583618",
            recommended_role="Fuse face, clothing, and pose references into a stable hero identity set.",
            required_inputs=["portrait_image", "clothing_image", "character_prompt"],
            optional_inputs=["pose_control", "expression_variants"],
            notes="Best suited for initial identity forging and training-set style portrait packs.",
            workflow_id_env="RUNNINGHUB_WORKFLOW_RH_CHAR_IDENTITY_FORGE_V1",
        ),
        RunningHubWorkflowProfile(
            workflow_key="rh_char_sheet_multiview_v1",
            category="character",
            title="Flux Consistent Character Sheet",
            page_url="https://www.runninghub.cn/post/1868838232778752001",
            recommended_role="Generate multiview turnaround sheets from one approved hero portrait.",
            required_inputs=["hero_reference_image", "pose_sheet", "character_prompt"],
            optional_inputs=["ipadapter_reference", "upscale_toggle"],
            notes="Recommended for stable front, three-quarter, profile, and close-up angles.",
            workflow_id_env="RUNNINGHUB_WORKFLOW_RH_CHAR_SHEET_MULTIVIEW_V1",
        ),
        RunningHubWorkflowProfile(
            workflow_key="rh_scene_set_forge_v1",
            category="scene",
            title="Flux+Redux换背景新思路",
            page_url="https://www.runninghub.cn/post/1864317414564626433",
            recommended_role="Lock a production set plate or environment background before shot rendering.",
            required_inputs=["foreground_or_character_image", "background_reference_image"],
            optional_inputs=["repair_mask", "set_prompt"],
            notes="Useful for establishing plates and background continuity anchors.",
            workflow_id_env="RUNNINGHUB_WORKFLOW_RH_SCENE_SET_FORGE_V1",
        ),
        RunningHubWorkflowProfile(
            workflow_key="rh_shot_keyframe_faceid_v1",
            category="shot_image",
            title="PuLID for FLUX workflow",
            page_url="https://www.runninghub.cn/post/1889954985067814914/",
            recommended_role="Generate identity-stable shot keyframes with face guidance.",
            required_inputs=["face_reference_image", "shot_prompt"],
            optional_inputs=["scene_reference_image", "lora_list", "img2img_source"],
            notes="Recommended as the private face-consistent keyframe workflow base.",
            workflow_id_env="RUNNINGHUB_WORKFLOW_RH_SHOT_KEYFRAME_FACEID_V1",
        ),
        RunningHubWorkflowProfile(
            workflow_key="rh_shot_relight_match_v1",
            category="shot_post",
            title="IC-Light光照控制_在线打光",
            page_url="https://www.runninghub.cn/post/1897257503439147010",
            recommended_role="Match lighting across shots and scene plates after keyframe generation.",
            required_inputs=["foreground_image"],
            optional_inputs=["background_reference_image", "lighting_prompt"],
            notes="Important for keeping character identity believable under changing shot setups.",
            workflow_id_env="RUNNINGHUB_WORKFLOW_RH_SHOT_RELIGHT_MATCH_V1",
        ),
        RunningHubWorkflowProfile(
            workflow_key="rh_shot_repair_inpaint_v1",
            category="shot_post",
            title="Photoshop创意填充Inpaint",
            page_url="https://www.runninghub.cn/post/1875182805771866113",
            recommended_role="Repair faces, hands, clothing seams, and props without regenerating the whole frame.",
            required_inputs=["source_image", "mask"],
            optional_inputs=["repair_prompt", "reference_image"],
            notes="Keep this as the default local repair workflow in editorial loops.",
            workflow_id_env="RUNNINGHUB_WORKFLOW_RH_SHOT_REPAIR_INPAINT_V1",
        ),
        RunningHubWorkflowProfile(
            workflow_key="rh_shot_i2v_wan22_full_v1",
            category="shot_video",
            title="wan2.2_14B官方版本_全功能",
            page_url="https://www.runninghub.cn/post/1954176509042946049",
            recommended_role="Default production image-to-video workflow for previs clips.",
            mode="auto",
            required_inputs=["prompt"],
            optional_inputs=["first_frame_image", "last_frame_image", "negative_prompt"],
            notes="Supports text-to-video, image-to-video, and first-last-frame video. Use as the primary clip generator.",
            workflow_id_env="RUNNINGHUB_WORKFLOW_RH_SHOT_I2V_WAN22_FULL_V1",
        ),
        RunningHubWorkflowProfile(
            workflow_key="rh_shot_i2v_wan21_hq_v1",
            category="shot_video",
            title="wan2.1图生视频（高清放大+补帧）",
            page_url="https://www.runninghub.cn/post/1921072224961495041",
            recommended_role="High-quality rerender workflow for hero shots, close-ups, and emotional beats.",
            mode="quality",
            required_inputs=["first_frame_image", "prompt"],
            optional_inputs=["negative_prompt", "upscale_toggle"],
            notes="Slower but stronger on single-shot quality and detail retention.",
            workflow_id_env="RUNNINGHUB_WORKFLOW_RH_SHOT_I2V_WAN21_HQ_V1",
        ),
        RunningHubWorkflowProfile(
            workflow_key="rh_shot_i2v_framepack_fast_v1",
            category="shot_video",
            title="RunningHub专用节点 Framepack F1图生视频〖高清放大+补帧〗工作流V2优化版",
            page_url="https://www.runninghub.cn/post/1920492525130551297",
            recommended_role="Fast previs workflow for motion blocking and iteration passes.",
            mode="fast",
            required_inputs=["first_frame_image", "prompt"],
            optional_inputs=["negative_prompt"],
            notes="Use only for quick previz or motion validation, not final hero delivery.",
            workflow_id_env="RUNNINGHUB_WORKFLOW_RH_SHOT_I2V_FRAMEPACK_FAST_V1",
        ),
    ]


def runninghub_workflow_suite() -> RunningHubWorkflowSuite:
    """Build the curated workflow suite document."""

    return RunningHubWorkflowSuite(
        generated_at=datetime.now(timezone.utc).isoformat(),
        profiles=recommended_runninghub_workflows(),
    )


def select_runninghub_video_profile(
    shot_type: str,
    mode: str = "auto",
    primary_subject_count: int = 1,
) -> RunningHubWorkflowProfile:
    """Select the best RunningHub video workflow profile for a shot."""

    profiles = {profile.workflow_key: profile for profile in recommended_runninghub_workflows()}
    if mode == "fast":
        return profiles["rh_shot_i2v_framepack_fast_v1"]
    if mode == "quality":
        return profiles["rh_shot_i2v_wan21_hq_v1"]

    if shot_type in {"CLOSE_UP", "MCU"}:
        return profiles["rh_shot_i2v_wan21_hq_v1"]
    if primary_subject_count >= 2 and shot_type == "OVER_SHOULDER":
        return profiles["rh_shot_i2v_wan22_full_v1"]
    return profiles["rh_shot_i2v_wan22_full_v1"]


def build_runninghub_video_bundle(
    package: StoryboardPackage,
    mode: str = "auto",
) -> list[RunningHubVideoPlanItem]:
    """Build RunningHub video plans for each generated segment."""

    shots_by_id = {shot.shot_id: shot for shot in package.shots}
    bundle: list[RunningHubVideoPlanItem] = []

    for segment in package.video_segments:
        shot = shots_by_id.get(segment.shot_id)
        shot_type = shot.shot_type if shot else ""
        primary_subjects = list(shot.primary_subjects) if shot else []
        profile = select_runninghub_video_profile(
            shot_type=shot_type,
            mode=mode,
            primary_subject_count=len(primary_subjects),
        )

        first_frame_candidates = list(segment.character_reference_images or [])
        scene_candidates = list(segment.scene_reference_images or [])
        if not first_frame_candidates and shot:
            first_frame_candidates = list(shot.character_reference_images)
        if not scene_candidates and shot:
            scene_candidates = list(shot.scene_reference_images)

        if mode == "fast":
            rationale = "fast previz pass for blocking and motion validation"
        elif mode == "quality":
            rationale = "high-quality pass for close detail or hero shot retention"
        elif shot_type in {"CLOSE_UP", "MCU"}:
            rationale = "auto-selected high-quality workflow for identity-sensitive close framing"
        else:
            rationale = "auto-selected default workflow for general previs clip generation"

        bundle.append(
            RunningHubVideoPlanItem(
                segment_id=segment.segment_id,
                shot_id=segment.shot_id,
                workflow_key=profile.workflow_key,
                workflow_title=profile.title,
                workflow_page_url=profile.page_url,
                mode=profile.mode or mode,
                rationale=rationale,
                workflow_id_env=profile.workflow_id_env,
                request_contract={
                    "prompt": segment.prompt,
                    "negative_prompt": segment.negative_prompt,
                    "duration_seconds": segment.generation_duration_seconds,
                    "first_frame_candidates": first_frame_candidates[:4],
                    "scene_reference_images": scene_candidates[:4],
                    "faceid_profile_ids": dict(segment.faceid_profile_ids),
                    "continuity_group": segment.continuity_group,
                    "reference_shot_id": segment.reference_shot_id,
                    "selected_mode": mode,
                },
            )
        )

    return bundle
