# Auto-CineFlow

An LLM-driven cinematic storyboard parameter generator. Converts a 1–2 person scene description into a structured JSON instruction set (per-shot `ShotBlock`) that can drive AI image/video workflows.

## Features

* **[REQ-01] Character State Machine** – maintains character IDs, visual anchors and canvas positions across the entire shot sequence
* **[REQ-02] 180° Axis Guard** – vector-based cross-product check prevents axis crossing between consecutive shots
* **[REQ-03] Shot Template Library** – Master Shot (24 mm), Medium Shot (35 mm), MCU (50 mm), Close-Up (85 mm), Over-Shoulder (35 mm)
* **[REQ-04] Semantic-Parameter Mapping** – emotion keywords (`angry`, `tense`, …) are mapped to concrete image parameters (`contrast`, `saturation`, camera angle)

## Architecture

Pipe-and-Filter pipeline:

```
[Parser Filter]  →  [Director Filter]  →  [Geometry Filter]  →  [Formatter Filter]
script_analyzer     director_logic         spatial_solver          prompt_builder
```

## Installation

```bash
pip install -e ".[dev]"
```

Optionally add your OpenAI API key for LLM-backed script analysis:

```bash
export OPENAI_API_KEY=sk-...
```

## Quick Start

```python
from autocineflow.pipeline import CineFlowPipeline

pipeline = CineFlowPipeline()

# Use use_llm=False for fully offline / rule-based analysis
ctx = pipeline.run(
    description="Two people sit facing each other in a tavern, tension rising.",
    num_shots=5,
    use_llm=False,          # set True to call GPT-4o
    emotion_override="tense",
)

# Serialise the full scene to JSON
print(pipeline.to_json(ctx, indent=2))

# Get ControlNet-compatible coordinates for the final shot
print(pipeline.controlnet_coords(ctx))

# Verify acceptance criteria
assert pipeline.validate_axis_consistency(ctx)          # axis_side constant across shots
assert pipeline.validate_visual_anchor_consistency(ctx) # character descriptions identical
```

## Output Schema

Each `ShotBlock` in the JSON contains:

| Field | Description |
|-------|-------------|
| `framing` | `shot_type`, `focal_length_mm`, `subjects` |
| `camera_angle` | `angle_type`, `axis_side`, `tilt_degrees` |
| `lighting` | `contrast`, `saturation`, `mood` |
| `motion_instruction` | `motion_type`, `intensity` |
| `characters` | `char_id`, `visual_anchor`, `pos` (x/y ∈ [0,1]), `facing` |
| `sd_prompt` | Ready-to-use Stable Diffusion positive prompt |
| `negative_prompt` | SD negative prompt |

## Running Tests

```bash
pytest
```

All 83 tests cover:

* Pydantic model validation and emotion matrix values
* 180° axis cross-product geometry
* Director logic (shot selection, axis enforcement, emotion mapping)
* Prompt builder token generation
* End-to-end pipeline acceptance criteria (axis consistency, gaze logic, data completeness, JSON output)
