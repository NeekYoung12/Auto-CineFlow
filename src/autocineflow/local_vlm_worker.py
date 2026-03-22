"""Standalone worker executed by an external Python env for local visual review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _extract_json_block(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:500]}")
    return json.loads(text[start:end + 1])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Qwen-VL visual review on one image.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--device-preference", default="cuda")
    parser.add_argument("--min-free-vram-gb", type=float, default=4.0)
    args = parser.parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(json.dumps({"status": "error", "reason": "missing_image", "image_path": str(image_path)}))
        return 1

    try:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "reason": "missing_runtime", "detail": str(exc)}))
        return 1

    if args.device_preference == "cuda":
        if not torch.cuda.is_available():
            print(json.dumps({"status": "skipped", "reason": "cuda_unavailable"}))
            return 0
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
        if free_gb < args.min_free_vram_gb:
            print(
                json.dumps(
                    {
                        "status": "skipped",
                        "reason": "gpu_busy",
                        "free_vram_gb": round(free_gb, 2),
                        "required_vram_gb": args.min_free_vram_gb,
                    }
                )
            )
            return 0

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {
                    "type": "text",
                    "text": (
                        "You are a strict cinematic keyframe QA reviewer. "
                        "Evaluate this frame for: text artifacts or fake signage, face distortion, hand or anatomy errors, "
                        "blur/softness, wardrobe drift, and scene coherence. "
                        "Return ONLY JSON with this schema: "
                        '{"text_artifact": bool, "face_distortion": bool, "anatomy_issue": bool, '
                        '"blur_issue": bool, "wardrobe_drift": bool, "scene_incoherence": bool, '
                        '"score": float, "recommendation": "approve|repair|rerender", "notes": [str]}. '
                        "The score must be between 0.0 and 1.0."
                    ),
                },
            ],
        }
    ]

    try:
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            dtype="auto",
            device_map="auto" if args.device_preference == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        if hasattr(model, "device"):
            inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        payload = _extract_json_block(output_text)
        print(json.dumps({"status": "ok", "result": payload}, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "reason": "inference_failed", "detail": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
