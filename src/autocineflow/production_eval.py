"""CLI for running production-readiness evaluations against real or offline analyzers."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

from .pipeline import CineFlowPipeline


@dataclass(frozen=True)
class EvalScenario:
    """Named scene prompt for production evaluation."""

    scene_id: str
    description: str
    num_shots: int = 5


DEFAULT_SCENARIOS = [
    EvalScenario(
        scene_id="EN_TAVERN",
        description=(
            "A man in black coat and a woman in red dress sit across a tavern table. "
            "He suddenly slams the glass down and erupts in anger."
        ),
    ),
    EvalScenario(
        scene_id="ZH_TAVERN",
        description="穿黑色大衣的男人与红裙女人在酒馆对坐，空气越来越紧张，男人突然愤怒地拍桌。",
    ),
    EvalScenario(
        scene_id="DIALOGUE_NOIR",
        description=(
            'A detective in a rain-soaked trench coat faces a wounded informant in a neon alley. '
            '"Tell me who sent you," the detective says.'
        ),
    ),
    EvalScenario(
        scene_id="ROMANCE_BAR",
        description=(
            "A young woman in a cream coat and a tired musician sit under candlelight in a quiet bar, "
            "speaking softly as the room fades away around them."
        ),
    ),
]


def run_evaluation(
    config_path: str | None = None,
    model: str = "gpt-5",
    use_llm: bool = True,
) -> list[dict]:
    """Run the default production scenarios and return structured results."""

    pipeline = CineFlowPipeline(config_path=config_path, model=model)
    results: list[dict] = []

    for scenario in DEFAULT_SCENARIOS:
        context = pipeline.run(
            description=scenario.description,
            num_shots=scenario.num_shots,
            scene_id=scenario.scene_id,
            use_llm=use_llm,
        )
        readiness = pipeline.production_readiness_report(context)
        results.append(
            {
                "scene_id": scenario.scene_id,
                "analysis_source": context.analysis_source,
                "emotion": context.detected_emotion,
                "scene_location": context.scene_location,
                "characters": [char.visual_anchor for char in context.characters],
                "shot_types": [shot.framing.shot_type.value for shot in context.shot_blocks],
                "readiness": readiness,
                "ready": all(readiness.values()),
            }
        )

    return results


def main() -> int:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description="Run Auto-CineFlow production-readiness scenarios.")
    parser.add_argument("--config-path", default=None, help="Path to OpenAI-compatible config file")
    parser.add_argument("--model", default="gpt-5", help="LLM model name")
    parser.add_argument("--offline", action="store_true", help="Use rule-based analysis only")
    parser.add_argument(
        "--fail-on-readiness",
        action="store_true",
        help="Exit non-zero if any scenario is not production-ready",
    )
    args = parser.parse_args()

    results = run_evaluation(
        config_path=args.config_path,
        model=args.model,
        use_llm=not args.offline,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))

    if args.fail_on_readiness and not all(item["ready"] for item in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
