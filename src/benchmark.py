from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import tempfile
from typing import Any

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import load_config


@dataclass
class BenchmarkRow:
    agent_name: str
    agent_tokens_only: int
    prompt_tokens_processed: int
    recall_score: float
    response_quality: float
    memory_growth_bytes: int
    compactions: int


def load_conversations(path: Path) -> list[dict[str, Any]]:
    """Read JSON conversations from disk."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of conversations in {path}")
    return data


def recall_points(answer: str, expected: list[str]) -> float:
    """Return 0 / 0.5 / 1 depending on how many expected facts appear."""

    if not expected:
        return 1.0
    normalized_answer = answer.casefold()
    hits = sum(1 for item in expected if item.casefold() in normalized_answer)
    ratio = hits / len(expected)
    if ratio >= 0.99:
        return 1.0
    if ratio > 0:
        return 0.5
    return 0.0


def heuristic_quality(answer: str, expected: list[str]) -> float:
    """Compute a lightweight quality score for offline mode."""

    if not answer.strip():
        return 0.0
    recall = recall_points(answer, expected)
    concise_bonus = 1.0 if len(answer) <= 360 else 0.75
    structure_bonus = 1.0 if any(mark in answer for mark in [":", ";", "-", "\n"]) else 0.85
    return round((recall * 0.7) + (concise_bonus * 0.15) + (structure_bonus * 0.15), 3)


def run_agent_benchmark(agent_name: str, agent, conversations: list[dict[str, Any]], config) -> BenchmarkRow:
    """Evaluate one agent over many conversations.

    Pseudocode:
    1. Feed all turns to the agent.
    2. Track `agent tokens only`.
    3. Track `prompt tokens processed`.
    4. Ask recall questions in a fresh thread.
    5. Compute average recall and quality.
    6. Record memory file growth and compaction count.
    """

    user_ids: set[str] = {conversation["user_id"] for conversation in conversations}
    thread_ids: list[str] = []
    memory_before = sum(agent.memory_file_size(user_id) for user_id in user_ids) if hasattr(agent, "memory_file_size") else 0
    memory_after = 0
    recall_scores: list[float] = []
    quality_scores: list[float] = []

    for conversation in conversations:
        user_id = conversation["user_id"]
        thread_id = conversation["id"]
        thread_ids.append(thread_id)

        for turn in conversation.get("turns", []):
            agent.reply(user_id, thread_id, turn)

        for index, recall in enumerate(conversation.get("recall_questions", []), start=1):
            recall_thread = f"{thread_id}-recall-{index}"
            thread_ids.append(recall_thread)
            result = agent.reply(user_id, recall_thread, recall["question"])
            answer = result["answer"]
            expected = recall.get("expected_contains", [])
            recall_scores.append(recall_points(answer, expected))
            quality_scores.append(heuristic_quality(answer, expected))

    if hasattr(agent, "memory_file_size"):
        memory_after = sum(agent.memory_file_size(user_id) for user_id in user_ids)

    agent_tokens = sum(agent.token_usage(thread_id) for thread_id in set(thread_ids))
    prompt_tokens = sum(agent.prompt_token_usage(thread_id) for thread_id in set(thread_ids))
    compactions = sum(agent.compaction_count(thread_id) for thread_id in set(thread_ids))

    return BenchmarkRow(
        agent_name=agent_name,
        agent_tokens_only=agent_tokens,
        prompt_tokens_processed=prompt_tokens,
        recall_score=round(sum(recall_scores) / max(1, len(recall_scores)), 3),
        response_quality=round(sum(quality_scores) / max(1, len(quality_scores)), 3),
        memory_growth_bytes=max(0, memory_after - memory_before),
        compactions=compactions,
    )


def format_rows(rows: list[BenchmarkRow]) -> str:
    """Format benchmark rows as a markdown table."""

    headers = [
        "Agent",
        "Agent tokens only",
        "Prompt tokens processed",
        "Cross-session recall",
        "Response quality",
        "Memory growth (bytes)",
        "Compactions",
    ]
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        table.append(
            "| "
            + " | ".join(
                [
                    row.agent_name,
                    str(row.agent_tokens_only),
                    str(row.prompt_tokens_processed),
                    f"{row.recall_score:.3f}",
                    f"{row.response_quality:.3f}",
                    str(row.memory_growth_bytes),
                    str(row.compactions),
                ]
            )
            + " |"
        )
    return "\n".join(table)


def main() -> None:
    """Run both benchmark suites.

    Required benchmark sections:
    - Standard benchmark from `data/conversations.json`
    - Long-context stress benchmark from `data/advanced_long_context.json`

    Compare:
    - Baseline
    - Advanced

    Keep the same output columns as the solved lab:
    - Agent tokens only
    - Prompt tokens processed
    - Cross-session recall
    - Response quality
    - Memory growth (bytes)
    - Compactions
    """

    base_config = load_config(Path(__file__).resolve().parent.parent)

    with tempfile.TemporaryDirectory(prefix="day17_memory_benchmark_") as state_dir:
        config = replace(base_config, state_dir=Path(state_dir))
        standard = load_conversations(config.data_dir / "conversations.json")
        stress = load_conversations(config.data_dir / "advanced_long_context.json")

        print("# Standard benchmark")
        standard_rows = [
            run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), standard, config),
            run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), standard, config),
        ]
        print(format_rows(standard_rows))

        print("\n# Long-context stress benchmark")
        stress_rows = [
            run_agent_benchmark("Baseline", BaselineAgent(config, force_offline=True), stress, config),
            run_agent_benchmark("Advanced", AdvancedAgent(config, force_offline=True), stress, config),
        ]
        print(format_rows(stress_rows))


if __name__ == "__main__":
    main()
