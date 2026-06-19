from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig
from memory_store import UserProfileStore
from model_provider import ProviderConfig


def make_config(tmp_path: Path):
    """Build an isolated config for tests."""

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    provider = ProviderConfig(provider="openai", model_name="offline-test", temperature=0)
    return LabConfig(
        base_dir=tmp_path,
        data_dir=tmp_path / "data",
        state_dir=state_dir,
        compact_threshold_tokens=70,
        compact_keep_messages=3,
        model=provider,
        judge_model=provider,
    )


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    assert "## Facts" in store.read_text("Dũng CT")

    path = store.write_text("Dũng CT", "# User Profile\n\n## Facts\n- name: DũngCT")
    assert path.exists()
    assert "DũngCT" in store.read_text("Dũng CT")

    changed = store.edit_text("Dũng CT", "DũngCT", "DũngCT Updated")
    assert changed is True
    assert "DũngCT Updated" in store.read_text("Dũng CT")
    assert store.file_size("Dũng CT") > 0


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction."""

    config = make_config(tmp_path)
    agent = AdvancedAgent(config, force_offline=True)
    for index in range(8):
        agent.reply(
            "user-1",
            "thread-long",
            f"Lượt {index}: mình đang ghi một đoạn rất dài về memory compaction "
            "để vượt ngưỡng token và buộc hệ thống phải nén lịch sử cũ.",
        )

    assert agent.compaction_count("thread-long") > 0


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify advanced remembers across sessions and baseline does not."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    fact = "Chào bạn, mình tên là DũngCT. Mình đang ở Huế và đang làm MLOps engineer."
    baseline.reply("user-1", "session-a", fact)
    advanced.reply("user-1", "session-a", fact)

    baseline_answer = baseline.reply("user-1", "session-b", "Mình tên gì và hiện đang ở đâu?")["answer"]
    advanced_answer = advanced.reply("user-1", "session-b", "Mình tên gì và hiện đang ở đâu?")["answer"]

    assert "DũngCT" not in baseline_answer
    assert "DũngCT" in advanced_answer
    assert "Huế" in advanced_answer


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare prompt load of baseline vs advanced on a long thread."""

    config = make_config(tmp_path)
    baseline = BaselineAgent(config, force_offline=True)
    advanced = AdvancedAgent(config, force_offline=True)

    for index in range(12):
        message = (
            f"Lượt dài {index}: mình tên là DũngCT và đang làm MLOps engineer. "
            "Đoạn này cố tình dài để baseline phải kéo toàn bộ lịch sử qua mỗi lượt, "
            "trong khi advanced nên compact phần cũ thành summary ngắn hơn. " * 3
        )
        baseline.reply("user-1", "thread-load", message)
        advanced.reply("user-1", "thread-load", message)

    assert advanced.compaction_count("thread-load") > 0
    assert advanced.prompt_token_usage("thread-load") < baseline.prompt_token_usage("thread-load")
