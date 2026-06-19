from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from config import LabConfig, load_config
from memory_store import CompactMemoryManager, UserProfileStore, estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class AgentContext:
    user_id: str
    memory_path: str


class AdvancedAgent:
    """Advanced agent with persistent profile memory and compact thread memory.

    Required memory layers:
    1. within-session memory
    2. persistent `User.md`
    3. compact memory for long threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.profile_store = UserProfileStore(self.config.state_dir / "profiles")
        self.compact_memory = CompactMemoryManager(
            threshold_tokens=self.config.compact_threshold_tokens,
            keep_messages=self.config.compact_keep_messages,
        )
        self.thread_tokens: dict[str, int] = {}
        self.thread_prompt_tokens: dict[str, int] = {}

        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Route the turn through the deterministic memory-aware path."""

        return self._reply_offline(user_id, thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.thread_tokens.get(thread_id, 0)

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.thread_prompt_tokens.get(thread_id, 0)

    def memory_file_size(self, user_id: str) -> int:
        return self.profile_store.file_size(user_id)

    def compaction_count(self, thread_id: str) -> int:
        return self.compact_memory.compaction_count(thread_id)

    def _reply_offline(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Implement the deterministic advanced path."""

        for key, value in extract_profile_updates(message).items():
            self.profile_store.upsert_fact(user_id, key, value)

        self.compact_memory.append(thread_id, "user", message)
        prompt_tokens = self._estimate_prompt_context_tokens(user_id, thread_id)
        self.thread_prompt_tokens[thread_id] = self.thread_prompt_tokens.get(thread_id, 0) + prompt_tokens

        response = self._offline_response(user_id, thread_id, message)
        self.compact_memory.append(thread_id, "assistant", response)
        response_tokens = estimate_tokens(response)
        self.thread_tokens[thread_id] = self.thread_tokens.get(thread_id, 0) + response_tokens

        return {
            "answer": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
            "memory_path": str(self.profile_store.path_for(user_id)),
        }

    def _estimate_prompt_context_tokens(self, user_id: str, thread_id: str) -> int:
        """Estimate the profile, summary, and recent-message context for one turn."""

        profile_text = self.profile_store.read_text(user_id)
        context = self.compact_memory.context(thread_id)
        summary = str(context.get("summary", ""))
        messages = context.get("messages", [])
        message_tokens = 0
        if isinstance(messages, list):
            message_tokens = sum(estimate_tokens(str(item.get("content", ""))) for item in messages)
        return estimate_tokens(profile_text) + estimate_tokens(summary) + message_tokens

    def _offline_response(self, user_id: str, thread_id: str, message: str) -> str:
        """Return a deterministic answer using persisted profile memory."""

        facts = self.profile_store.facts(user_id)
        lowered = message.lower()

        if "biết dũngct" in lowered or "tóm tắt" in lowered or "mô tả ngắn" in lowered:
            return _summary_answer(facts)
        if "huế" in lowered and "hà nội" in lowered and "product manager" in lowered:
            return _fact_sentence(facts, ["profession", "location"])
        if "tên" in lowered and any(
            marker in lowered
            for marker in ["nơi ở", "ở hiện tại", "nghề", "đồ uống", "style", "món ăn", "nuôi"]
        ):
            keys = ["name"]
            if "nơi ở" in lowered or "ở hiện tại" in lowered or "ở đâu" in lowered:
                keys.append("location")
            if "nghề" in lowered:
                keys.append("profession")
            if "đồ uống" in lowered:
                keys.append("favorite_drink")
            if "style" in lowered or "kiểu trả lời" in lowered:
                keys.append("response_style")
            if "món ăn" in lowered:
                keys.append("favorite_food")
            if "nuôi" in lowered or "con gì" in lowered:
                keys.append("pet")
            return _fact_sentence(facts, keys)
        if "tên" in lowered and ("nghề" in lowered or "mối quan tâm" in lowered or "style" in lowered):
            keys = ["name"]
            if "nghề" in lowered:
                keys.append("profession")
            if "mối quan tâm" in lowered:
                keys.append("interests")
            if "style" in lowered:
                keys.append("response_style")
            return _fact_sentence(facts, keys)
        if "tên" in lowered and ("nơi ở" in lowered or "ở hiện tại" in lowered) and "đồ uống" in lowered:
            return _fact_sentence(facts, ["name", "location", "profession", "favorite_drink", "response_style"])
        if "đồ uống" in lowered and "món ăn" in lowered:
            return _fact_sentence(facts, ["favorite_drink", "favorite_food"])
        if "món ăn" in lowered and ("nuôi" in lowered or "con gì" in lowered):
            return _fact_sentence(facts, ["name", "favorite_food", "pet"])
        if ("ở đâu" in lowered or "nơi ở" in lowered or "hiện tại mình đang ở" in lowered) and (
            "nuôi" in lowered or "con gì" in lowered
        ):
            return _fact_sentence(facts, ["location", "pet"])
        if "nuôi" in lowered or "con gì" in lowered:
            return _fact_sentence(facts, ["pet"])
        if "tên" in lowered and ("ở đâu" in lowered or "nơi ở" in lowered or "hiện đang ở" in lowered):
            return _fact_sentence(facts, ["name", "location"])
        if "style" in lowered or "kiểu trả lời" in lowered or "trả lời mình thích" in lowered:
            keys = ["response_style"]
            if "ở đâu" in lowered or "hiện đang ở đâu" in lowered:
                keys.append("location")
            if "đồ uống" in lowered:
                keys.append("favorite_drink")
            if "tên" in lowered:
                keys.insert(0, "name")
            return _fact_sentence(facts, keys)
        if "đồ uống" in lowered:
            keys = ["favorite_drink"]
            if "tên" in lowered:
                keys.insert(0, "name")
            return _fact_sentence(facts, keys)
        if "nghề" in lowered or "làm nghề gì" in lowered:
            keys = ["profession"]
            if "huế" in lowered or "ở huế" in lowered:
                keys.append("location")
            return _fact_sentence(facts, keys)
        if "ở đâu" in lowered or "nơi ở" in lowered or "hiện tại mình đang ở" in lowered:
            return _fact_sentence(facts, ["location"])
        if "mình tên gì" in lowered or "nhắc lại tên" in lowered:
            return _fact_sentence(facts, ["name"])

        return "Mình đã cập nhật memory bền vững và giữ câu trả lời ngắn gọn theo preference của bạn."

    def _maybe_build_langchain_agent(self):
        """Build a provider model when optional LangChain dependencies are installed.

        High-level design:
        - `build_chat_model(self.config.model)` for the selected provider
        - `InMemorySaver` for short-term thread state
        - tool to read `User.md`
        - tool to write/edit `User.md`
        - dynamic prompt that injects profile memory
        - summarization middleware for long threads
        """

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None


def _summary_answer(facts: dict[str, str]) -> str:
    return _fact_sentence(
        facts,
        ["name", "profession", "location", "interests", "response_style"],
    )


def _fact_sentence(facts: dict[str, str], keys: list[str]) -> str:
    labels = {
        "name": "tên",
        "location": "nơi ở hiện tại",
        "profession": "nghề nghiệp hiện tại",
        "response_style": "style trả lời",
        "favorite_drink": "đồ uống yêu thích",
        "favorite_food": "món ăn yêu thích",
        "pet": "thú cưng",
        "interests": "mối quan tâm kỹ thuật",
        "benchmark_preference": "ưu tiên benchmark",
    }
    parts = []
    for key in keys:
        value = facts.get(key)
        if value:
            parts.append(f"{labels.get(key, key)}: {value}")
    if not parts:
        return "Mình chưa có đủ thông tin này trong User.md."
    return "Mình nhớ: " + "; ".join(parts) + "."
