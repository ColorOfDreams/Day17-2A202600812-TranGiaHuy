from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config import LabConfig, load_config
from memory_store import estimate_tokens, extract_profile_updates
from model_provider import build_chat_model


@dataclass
class SessionState:
    messages: list[dict[str, str]] = field(default_factory=list)
    token_usage: int = 0
    prompt_tokens_processed: int = 0


class BaselineAgent:
    """Baseline agent with within-session memory only.

    Requirements:
    - Within-session memory only
    - No persistent `User.md`
    - Should forget long-term facts across new threads
    """

    def __init__(self, config: LabConfig | None = None, force_offline: bool = False) -> None:
        self.config = config or load_config()
        self.force_offline = force_offline
        self.sessions: dict[str, SessionState] = {}

        self.langchain_agent = None

    def reply(self, user_id: str, thread_id: str, message: str) -> dict[str, Any]:
        """Return a deterministic response and token accounting."""

        return self._reply_offline(thread_id, message)

    def token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).token_usage

    def prompt_token_usage(self, thread_id: str) -> int:
        return self.sessions.get(thread_id, SessionState()).prompt_tokens_processed

    def compaction_count(self, thread_id: str) -> int:
        # Baseline has no compact memory.
        return 0

    def _reply_offline(self, thread_id: str, message: str) -> dict[str, Any]:
        """Generate a simple offline response without cross-thread memory."""

        session = self.sessions.setdefault(thread_id, SessionState())
        session.messages.append({"role": "user", "content": message})

        prompt_tokens = sum(estimate_tokens(item["content"]) for item in session.messages)
        session.prompt_tokens_processed += prompt_tokens

        facts = _facts_from_messages(session.messages)
        response = _baseline_response(message, facts)

        session.messages.append({"role": "assistant", "content": response})
        response_tokens = estimate_tokens(response)
        session.token_usage += response_tokens

        return {
            "answer": response,
            "agent_tokens": response_tokens,
            "prompt_tokens": prompt_tokens,
            "thread_id": thread_id,
        }

    def _maybe_build_langchain_agent(self):
        """Build a provider model when optional LangChain dependencies are installed.

        Use `build_chat_model(self.config.model)` so the baseline can run with any supported provider.
        """

        try:
            return build_chat_model(self.config.model)
        except Exception:
            return None


def _facts_from_messages(messages: list[dict[str, str]]) -> dict[str, str]:
    facts: dict[str, str] = {}
    for item in messages:
        if item.get("role") != "user":
            continue
        facts.update(extract_profile_updates(item.get("content", "")))
    return facts


def _baseline_response(message: str, facts: dict[str, str]) -> str:
    lowered = message.lower()
    if any(marker in lowered for marker in ["mình tên gì", "tên mình", "tên,", "nhắc lại tên"]):
        name = facts.get("name")
        return f"Trong thread này, mình thấy bạn tên là {name}." if name else "Mình chưa có thông tin tên trong thread này."
    if "nghề" in lowered or "làm gì" in lowered:
        profession = facts.get("profession")
        return f"Trong thread này, nghề hiện tại của bạn là {profession}." if profession else "Mình chưa thấy nghề nghiệp trong thread này."
    if "ở đâu" in lowered or "nơi ở" in lowered:
        location = facts.get("location")
        return f"Trong thread này, nơi ở hiện tại của bạn là {location}." if location else "Mình chưa thấy nơi ở trong thread này."
    if "đồ uống" in lowered:
        drink = facts.get("favorite_drink")
        return f"Trong thread này, đồ uống yêu thích của bạn là {drink}." if drink else "Mình chưa thấy đồ uống yêu thích trong thread này."
    if "món ăn" in lowered:
        food = facts.get("favorite_food")
        return f"Trong thread này, món ăn yêu thích của bạn là {food}." if food else "Mình chưa thấy món ăn yêu thích trong thread này."
    if "style" in lowered or "kiểu trả lời" in lowered:
        style = facts.get("response_style")
        return f"Trong thread này, bạn thích câu trả lời {style}." if style else "Mình chưa thấy style trả lời trong thread này."
    return "Mình đã ghi nhận trong phạm vi thread hiện tại."
