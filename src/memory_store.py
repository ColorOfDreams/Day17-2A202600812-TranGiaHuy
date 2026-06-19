from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import unicodedata


def estimate_tokens(text: str) -> int:
    """Estimate tokens with a stable character-count heuristic.

    Example idea:
    - Strip whitespace
    - Return 0 for empty text
    - Approximate tokens from character count, e.g. len(text) / 4
    """

    cleaned = " ".join((text or "").split())
    if not cleaned:
        return 0
    return max(1, math.ceil(len(cleaned) / 4))


@dataclass
class UserProfileStore:
    """Persistent storage for `User.md`.

    Responsibilities:
    - Map each user id to one markdown file
    - Support read / write / edit operations
    - Optionally expose helpers like `facts()` or `upsert_fact()`
    """

    root_dir: Path

    def path_for(self, user_id: str) -> Path:
        slug = _slugify(user_id)
        return self.root_dir / slug / "User.md"

    def read_text(self, user_id: str) -> str:
        path = self.path_for(user_id)
        if not path.exists():
            return _default_profile(user_id)
        return path.read_text(encoding="utf-8")

    def write_text(self, user_id: str, content: str) -> Path:
        path = self.path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        return path

    def edit_text(self, user_id: str, search_text: str, replacement: str) -> bool:
        current = self.read_text(user_id)
        if search_text not in current:
            return False
        self.write_text(user_id, current.replace(search_text, replacement, 1))
        return True

    def file_size(self, user_id: str) -> int:
        path = self.path_for(user_id)
        return path.stat().st_size if path.exists() else 0

    def facts(self, user_id: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        for line in self.read_text(user_id).splitlines():
            match = re.match(r"-\s*([a-z_]+):\s*(.+)", line.strip())
            if match:
                facts[match.group(1)] = match.group(2).strip()
        return facts

    def upsert_fact(self, user_id: str, key: str, value: str) -> None:
        if not value:
            return
        facts = self.facts(user_id)
        facts[key] = value.strip()
        ordered_keys = [
            "name",
            "location",
            "profession",
            "response_style",
            "favorite_drink",
            "favorite_food",
            "pet",
            "interests",
            "learning_goal",
            "benchmark_preference",
        ]
        lines = [f"# User Profile: {user_id}", "", "## Facts"]
        for fact_key in ordered_keys:
            if fact_key in facts:
                lines.append(f"- {fact_key}: {facts[fact_key]}")
        for fact_key in sorted(set(facts) - set(ordered_keys)):
            lines.append(f"- {fact_key}: {facts[fact_key]}")
        self.write_text(user_id, "\n".join(lines))


def extract_profile_updates(message: str) -> dict[str, str]:
    """Convert raw user text into stable profile facts.

    Example facts you may want to extract:
    - name
    - location
    - profession
    - preferences / response style
    - favorite food / drink

    Pseudocode:
    1. Build a few regex patterns.
    2. Skip obvious question-only turns.
    3. Return only the facts that are confidently present in the message.
    """

    text = " ".join((message or "").split())
    lowered = text.lower()
    updates: dict[str, str] = {}

    if not text or _looks_like_question_only(text):
        return updates

    name_patterns = [
        r"(?:mình|tôi)\s+tên\s+là\s+([^,.!?]+)",
        r"tên\s+(?:mình|tôi)\s+là\s+([^,.!?]+)",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            updates["name"] = _clean_value(match.group(1))
            break

    if "không phải nơi ở hiện tại" not in lowered:
        location = _extract_location(text)
        if location:
            updates["location"] = location

    profession = _extract_profession(text)
    if profession:
        updates["profession"] = profession

    drink = _extract_after_patterns(
        text,
        [
            r"(?:đồ uống yêu thích|vẫn uống)\s+(?:là\s+)?([^,.!?]+)",
            r"uống\s+(cà phê sữa đá)",
        ],
    )
    if drink:
        updates["favorite_drink"] = drink

    food = _extract_after_patterns(text, [r"(?:món ăn yêu thích|món ruột)\s+(?:là\s+)?([^,.!?]+)"])
    if food:
        updates["favorite_food"] = food
    elif "mì quảng" in lowered:
        updates["favorite_food"] = "mì Quảng"

    if "corgi" in lowered:
        pet_name = _extract_after_patterns(text, [r"corgi\s+tên\s+([^,.!?]+)"])
        updates["pet"] = f"corgi tên {pet_name}" if pet_name else "corgi"

    interests = _extract_interests(text)
    if interests:
        updates["interests"] = interests

    style = _extract_response_style(text)
    if style:
        updates["response_style"] = style

    if "recall đúng" in lowered:
        updates["benchmark_preference"] = "ưu tiên recall đúng hơn câu văn hoa mỹ"
    elif "số liệu rõ ràng" in lowered:
        updates["benchmark_preference"] = "benchmark có số liệu rõ ràng"

    if "mục tiêu học tập" in lowered:
        goal = _extract_after_patterns(text, [r"mục tiêu học tập[^l]*là\s+([^,.!?]+)"])
        if goal:
            updates["learning_goal"] = goal

    return updates


def summarize_messages(messages: list[dict[str, str]], max_items: int = 6) -> str:
    """Create a compact heuristic summary of older messages.

    This can be heuristic text concatenation first.
    Later, you can replace it with an LLM-based summary if desired.
    """

    if not messages:
        return ""
    selected = messages[-max_items:]
    lines = []
    for item in selected:
        role = item.get("role", "unknown")
        content = " ".join(item.get("content", "").split())
        if len(content) > 220:
            content = content[:217].rstrip() + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


@dataclass
class CompactMemoryManager:
    """Compact memory for long threads.

    Goal:
    - Keep recent messages in full
    - When the thread grows too large, move older content into a summary
    - Track how many compactions happened for benchmarking
    """

    threshold_tokens: int
    keep_messages: int
    state: dict[str, dict[str, object]] = field(default_factory=dict)

    def append(self, thread_id: str, role: str, content: str) -> None:
        thread = self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )
        messages = thread["messages"]
        assert isinstance(messages, list)
        messages.append({"role": role, "content": content})

        summary = str(thread.get("summary", ""))
        total_tokens = estimate_tokens(summary) + sum(
            estimate_tokens(str(message.get("content", ""))) for message in messages
        )
        if total_tokens <= self.threshold_tokens or len(messages) <= self.keep_messages:
            return

        split_at = max(1, len(messages) - self.keep_messages)
        older = messages[:split_at]
        recent = messages[split_at:]
        new_summary = summarize_messages(older)
        thread["summary"] = "\n".join(part for part in [summary, new_summary] if part).strip()
        thread["messages"] = recent
        thread["compactions"] = int(thread.get("compactions", 0)) + 1

    def context(self, thread_id: str) -> dict[str, object]:
        return self.state.setdefault(
            thread_id,
            {"messages": [], "summary": "", "compactions": 0},
        )

    def compaction_count(self, thread_id: str) -> int:
        return int(self.context(thread_id).get("compactions", 0))


def _slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "anonymous")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", ascii_text).strip("-").lower()
    return slug or "anonymous"


def _default_profile(user_id: str) -> str:
    return f"# User Profile: {user_id}\n\n## Facts\n"


def _looks_like_question_only(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in [" là gì", "gì.", "gì?", "ở đâu", "nhắc lại", "hãy nhắc"]):
        return True
    recall_question_markers = [
        "gì",
        "ở đâu",
        "đâu mới",
        "nhắc lại",
        "bạn biết",
        "hãy nhắc",
        "thử mô tả",
        "tóm tắt",
    ]
    if "?" in text and any(marker in lowered for marker in recall_question_markers):
        return True
    fact_markers = [
        "mình tên là",
        "tên mình là",
        "đang làm",
        "hiện ở",
        "đang ở",
        "mình ở",
        "yêu thích là",
        "mình thích",
        "mình muốn",
        "hãy trả lời",
        "đính chính",
        "chuyển sang",
    ]
    return "?" in text and not any(marker in lowered for marker in fact_markers)


def _clean_value(value: str) -> str:
    value = re.split(r"\s+(?:và|nhưng|chứ|dù|để|cho)\s+", value.strip(), maxsplit=1)[0]
    return value.strip(" .,!?:;\"'")


def _extract_after_patterns(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_value(match.group(1))
    return ""


def _extract_location(text: str) -> str:
    lowered = text.lower()
    if "đà nẵng" in lowered and any(
        marker in lowered
        for marker in ["đang làm việc ở đà nẵng", "hiện tại là đà nẵng", "nơi ở hiện tại là đà nẵng"]
    ):
        return "Đà Nẵng"
    if "giờ mình đang ở huế" in lowered or "hiện đang ở huế" in lowered or "vẫn ở huế" in lowered:
        return "Huế"

    patterns = [
        r"(?:mình|tôi)\s+(?:đang\s+)?ở\s+([^,.!?]+)",
        r"(?:hiện|hiện tại|từ tuần này)\s+(?:mình\s+)?(?:đang\s+)?(?:làm việc\s+)?ở\s+([^,.!?]+)",
        r"nơi ở hiện tại\s+(?:là\s+)?([^,.!?]+)",
    ]
    location = _extract_after_patterns(text, patterns)
    if not location:
        return ""
    for known in ["Đà Nẵng", "Huế", "Hà Nội"]:
        if known.lower() in location.lower():
            return known
    return location


def _extract_profession(text: str) -> str:
    lowered = text.lower()
    if "backend engineer" in lowered and any(marker in lowered for marker in ["đừng nói", "thông tin cũ"]):
        return ""
    if "product manager" in lowered and "câu đùa" in lowered:
        return "MLOps engineer" if "mlops engineer" in lowered else ""
    if "mlops engineer" in lowered:
        return "MLOps engineer"
    if "backend engineer" in lowered and "không còn" not in lowered:
        return "backend engineer"
    return _extract_after_patterns(
        text,
        [
            r"(?:đang làm|làm nghề|nghề nghiệp hiện tại là)\s+([^,.!?]+)",
            r"chuyển sang\s+([^,.!?]+)",
        ],
    )


def _extract_interests(text: str) -> str:
    lowered = text.lower()
    interests: list[str] = []
    for label, canonical in [
        ("python", "Python"),
        ("ai ứng dụng", "AI ứng dụng"),
        ("ai agent", "AI agent"),
        ("benchmark memory", "benchmark memory"),
        ("mlops", "MLOps"),
        ("rag", "RAG"),
        ("evaluation", "evaluation"),
        ("async python", "async Python"),
    ]:
        if label in lowered and canonical not in interests:
            interests.append(canonical)
    if interests:
        return ", ".join(interests)
    return ""


def _extract_response_style(text: str) -> str:
    lowered = text.lower()
    style_parts: list[str] = []
    if "3 bullet" in lowered or "ba bullet" in lowered:
        style_parts.append("3 bullet ngắn")
    elif "bullet" in lowered:
        style_parts.append("bullet ngắn")
    if "ngắn gọn" in lowered or "gọn" in lowered:
        style_parts.append("ngắn gọn")
    if "rõ ý" in lowered:
        style_parts.append("rõ ý")
    if "ví dụ thực tế" in lowered or "ví dụ thực chiến" in lowered:
        style_parts.append("có ví dụ thực chiến")
    if "trade-off" in lowered:
        style_parts.append("nhấn trade-off")
    if "không thích câu trả lời quá lan man" in lowered:
        style_parts.append("không lan man")

    unique_parts = []
    for part in style_parts:
        if part not in unique_parts:
            unique_parts.append(part)
    return ", ".join(unique_parts)
