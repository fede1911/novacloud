"""Chat histories on JSON files, one per chat: swap this store for a database in production."""

import datetime as dt
import pathlib
import threading

from agent.models import chat as chat_models
from shared import logger

_LOGGER = logger.get_logger(__name__)


class ChatNotFoundError(LookupError):
    """Raised when a chat has no stored history."""


class ChatStore:
    """Reads and updates the chats stored as <chats_dir>/<chat_id>.json.

    Every update reads, changes and rewrites the whole record under a lock:
    good enough for a local demo with one process.
    """

    def __init__(self, chats_dir: pathlib.Path):
        """Initializes the store.

        Args:
            chats_dir: Folder of the chat files, created if missing.
        """
        self._chats_dir = chats_dir
        self._chats_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def init_chat(self, *, chat_id: str, user: str) -> chat_models.ChatRecord:
        """Creates the chat if it does not exist and returns it."""
        with self._lock:
            path = self._path(chat_id)
            if path.exists():
                return self._read(chat_id)
            record = chat_models.ChatRecord(
                chat_id=chat_id, user=user, created_at=_now()
            )
            self._write(record)
            _LOGGER.info("Chat %s created", chat_id)
            return record

    def get(self, chat_id: str) -> chat_models.ChatRecord:
        """Returns the stored chat.

        Raises:
            ChatNotFoundError: If the chat does not exist.
        """
        with self._lock:
            return self._read(chat_id)

    def get_state(self, chat_id: str) -> chat_models.ChatState:
        """Returns the state read by the orchestrator, empty for a new chat."""
        with self._lock:
            if not self._path(chat_id).exists():
                return chat_models.ChatState()
            return self._read(chat_id).to_state()

    def update_phase(self, chat_id: str, phase: chat_models.AgentType) -> None:
        """Stores the agent that handles the next user message."""
        self._update(chat_id, lambda record: setattr(record, "phase", phase))

    def update_active_scope(
        self, chat_id: str, *, scope_ids: list[str], source: chat_models.ScopeSource
    ) -> None:
        """Stores the scopes of the chat, keeping every change in active_scope_history."""

        def apply(record: chat_models.ChatRecord) -> None:
            active_scope = chat_models.ActiveScope(
                scope_ids=scope_ids, source=source, updated_at=_now()
            )
            record.active_scope = active_scope
            record.active_scope_history.append(active_scope)

        self._update(chat_id, apply)

    def add_message(self, chat_id: str, item: chat_models.HistoryItem) -> None:
        """Appends a message to the history."""
        self._update(chat_id, lambda record: record.history.append(item))

    def record_models(self, chat_id: str, models_used: chat_models.ModelsUsed) -> None:
        """Records the models of an agent, if they changed since the last record of the same agent type."""

        def apply(record: chat_models.ChatRecord) -> None:
            previous = [
                item
                for item in record.models_used
                if item.agent_type == models_used.agent_type
            ]
            if not previous or previous[-1] != models_used:
                record.models_used.append(models_used)

        self._update(chat_id, apply)

    def record_prompts(self, chat_id: str, prompt_names: list[str]) -> None:
        """Records the names of the prompts used by the chat, without duplicates."""

        def apply(record: chat_models.ChatRecord) -> None:
            record.prompts.extend(
                name for name in prompt_names if name not in record.prompts
            )

        self._update(chat_id, apply)

    def _update(self, chat_id: str, apply) -> None:
        """Reads the chat, applies a change and writes it back under the lock."""
        with self._lock:
            record = self._read(chat_id)
            apply(record)
            self._write(record)

    def _read(self, chat_id: str) -> chat_models.ChatRecord:
        path = self._path(chat_id)
        try:
            return chat_models.ChatRecord.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except FileNotFoundError as e:
            raise ChatNotFoundError(f"Chat '{chat_id}' not found") from e

    def _write(self, record: chat_models.ChatRecord) -> None:
        self._path(record.chat_id).write_text(
            record.model_dump_json(indent=2), encoding="utf-8"
        )

    def _path(self, chat_id: str) -> pathlib.Path:
        if pathlib.Path(chat_id).name != chat_id:
            raise ValueError("chat_id cannot contain path separators")
        return self._chats_dir / f"{chat_id}.json"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
