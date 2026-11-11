"""Tests of the chat store on a temporary folder."""

import pytest

from agent.stores import chat_store

pytestmark = pytest.mark.integration


class TestChatStore:
    """The store keeps state and history of the chats on JSON files."""

    def test_new_chat_has_empty_state(self, tmp_path):
        """Test: state of a chat never stored.
        Expected: empty ChatState.
        """
        # Act
        state = chat_store.ChatStore(tmp_path).get_state("unknown")

        # Assert
        assert (state.phase, state.active_scope) == (None, None)

    def test_scope_changes_are_kept_in_history(self, tmp_path):
        """Test: scopes chosen by the triage, then changed after a rerouting.
        Expected: the last scopes are active, both changes are in active_scope_history.
        """
        # Arrange
        store = chat_store.ChatStore(tmp_path)
        store.init_chat(chat_id="chat-1", user="user-1")
        store.update_active_scope("chat-1", scope_ids=["novadrive"], source="triage")

        # Act
        store.update_active_scope("chat-1", scope_ids=["novabill"], source="triage")

        # Assert
        record = store.get("chat-1")
        assert record.active_scope.scope_ids == ["novabill"]
        assert [item.scope_ids for item in record.active_scope_history] == [
            ["novadrive"],
            ["novabill"],
        ]

    def test_chat_id_with_path_separator_is_rejected(self, tmp_path):
        """Test: chat id that would escape the chats folder.
        Expected: ValueError.
        """
        # Act / Assert
        with pytest.raises(ValueError, match="path separators"):
            chat_store.ChatStore(tmp_path).init_chat(chat_id="../evil", user="user-1")
