"""Tests for SessionManager module"""
import pytest
from session_manager import SessionManager, Message


class TestSessionCreation:
    """Tests for SessionManager.create_session"""

    def test_create_session_returns_id(self):
        """Test that create_session returns a session ID"""
        manager = SessionManager()
        session_id = manager.create_session()

        assert session_id is not None
        assert session_id.startswith("session_")

    def test_create_multiple_sessions(self):
        """Test creating multiple unique sessions"""
        manager = SessionManager()

        session1 = manager.create_session()
        session2 = manager.create_session()
        session3 = manager.create_session()

        assert session1 != session2
        assert session2 != session3
        assert session1 == "session_1"
        assert session2 == "session_2"
        assert session3 == "session_3"

    def test_create_session_initializes_empty_history(self):
        """Test that new session has empty history"""
        manager = SessionManager()
        session_id = manager.create_session()

        assert session_id in manager.sessions
        assert manager.sessions[session_id] == []


class TestAddMessage:
    """Tests for SessionManager.add_message"""

    def test_add_message_to_session(self):
        """Test adding message to existing session"""
        manager = SessionManager()
        session_id = manager.create_session()

        manager.add_message(session_id, "user", "Hello")

        assert len(manager.sessions[session_id]) == 1
        assert manager.sessions[session_id][0].role == "user"
        assert manager.sessions[session_id][0].content == "Hello"

    def test_add_message_creates_session(self):
        """Test that add_message creates session if not exists"""
        manager = SessionManager()

        manager.add_message("new_session", "user", "Hello")

        assert "new_session" in manager.sessions
        assert len(manager.sessions["new_session"]) == 1

    def test_add_multiple_messages(self):
        """Test adding multiple messages"""
        manager = SessionManager()
        session_id = manager.create_session()

        manager.add_message(session_id, "user", "Question 1")
        manager.add_message(session_id, "assistant", "Answer 1")
        manager.add_message(session_id, "user", "Question 2")

        assert len(manager.sessions[session_id]) == 3


class TestHistoryLimit:
    """Tests for conversation history limits"""

    def test_history_limit_enforced(self):
        """Test that history limit is enforced"""
        manager = SessionManager(max_history=2)  # 2 exchanges = 4 messages
        session_id = manager.create_session()

        # Add 6 messages (3 exchanges)
        for i in range(6):
            role = "user" if i % 2 == 0 else "assistant"
            manager.add_message(session_id, role, f"Message {i}")

        # Should only keep last 4 messages (2 * max_history)
        assert len(manager.sessions[session_id]) == 4

    def test_history_keeps_recent_messages(self):
        """Test that recent messages are kept"""
        manager = SessionManager(max_history=1)  # 1 exchange = 2 messages
        session_id = manager.create_session()

        manager.add_message(session_id, "user", "Old message")
        manager.add_message(session_id, "assistant", "Old response")
        manager.add_message(session_id, "user", "New message")
        manager.add_message(session_id, "assistant", "New response")

        messages = manager.sessions[session_id]
        assert len(messages) == 2
        assert messages[0].content == "New message"
        assert messages[1].content == "New response"


class TestAddExchange:
    """Tests for SessionManager.add_exchange"""

    def test_add_exchange(self):
        """Test adding a complete exchange"""
        manager = SessionManager()
        session_id = manager.create_session()

        manager.add_exchange(session_id, "What is Python?", "Python is a programming language.")

        messages = manager.sessions[session_id]
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].content == "What is Python?"
        assert messages[1].role == "assistant"
        assert messages[1].content == "Python is a programming language."


class TestGetConversationHistory:
    """Tests for SessionManager.get_conversation_history"""

    def test_get_history_returns_formatted_string(self):
        """Test that history is returned as formatted string"""
        manager = SessionManager()
        session_id = manager.create_session()

        manager.add_message(session_id, "user", "Hello")
        manager.add_message(session_id, "assistant", "Hi there!")

        history = manager.get_conversation_history(session_id)

        assert "User: Hello" in history
        assert "Assistant: Hi there!" in history

    def test_get_history_empty_session(self):
        """Test getting history from empty session"""
        manager = SessionManager()
        session_id = manager.create_session()

        history = manager.get_conversation_history(session_id)

        assert history is None

    def test_get_history_invalid_session(self):
        """Test getting history from non-existent session"""
        manager = SessionManager()

        history = manager.get_conversation_history("invalid_session")

        assert history is None

    def test_get_history_none_session_id(self):
        """Test getting history with None session ID"""
        manager = SessionManager()

        history = manager.get_conversation_history(None)

        assert history is None


class TestClearSession:
    """Tests for SessionManager.clear_session"""

    def test_clear_session(self):
        """Test clearing a session"""
        manager = SessionManager()
        session_id = manager.create_session()

        manager.add_message(session_id, "user", "Hello")
        manager.clear_session(session_id)

        assert manager.sessions[session_id] == []

    def test_clear_nonexistent_session(self):
        """Test clearing a non-existent session doesn't raise error"""
        manager = SessionManager()

        # Should not raise an error
        manager.clear_session("nonexistent")


class TestMessageDataclass:
    """Tests for Message dataclass"""

    def test_message_creation(self):
        """Test creating a Message"""
        msg = Message(role="user", content="Hello")

        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_equality(self):
        """Test Message equality"""
        msg1 = Message(role="user", content="Hello")
        msg2 = Message(role="user", content="Hello")

        assert msg1 == msg2
