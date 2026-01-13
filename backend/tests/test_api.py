"""Tests for FastAPI endpoints"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_rag_system():
    """Create a mock RAGSystem"""
    rag = Mock()
    rag.query.return_value = ("Test answer", ["Test Course - Lesson 1"])
    rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Test Course", "Another Course"]
    }
    rag.session_manager = Mock()
    rag.session_manager.create_session.return_value = "session_1"
    return rag


@pytest.fixture
def client(mock_rag_system):
    """Create test client with mocked RAGSystem"""
    with patch('app.RAGSystem') as MockRAG:
        MockRAG.return_value = mock_rag_system
        with patch('app.rag_system', mock_rag_system):
            from app import app
            with TestClient(app, raise_server_exceptions=False) as test_client:
                yield test_client


class TestQueryEndpoint:
    """Tests for POST /api/query endpoint"""

    def test_query_success(self, client, mock_rag_system):
        """Test successful query returns expected response structure"""
        response = client.post(
            "/api/query",
            json={"query": "What is this course about?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert isinstance(data["sources"], list)

    def test_query_creates_session_when_not_provided(self, client, mock_rag_system):
        """Test that a new session is created when session_id is not provided"""
        response = client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session_1"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_uses_existing_session(self, client, mock_rag_system):
        """Test that existing session_id is used when provided"""
        response = client.post(
            "/api/query",
            json={"query": "Test query", "session_id": "existing_session"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "existing_session"
        mock_rag_system.session_manager.create_session.assert_not_called()

    def test_query_returns_answer_and_sources(self, client, mock_rag_system):
        """Test that query returns both answer and sources"""
        response = client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer"
        assert data["sources"] == ["Test Course - Lesson 1"]

    def test_query_handles_rag_error(self, client, mock_rag_system):
        """Test that RAG system errors return 500"""
        mock_rag_system.query.side_effect = Exception("RAG error")
        response = client.post(
            "/api/query",
            json={"query": "Test query"}
        )
        assert response.status_code == 500

    def test_query_empty_query_validation(self, client):
        """Test that empty query is handled"""
        response = client.post(
            "/api/query",
            json={"query": ""}
        )
        # Empty string is technically valid, endpoint should handle it
        assert response.status_code in [200, 422]


class TestCoursesEndpoint:
    """Tests for GET /api/courses endpoint"""

    def test_courses_success(self, client, mock_rag_system):
        """Test successful course stats retrieval"""
        response = client.get("/api/courses")
        assert response.status_code == 200
        data = response.json()
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2

    def test_courses_returns_correct_structure(self, client, mock_rag_system):
        """Test that courses endpoint returns correct structure"""
        response = client.get("/api/courses")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

    def test_courses_handles_error(self, client, mock_rag_system):
        """Test that course stats errors return 500"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        response = client.get("/api/courses")
        assert response.status_code == 500
