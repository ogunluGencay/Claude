"""Shared fixtures and mocks for all tests"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass


@dataclass
class TestConfig:
    """Test configuration with mocked API key"""
    ANTHROPIC_API_KEY: str = "test-api-key"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 100
    MAX_RESULTS: int = 5
    MAX_HISTORY: int = 2
    CHROMA_PATH: str = "./test_chroma_db"


@pytest.fixture
def test_config():
    """Create a test configuration"""
    return TestConfig()


@pytest.fixture
def sample_course_document(tmp_path):
    """Create a sample course document file for testing"""
    content = """Course Title: Test Course
Course Link: https://example.com/course
Course Instructor: Test Instructor

Lesson 0: Introduction
Lesson Link: https://example.com/lesson0
This is the introduction content for the test course. It contains important information about what students will learn.

Lesson 1: First Lesson
Lesson Link: https://example.com/lesson1
This is the first lesson content with more details. Students will learn about the basics here.
"""
    doc_path = tmp_path / "test_course.txt"
    doc_path.write_text(content)
    return str(doc_path)


@pytest.fixture
def sample_course_no_lessons(tmp_path):
    """Create a sample course document without lesson markers"""
    content = """Course Title: Simple Course
Course Link: https://example.com/simple
Course Instructor: Simple Instructor

This is just plain content without any lesson markers.
It should be processed as a single document.
"""
    doc_path = tmp_path / "simple_course.txt"
    doc_path.write_text(content)
    return str(doc_path)


@pytest.fixture
def mock_anthropic_response():
    """Create a mock Anthropic API response"""
    response = Mock()
    response.stop_reason = "end_turn"
    text_block = Mock()
    text_block.type = "text"
    text_block.text = "This is a test response from Claude."
    response.content = [text_block]
    return response


@pytest.fixture
def mock_anthropic_tool_response():
    """Create a mock Anthropic API response with tool use"""
    response = Mock()
    response.stop_reason = "tool_use"

    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.id = "tool_123"
    tool_block.input = {"query": "test query"}

    response.content = [tool_block]
    return response


@pytest.fixture
def mock_chroma_collection():
    """Mock a ChromaDB collection"""
    collection = Mock()
    collection.query.return_value = {
        'documents': [['Test content from course']],
        'metadatas': [[{'course_title': 'Test Course', 'lesson_number': 1}]],
        'distances': [[0.5]]
    }
    collection.get.return_value = {
        'ids': ['Test Course'],
        'metadatas': [{'title': 'Test Course', 'instructor': 'Test Instructor', 'course_link': 'https://example.com', 'lessons_json': '[]', 'lesson_count': 0}]
    }
    collection.add.return_value = None
    collection.count.return_value = 1
    return collection


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    """Mock the ChromaDB PersistentClient"""
    client = Mock()
    client.get_or_create_collection.return_value = mock_chroma_collection
    client.delete_collection.return_value = None
    return client


@pytest.fixture
def mock_search_results():
    """Create mock search results"""
    from vector_store import SearchResults
    return SearchResults(
        documents=["Test content about Python programming"],
        metadata=[{"course_title": "Test Course", "lesson_number": 1}],
        distances=[0.25]
    )


@pytest.fixture
def mock_empty_search_results():
    """Create empty mock search results"""
    from vector_store import SearchResults
    return SearchResults.empty("No results found")
