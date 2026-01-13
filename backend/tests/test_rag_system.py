"""Tests for RAGSystem orchestrator"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import os


class TestRAGSystemInitialization:
    """Tests for RAGSystem initialization"""

    def test_initialization_creates_components(self, test_config):
        """Test that RAGSystem initializes all components"""
        with patch('rag_system.DocumentProcessor') as MockDP, \
             patch('rag_system.VectorStore') as MockVS, \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.SessionManager') as MockSM, \
             patch('rag_system.ToolManager') as MockTM, \
             patch('rag_system.CourseSearchTool') as MockCST:

            from rag_system import RAGSystem
            rag = RAGSystem(test_config)

            MockDP.assert_called_once_with(test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP)
            MockVS.assert_called_once_with(test_config.CHROMA_PATH, test_config.EMBEDDING_MODEL, test_config.MAX_RESULTS)
            MockAI.assert_called_once_with(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            MockSM.assert_called_once_with(test_config.MAX_HISTORY)


class TestRAGSystemQuery:
    """Tests for RAGSystem.query method"""

    @pytest.fixture
    def mock_rag_system(self, test_config):
        """Create RAGSystem with mocked dependencies"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as MockAI, \
             patch('rag_system.SessionManager') as MockSM, \
             patch('rag_system.ToolManager') as MockTM, \
             patch('rag_system.CourseSearchTool'):

            from rag_system import RAGSystem
            rag = RAGSystem(test_config)

            # Configure mocks
            rag.ai_generator.generate_response.return_value = "Test response"
            rag.session_manager.get_conversation_history.return_value = None
            rag.tool_manager.get_tool_definitions.return_value = []
            rag.tool_manager.get_last_sources.return_value = ["Source 1"]

            return rag

    def test_query_without_session(self, mock_rag_system):
        """Test query works without session"""
        response, sources = mock_rag_system.query("Test question")

        assert response == "Test response"
        assert sources == ["Source 1"]
        mock_rag_system.ai_generator.generate_response.assert_called_once()

    def test_query_with_session(self, mock_rag_system):
        """Test query uses session history when provided"""
        mock_rag_system.session_manager.get_conversation_history.return_value = "Previous: Context"

        response, sources = mock_rag_system.query("Test question", "session_1")

        mock_rag_system.session_manager.get_conversation_history.assert_called_with("session_1")
        mock_rag_system.session_manager.add_exchange.assert_called_once()

    def test_query_resets_sources(self, mock_rag_system):
        """Test that query resets sources after retrieval"""
        mock_rag_system.query("Test question")
        mock_rag_system.tool_manager.reset_sources.assert_called_once()

    def test_query_passes_tools_to_ai_generator(self, mock_rag_system):
        """Test that query passes tools to AI generator"""
        mock_rag_system.tool_manager.get_tool_definitions.return_value = [{"name": "test_tool"}]

        mock_rag_system.query("Test question")

        call_args = mock_rag_system.ai_generator.generate_response.call_args
        assert call_args.kwargs["tools"] == [{"name": "test_tool"}]


class TestRAGSystemAddCourseDocument:
    """Tests for RAGSystem.add_course_document method"""

    @pytest.fixture
    def mock_rag_for_docs(self, test_config):
        """Create RAGSystem with mocked dependencies for document tests"""
        with patch('rag_system.DocumentProcessor') as MockDP, \
             patch('rag_system.VectorStore') as MockVS, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'):

            from rag_system import RAGSystem
            from models import Course, CourseChunk

            rag = RAGSystem(test_config)

            # Configure document processor mock
            mock_course = Course(title="Test Course", course_link="https://example.com")
            mock_chunks = [
                CourseChunk(content="Chunk 1", course_title="Test Course", chunk_index=0),
                CourseChunk(content="Chunk 2", course_title="Test Course", chunk_index=1)
            ]
            rag.document_processor.process_course_document.return_value = (mock_course, mock_chunks)

            return rag

    def test_add_course_document_success(self, mock_rag_for_docs):
        """Test successful document addition"""
        course, chunk_count = mock_rag_for_docs.add_course_document("/path/to/doc.txt")

        assert course.title == "Test Course"
        assert chunk_count == 2
        mock_rag_for_docs.vector_store.add_course_metadata.assert_called_once()
        mock_rag_for_docs.vector_store.add_course_content.assert_called_once()

    def test_add_course_document_handles_error(self, mock_rag_for_docs):
        """Test error handling during document addition"""
        mock_rag_for_docs.document_processor.process_course_document.side_effect = Exception("Parse error")

        course, chunk_count = mock_rag_for_docs.add_course_document("/path/to/doc.txt")

        assert course is None
        assert chunk_count == 0


class TestRAGSystemAddCourseFolder:
    """Tests for RAGSystem.add_course_folder method"""

    @pytest.fixture
    def mock_rag_for_folder(self, test_config, tmp_path):
        """Create RAGSystem with mocked dependencies for folder tests"""
        with patch('rag_system.DocumentProcessor') as MockDP, \
             patch('rag_system.VectorStore') as MockVS, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'):

            from rag_system import RAGSystem
            from models import Course, CourseChunk

            rag = RAGSystem(test_config)

            # Configure vector store mock
            rag.vector_store.get_existing_course_titles.return_value = []

            # Configure document processor mock
            mock_course = Course(title="Test Course")
            mock_chunks = [CourseChunk(content="Chunk 1", course_title="Test Course", chunk_index=0)]
            rag.document_processor.process_course_document.return_value = (mock_course, mock_chunks)

            return rag

    def test_add_course_folder_success(self, mock_rag_for_folder, tmp_path):
        """Test successful folder processing"""
        # Create test file
        (tmp_path / "course.txt").write_text("Course Title: Test")

        courses, chunks = mock_rag_for_folder.add_course_folder(str(tmp_path))

        assert courses == 1
        assert chunks == 1

    def test_add_course_folder_skips_existing(self, mock_rag_for_folder, tmp_path):
        """Test that existing courses are skipped"""
        (tmp_path / "course.txt").write_text("Course Title: Test")
        mock_rag_for_folder.vector_store.get_existing_course_titles.return_value = ["Test Course"]

        courses, chunks = mock_rag_for_folder.add_course_folder(str(tmp_path))

        assert courses == 0
        assert chunks == 0

    def test_add_course_folder_nonexistent(self, mock_rag_for_folder):
        """Test handling of non-existent folder"""
        courses, chunks = mock_rag_for_folder.add_course_folder("/nonexistent/path")

        assert courses == 0
        assert chunks == 0

    def test_add_course_folder_clear_existing(self, mock_rag_for_folder, tmp_path):
        """Test clear_existing option"""
        (tmp_path / "course.txt").write_text("Course Title: Test")

        mock_rag_for_folder.add_course_folder(str(tmp_path), clear_existing=True)

        mock_rag_for_folder.vector_store.clear_all_data.assert_called_once()


class TestRAGSystemGetCourseAnalytics:
    """Tests for RAGSystem.get_course_analytics method"""

    def test_get_course_analytics(self, test_config):
        """Test course analytics retrieval"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as MockVS, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.SessionManager'), \
             patch('rag_system.ToolManager'), \
             patch('rag_system.CourseSearchTool'):

            from rag_system import RAGSystem

            rag = RAGSystem(test_config)
            rag.vector_store.get_course_count.return_value = 3
            rag.vector_store.get_existing_course_titles.return_value = ["Course A", "Course B", "Course C"]

            analytics = rag.get_course_analytics()

            assert analytics["total_courses"] == 3
            assert len(analytics["course_titles"]) == 3
