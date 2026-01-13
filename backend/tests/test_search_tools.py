"""Tests for search_tools module"""
import pytest
from unittest.mock import Mock, MagicMock
from search_tools import CourseSearchTool, ToolManager, Tool
from vector_store import SearchResults


class TestCourseSearchToolDefinition:
    """Tests for CourseSearchTool.get_tool_definition"""

    @pytest.fixture
    def search_tool(self):
        """Create CourseSearchTool with mocked vector store"""
        mock_store = Mock()
        return CourseSearchTool(mock_store)

    def test_tool_definition_structure(self, search_tool):
        """Test that tool definition has correct structure"""
        definition = search_tool.get_tool_definition()

        assert "name" in definition
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["name"] == "search_course_content"

    def test_tool_definition_schema(self, search_tool):
        """Test that input schema is correctly defined"""
        definition = search_tool.get_tool_definition()
        schema = definition["input_schema"]

        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]
        assert "query" in schema["required"]


class TestCourseSearchToolExecute:
    """Tests for CourseSearchTool.execute method"""

    @pytest.fixture
    def search_tool_with_results(self, mock_search_results):
        """Create CourseSearchTool with mocked results"""
        mock_store = Mock()
        mock_store.search.return_value = mock_search_results
        return CourseSearchTool(mock_store)

    @pytest.fixture
    def search_tool_empty_results(self, mock_empty_search_results):
        """Create CourseSearchTool with empty results"""
        mock_store = Mock()
        mock_store.search.return_value = mock_empty_search_results
        return CourseSearchTool(mock_store)

    def test_execute_basic_search(self, search_tool_with_results):
        """Test basic search execution"""
        result = search_tool_with_results.execute(query="Python basics")

        assert isinstance(result, str)
        assert len(result) > 0
        search_tool_with_results.store.search.assert_called_once_with(
            query="Python basics",
            course_name=None,
            lesson_number=None
        )

    def test_execute_with_course_filter(self, search_tool_with_results):
        """Test search with course name filter"""
        search_tool_with_results.execute(query="test", course_name="Python Course")

        search_tool_with_results.store.search.assert_called_with(
            query="test",
            course_name="Python Course",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, search_tool_with_results):
        """Test search with lesson number filter"""
        search_tool_with_results.execute(query="test", lesson_number=1)

        search_tool_with_results.store.search.assert_called_with(
            query="test",
            course_name=None,
            lesson_number=1
        )

    def test_execute_with_all_filters(self, search_tool_with_results):
        """Test search with all filters"""
        search_tool_with_results.execute(
            query="test",
            course_name="Python Course",
            lesson_number=2
        )

        search_tool_with_results.store.search.assert_called_with(
            query="test",
            course_name="Python Course",
            lesson_number=2
        )

    def test_execute_no_results(self, search_tool_empty_results):
        """Test handling of no results"""
        result = search_tool_empty_results.execute(query="nonexistent topic")

        assert "No results found" in result or "No relevant content" in result

    def test_execute_tracks_sources(self, search_tool_with_results):
        """Test that sources are tracked after search"""
        search_tool_with_results.execute(query="test")

        assert len(search_tool_with_results.last_sources) > 0

    def test_execute_handles_error(self):
        """Test handling of search errors"""
        mock_store = Mock()
        error_results = SearchResults.empty("Search error: Connection failed")
        mock_store.search.return_value = error_results

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        assert "Search error" in result


class TestFormatResults:
    """Tests for CourseSearchTool._format_results method"""

    def test_format_results_with_lesson(self, mock_search_results):
        """Test formatting results with lesson number"""
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)

        formatted = tool._format_results(mock_search_results)

        assert "Test Course" in formatted
        assert "Lesson 1" in formatted
        assert "Test content" in formatted

    def test_format_results_tracks_sources(self, mock_search_results):
        """Test that formatting tracks sources"""
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)

        tool._format_results(mock_search_results)

        assert "Test Course - Lesson 1" in tool.last_sources


class TestToolManager:
    """Tests for ToolManager class"""

    @pytest.fixture
    def tool_manager(self):
        """Create empty ToolManager"""
        return ToolManager()

    @pytest.fixture
    def mock_tool(self):
        """Create mock tool"""
        tool = Mock(spec=Tool)
        tool.get_tool_definition.return_value = {
            "name": "test_tool",
            "description": "A test tool"
        }
        tool.execute.return_value = "Tool executed"
        tool.last_sources = ["Source 1"]
        return tool

    def test_register_tool(self, tool_manager, mock_tool):
        """Test registering a tool"""
        tool_manager.register_tool(mock_tool)

        assert "test_tool" in tool_manager.tools

    def test_register_tool_without_name(self, tool_manager):
        """Test registering tool without name raises error"""
        bad_tool = Mock()
        bad_tool.get_tool_definition.return_value = {"description": "No name"}

        with pytest.raises(ValueError):
            tool_manager.register_tool(bad_tool)

    def test_get_tool_definitions(self, tool_manager, mock_tool):
        """Test getting all tool definitions"""
        tool_manager.register_tool(mock_tool)

        definitions = tool_manager.get_tool_definitions()

        assert len(definitions) == 1
        assert definitions[0]["name"] == "test_tool"

    def test_execute_tool(self, tool_manager, mock_tool):
        """Test executing a registered tool"""
        tool_manager.register_tool(mock_tool)

        result = tool_manager.execute_tool("test_tool", param="value")

        assert result == "Tool executed"
        mock_tool.execute.assert_called_once_with(param="value")

    def test_execute_unknown_tool(self, tool_manager):
        """Test executing unknown tool returns error"""
        result = tool_manager.execute_tool("unknown_tool")

        assert "not found" in result

    def test_get_last_sources(self, tool_manager, mock_tool):
        """Test getting sources from tools"""
        tool_manager.register_tool(mock_tool)

        sources = tool_manager.get_last_sources()

        assert sources == ["Source 1"]

    def test_get_last_sources_empty(self, tool_manager):
        """Test getting sources when none available"""
        sources = tool_manager.get_last_sources()

        assert sources == []

    def test_reset_sources(self, tool_manager, mock_tool):
        """Test resetting sources"""
        tool_manager.register_tool(mock_tool)

        tool_manager.reset_sources()

        assert mock_tool.last_sources == []
