"""Tests for AIGenerator module"""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestAIGeneratorInitialization:
    """Tests for AIGenerator initialization"""

    def test_initialization(self):
        """Test AIGenerator initializes with correct parameters"""
        with patch('ai_generator.anthropic.Anthropic') as MockClient:
            from ai_generator import AIGenerator

            generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")

            MockClient.assert_called_once_with(api_key="test-api-key")
            assert generator.model == "claude-sonnet-4-20250514"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800


class TestGenerateResponse:
    """Tests for AIGenerator.generate_response method"""

    @pytest.fixture
    def mock_generator(self, mock_anthropic_response):
        """Create AIGenerator with mocked client"""
        with patch('ai_generator.anthropic.Anthropic') as MockClient:
            from ai_generator import AIGenerator

            generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
            generator.client.messages.create.return_value = mock_anthropic_response

            return generator

    def test_generate_simple_response(self, mock_generator):
        """Test generating a simple response without tools"""
        response = mock_generator.generate_response("What is Python?")

        assert response == "This is a test response from Claude."
        mock_generator.client.messages.create.assert_called_once()

    def test_generate_response_with_history(self, mock_generator):
        """Test generating response with conversation history"""
        history = "User: Previous question\nAssistant: Previous answer"

        mock_generator.generate_response("Follow up question", conversation_history=history)

        call_args = mock_generator.client.messages.create.call_args
        assert history in call_args.kwargs["system"]

    def test_generate_response_without_history(self, mock_generator):
        """Test that system prompt is used without history"""
        mock_generator.generate_response("Simple question")

        call_args = mock_generator.client.messages.create.call_args
        assert "AI assistant" in call_args.kwargs["system"]
        assert "Previous conversation" not in call_args.kwargs["system"]

    def test_generate_response_with_tools(self, mock_generator):
        """Test that tools are passed to API call"""
        tools = [{"name": "search", "description": "Search tool"}]

        mock_generator.generate_response("Question", tools=tools)

        call_args = mock_generator.client.messages.create.call_args
        assert call_args.kwargs["tools"] == tools
        assert call_args.kwargs["tool_choice"] == {"type": "auto"}


class TestToolExecution:
    """Tests for tool execution handling"""

    @pytest.fixture
    def mock_generator_with_tools(self, mock_anthropic_tool_response, mock_anthropic_response):
        """Create AIGenerator configured for tool use"""
        with patch('ai_generator.anthropic.Anthropic') as MockClient:
            from ai_generator import AIGenerator

            generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
            # First call returns tool use, second call returns final response
            generator.client.messages.create.side_effect = [
                mock_anthropic_tool_response,
                mock_anthropic_response
            ]

            return generator

    def test_handle_tool_execution(self, mock_generator_with_tools):
        """Test that tool execution flow works correctly"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results here"

        tools = [{"name": "search_course_content"}]

        response = mock_generator_with_tools.generate_response(
            "Search for Python",
            tools=tools,
            tool_manager=mock_tool_manager
        )

        assert response == "This is a test response from Claude."
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )

    def test_tool_results_sent_back(self, mock_generator_with_tools):
        """Test that tool results are sent back to Claude"""
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool output"

        mock_generator_with_tools.generate_response(
            "Query",
            tools=[{"name": "test"}],
            tool_manager=mock_tool_manager
        )

        # Second call should include tool results
        second_call = mock_generator_with_tools.client.messages.create.call_args_list[1]
        messages = second_call.kwargs["messages"]

        # Should have: user message, assistant tool use, user tool result
        assert len(messages) == 3
        assert messages[2]["role"] == "user"

    def test_no_tool_execution_without_manager(self):
        """Test that tool use response returns text content when no manager provided"""
        with patch('ai_generator.anthropic.Anthropic') as MockClient:
            from ai_generator import AIGenerator

            generator = AIGenerator("test-api-key", "claude-sonnet-4-20250514")

            # Create response with tool_use - when no tool_manager is provided,
            # the code falls through to return response.content[0].text
            response = Mock()
            response.stop_reason = "tool_use"
            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.text = "Tool response text"
            response.content = [tool_block]

            generator.client.messages.create.return_value = response

            # Without tool_manager, returns the text from first content block
            result = generator.generate_response("Query", tools=[{"name": "test"}])
            assert result == "Tool response text"
