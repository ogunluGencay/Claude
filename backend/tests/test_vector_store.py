"""Tests for VectorStore module"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from vector_store import VectorStore, SearchResults


class TestSearchResults:
    """Tests for SearchResults dataclass"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB results"""
        chroma_results = {
            'documents': [['Doc 1', 'Doc 2']],
            'metadatas': [[{'course': 'A'}, {'course': 'B'}]],
            'distances': [[0.1, 0.2]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ['Doc 1', 'Doc 2']
        assert results.metadata == [{'course': 'A'}, {'course': 'B'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.is_empty()

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("No course found")

        assert results.documents == []
        assert results.error == "No course found"
        assert results.is_empty()

    def test_is_empty(self):
        """Test is_empty method"""
        empty = SearchResults(documents=[], metadata=[], distances=[])
        not_empty = SearchResults(documents=["doc"], metadata=[{}], distances=[0.1])

        assert empty.is_empty()
        assert not not_empty.is_empty()


class TestVectorStoreInitialization:
    """Tests for VectorStore initialization"""

    def test_initialization(self, mock_chroma_client):
        """Test VectorStore initializes collections"""
        with patch('vector_store.chromadb.PersistentClient', return_value=mock_chroma_client), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore("./test_db", "all-MiniLM-L6-v2", max_results=5)

            assert store.max_results == 5
            assert mock_chroma_client.get_or_create_collection.call_count == 2


class TestVectorStoreSearch:
    """Tests for VectorStore.search method"""

    @pytest.fixture
    def mock_store(self, mock_chroma_client, mock_chroma_collection):
        """Create VectorStore with mocked dependencies"""
        with patch('vector_store.chromadb.PersistentClient', return_value=mock_chroma_client), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            store = VectorStore("./test_db", "all-MiniLM-L6-v2", max_results=5)
            return store

    def test_search_no_filters(self, mock_store):
        """Test search without any filters"""
        results = mock_store.search(query="Python basics")

        assert not results.is_empty()
        mock_store.course_content.query.assert_called_once()

    def test_search_with_course_name(self, mock_store):
        """Test search with course name filter"""
        # Configure mock to resolve course name
        mock_store.course_catalog.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course'}]],
            'distances': [[0.1]]
        }

        results = mock_store.search(query="test", course_name="Test")

        # Should have called course_catalog to resolve name
        mock_store.course_catalog.query.assert_called()

    def test_search_course_not_found(self, mock_store):
        """Test search when course is not found"""
        mock_store.course_catalog.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        results = mock_store.search(query="test", course_name="NonExistent")

        assert results.error is not None
        assert "No course found" in results.error

    def test_search_with_lesson_filter(self, mock_store):
        """Test search with lesson number filter"""
        results = mock_store.search(query="test", lesson_number=1)

        call_args = mock_store.course_content.query.call_args
        assert call_args.kwargs["where"] == {"lesson_number": 1}

    def test_search_with_limit(self, mock_store):
        """Test search respects limit parameter"""
        mock_store.search(query="test", limit=3)

        call_args = mock_store.course_content.query.call_args
        assert call_args.kwargs["n_results"] == 3

    def test_search_uses_default_limit(self, mock_store):
        """Test search uses default max_results"""
        mock_store.search(query="test")

        call_args = mock_store.course_content.query.call_args
        assert call_args.kwargs["n_results"] == 5


class TestBuildFilter:
    """Tests for VectorStore._build_filter method"""

    @pytest.fixture
    def mock_store(self, mock_chroma_client):
        """Create VectorStore for filter tests"""
        with patch('vector_store.chromadb.PersistentClient', return_value=mock_chroma_client), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            return VectorStore("./test_db", "all-MiniLM-L6-v2")

    def test_build_filter_no_params(self, mock_store):
        """Test filter with no parameters"""
        result = mock_store._build_filter(None, None)
        assert result is None

    def test_build_filter_course_only(self, mock_store):
        """Test filter with course title only"""
        result = mock_store._build_filter("Test Course", None)
        assert result == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, mock_store):
        """Test filter with lesson number only"""
        result = mock_store._build_filter(None, 1)
        assert result == {"lesson_number": 1}

    def test_build_filter_both(self, mock_store):
        """Test filter with both course and lesson"""
        result = mock_store._build_filter("Test Course", 2)

        assert result == {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 2}
            ]
        }


class TestVectorStoreAddMethods:
    """Tests for VectorStore add methods"""

    @pytest.fixture
    def mock_store(self, mock_chroma_client):
        """Create VectorStore for add tests"""
        with patch('vector_store.chromadb.PersistentClient', return_value=mock_chroma_client), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            return VectorStore("./test_db", "all-MiniLM-L6-v2")

    def test_add_course_metadata(self, mock_store):
        """Test adding course metadata"""
        from models import Course, Lesson

        course = Course(
            title="Test Course",
            course_link="https://example.com",
            instructor="Test Teacher",
            lessons=[Lesson(lesson_number=1, title="Intro", lesson_link="https://example.com/1")]
        )

        mock_store.add_course_metadata(course)

        mock_store.course_catalog.add.assert_called_once()
        call_args = mock_store.course_catalog.add.call_args
        assert call_args.kwargs["ids"] == ["Test Course"]

    def test_add_course_content(self, mock_store):
        """Test adding course content chunks"""
        from models import CourseChunk

        chunks = [
            CourseChunk(content="Chunk 1", course_title="Test", lesson_number=1, chunk_index=0),
            CourseChunk(content="Chunk 2", course_title="Test", lesson_number=1, chunk_index=1)
        ]

        mock_store.add_course_content(chunks)

        mock_store.course_content.add.assert_called_once()
        call_args = mock_store.course_content.add.call_args
        assert len(call_args.kwargs["documents"]) == 2

    def test_add_course_content_empty(self, mock_store):
        """Test adding empty chunks list"""
        mock_store.add_course_content([])

        mock_store.course_content.add.assert_not_called()


class TestVectorStoreClearAndGet:
    """Tests for VectorStore clear and get methods"""

    @pytest.fixture
    def mock_store(self, mock_chroma_client):
        """Create VectorStore for clear/get tests"""
        with patch('vector_store.chromadb.PersistentClient', return_value=mock_chroma_client), \
             patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction'):

            return VectorStore("./test_db", "all-MiniLM-L6-v2")

    def test_clear_all_data(self, mock_store):
        """Test clearing all data"""
        mock_store.clear_all_data()

        assert mock_store.client.delete_collection.call_count == 2

    def test_get_existing_course_titles(self, mock_store):
        """Test getting existing course titles"""
        titles = mock_store.get_existing_course_titles()

        assert titles == ['Test Course']

    def test_get_course_count(self, mock_store):
        """Test getting course count"""
        count = mock_store.get_course_count()

        assert count == 1

    def test_get_course_link(self, mock_store):
        """Test getting course link"""
        link = mock_store.get_course_link("Test Course")

        assert link == "https://example.com"
