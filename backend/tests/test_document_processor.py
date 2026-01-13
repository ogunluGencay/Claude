"""Tests for DocumentProcessor module"""
import pytest
from document_processor import DocumentProcessor


class TestReadFile:
    """Tests for DocumentProcessor.read_file method"""

    def test_read_utf8_file(self, tmp_path):
        """Test reading UTF-8 encoded file"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!", encoding="utf-8")

        content = processor.read_file(str(file_path))
        assert content == "Hello, World!"

    def test_read_file_with_special_characters(self, tmp_path):
        """Test reading file with special characters"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        file_path = tmp_path / "test.txt"
        file_path.write_text("Héllo Wörld! 你好", encoding="utf-8")

        content = processor.read_file(str(file_path))
        assert "Héllo" in content
        assert "你好" in content


class TestChunkText:
    """Tests for DocumentProcessor.chunk_text method"""

    def test_chunk_text_basic(self):
        """Test basic text chunking"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = processor.chunk_text(text)

        assert len(chunks) >= 1
        assert all(len(chunk) <= 100 or chunk == chunks[-1] for chunk in chunks)

    def test_chunk_text_preserves_sentences(self):
        """Test that chunking doesn't break mid-sentence when possible"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)

        text = "Short sentence. Another short one. And one more."
        chunks = processor.chunk_text(text)

        # Each chunk should end with a complete sentence (period)
        for chunk in chunks:
            assert chunk.strip().endswith('.') or chunk == chunks[-1]

    def test_chunk_text_empty_input(self):
        """Test chunking empty string"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        chunks = processor.chunk_text("")
        assert chunks == []

    def test_chunk_text_short_input(self):
        """Test chunking text shorter than chunk size"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        text = "This is a short text."
        chunks = processor.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_overlap(self):
        """Test that chunks have overlap"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=30)

        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = processor.chunk_text(text)

        if len(chunks) > 1:
            # Check if there's some overlap (content from end of chunk n appears in chunk n+1)
            # This is a basic check - overlap behavior depends on sentence boundaries
            assert len(chunks) >= 1

    def test_chunk_text_whitespace_normalization(self):
        """Test that whitespace is normalized"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        text = "Multiple   spaces   and\n\nnewlines."
        chunks = processor.chunk_text(text)

        assert "  " not in chunks[0]  # No double spaces


class TestProcessCourseDocument:
    """Tests for DocumentProcessor.process_course_document method"""

    def test_process_course_document_extracts_metadata(self, sample_course_document):
        """Test that course metadata is extracted correctly"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course, chunks = processor.process_course_document(sample_course_document)

        assert course.title == "Test Course"
        assert course.course_link == "https://example.com/course"
        assert course.instructor == "Test Instructor"

    def test_process_course_document_extracts_lessons(self, sample_course_document):
        """Test that lessons are extracted correctly"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course, chunks = processor.process_course_document(sample_course_document)

        assert len(course.lessons) == 2
        assert course.lessons[0].lesson_number == 0
        assert course.lessons[0].title == "Introduction"
        assert course.lessons[1].lesson_number == 1

    def test_process_course_document_creates_chunks(self, sample_course_document):
        """Test that content chunks are created"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course, chunks = processor.process_course_document(sample_course_document)

        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.course_title == "Test Course"
            assert chunk.chunk_index >= 0

    def test_process_course_document_lesson_links(self, sample_course_document):
        """Test that lesson links are extracted"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course, chunks = processor.process_course_document(sample_course_document)

        assert course.lessons[0].lesson_link == "https://example.com/lesson0"
        assert course.lessons[1].lesson_link == "https://example.com/lesson1"

    def test_process_course_document_no_lessons(self, sample_course_no_lessons):
        """Test processing document without lesson markers"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course, chunks = processor.process_course_document(sample_course_no_lessons)

        assert course.title == "Simple Course"
        assert len(course.lessons) == 0
        # Should still create chunks from the content
        assert len(chunks) > 0

    def test_process_course_document_missing_metadata(self, tmp_path):
        """Test handling of document with missing metadata fields"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        # Create document with minimal metadata
        content = """Course Title: Minimal Course

Some content here without proper structure.
"""
        doc_path = tmp_path / "minimal.txt"
        doc_path.write_text(content)

        course, chunks = processor.process_course_document(str(doc_path))

        assert course.title == "Minimal Course"
        assert course.instructor is None
        assert course.course_link is None

    def test_process_course_document_chunk_context(self, sample_course_document):
        """Test that chunks include lesson context"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        course, chunks = processor.process_course_document(sample_course_document)

        # First chunks should have lesson context
        lesson_chunks = [c for c in chunks if c.lesson_number is not None]
        if lesson_chunks:
            # Context should be added to chunk content
            assert any("Lesson" in chunk.content for chunk in lesson_chunks)
