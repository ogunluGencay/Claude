"""Tests for Pydantic models"""
import pytest
from models import Lesson, Course, CourseChunk


class TestLessonModel:
    """Tests for Lesson model"""

    def test_lesson_valid(self):
        """Test creating valid Lesson"""
        lesson = Lesson(
            lesson_number=1,
            title="Introduction",
            lesson_link="https://example.com/lesson1"
        )

        assert lesson.lesson_number == 1
        assert lesson.title == "Introduction"
        assert lesson.lesson_link == "https://example.com/lesson1"

    def test_lesson_optional_link(self):
        """Test Lesson with optional link"""
        lesson = Lesson(lesson_number=0, title="Overview")

        assert lesson.lesson_link is None

    def test_lesson_serialization(self):
        """Test Lesson serialization to dict"""
        lesson = Lesson(lesson_number=1, title="Test")

        data = lesson.model_dump()

        assert data["lesson_number"] == 1
        assert data["title"] == "Test"


class TestCourseModel:
    """Tests for Course model"""

    def test_course_valid(self):
        """Test creating valid Course"""
        course = Course(
            title="Python Basics",
            course_link="https://example.com/python",
            instructor="John Doe",
            lessons=[]
        )

        assert course.title == "Python Basics"
        assert course.course_link == "https://example.com/python"
        assert course.instructor == "John Doe"
        assert course.lessons == []

    def test_course_optional_fields(self):
        """Test Course with optional fields"""
        course = Course(title="Minimal Course")

        assert course.course_link is None
        assert course.instructor is None
        assert course.lessons == []

    def test_course_with_lessons(self):
        """Test Course with lessons"""
        lessons = [
            Lesson(lesson_number=0, title="Intro"),
            Lesson(lesson_number=1, title="Basics")
        ]
        course = Course(title="Test Course", lessons=lessons)

        assert len(course.lessons) == 2
        assert course.lessons[0].lesson_number == 0

    def test_course_serialization(self):
        """Test Course serialization"""
        course = Course(
            title="Test",
            instructor="Teacher",
            lessons=[Lesson(lesson_number=1, title="L1")]
        )

        data = course.model_dump()

        assert data["title"] == "Test"
        assert len(data["lessons"]) == 1


class TestCourseChunkModel:
    """Tests for CourseChunk model"""

    def test_course_chunk_valid(self):
        """Test creating valid CourseChunk"""
        chunk = CourseChunk(
            content="This is the lesson content.",
            course_title="Python Course",
            lesson_number=1,
            chunk_index=0
        )

        assert chunk.content == "This is the lesson content."
        assert chunk.course_title == "Python Course"
        assert chunk.lesson_number == 1
        assert chunk.chunk_index == 0

    def test_course_chunk_optional_lesson(self):
        """Test CourseChunk without lesson number"""
        chunk = CourseChunk(
            content="General content",
            course_title="Course",
            chunk_index=0
        )

        assert chunk.lesson_number is None

    def test_course_chunk_serialization(self):
        """Test CourseChunk serialization"""
        chunk = CourseChunk(
            content="Content",
            course_title="Test",
            lesson_number=2,
            chunk_index=5
        )

        data = chunk.model_dump()

        assert data["content"] == "Content"
        assert data["course_title"] == "Test"
        assert data["lesson_number"] == 2
        assert data["chunk_index"] == 5

    def test_course_chunk_required_fields(self):
        """Test that required fields are enforced"""
        with pytest.raises(Exception):  # Pydantic ValidationError
            CourseChunk(content="Test")  # Missing course_title and chunk_index
