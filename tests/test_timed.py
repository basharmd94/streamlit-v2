"""Unit tests for the timed() decorator in utils/utils.py.

timed() is a single point of failure: 125+ functions across the app are
wrapped by it, so a silent break there cascades everywhere. These tests
verify the five observable contracts of the decorator.
"""
import pytest
from unittest.mock import patch, MagicMock


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_timed():
    """Import timed with LogManager.logger mocked so no real logging I/O occurs."""
    mock_logger = MagicMock()
    with patch("utils.loggin_config.LogManager.logger", mock_logger):
        from utils.utils import timed
    return timed, mock_logger


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def timed_and_logger():
    mock_logger = MagicMock()
    with patch("utils.loggin_config.LogManager.logger", mock_logger):
        from utils import utils as u
        import importlib
        importlib.reload(u)          # fresh module so patch takes effect
        yield u.timed, mock_logger


# ── tests ─────────────────────────────────────────────────────────────────────

class TestTimedDecorator:
    """Contract tests for timed()."""

    def setup_method(self):
        self.mock_logger = MagicMock()
        self._patcher = patch("utils.loggin_config.LogManager.logger", self.mock_logger)
        self._patcher.start()
        import importlib, utils.utils as u
        importlib.reload(u)
        self.timed = u.timed

    def teardown_method(self):
        self._patcher.stop()

    # 1. Return value is passed through unchanged
    def test_return_value_passthrough(self):
        @self.timed
        def add(a, b):
            return a + b

        assert add(3, 4) == 7

    def test_return_none_passthrough(self):
        @self.timed
        def do_nothing():
            return None

        assert do_nothing() is None

    # 2. @wraps preserves __name__ and __doc__
    def test_wraps_preserves_name(self):
        @self.timed
        def my_function():
            """My docstring."""

        assert my_function.__name__ == "my_function"

    def test_wraps_preserves_docstring(self):
        @self.timed
        def my_function():
            """My docstring."""

        assert my_function.__doc__ == "My docstring."

    # 3. Positional and keyword args are forwarded correctly
    def test_positional_args_forwarded(self):
        received = []

        @self.timed
        def capture(*args):
            received.extend(args)

        capture(1, "hello", [3])
        assert received == [1, "hello", [3]]

    def test_keyword_args_forwarded(self):
        received = {}

        @self.timed
        def capture(**kwargs):
            received.update(kwargs)

        capture(x=10, y="world")
        assert received == {"x": 10, "y": "world"}

    def test_mixed_args_forwarded(self):
        @self.timed
        def multiply(a, b, factor=1):
            return a * b * factor

        assert multiply(2, 3, factor=4) == 24

    # 4. LogManager.logger.info is called once, containing the function name
    def test_logger_info_called_once(self):
        @self.timed
        def simple():
            pass

        simple()
        self.mock_logger.info.assert_called_once()

    def test_logger_info_contains_function_name(self):
        @self.timed
        def targeted_function():
            pass

        targeted_function()
        log_msg = self.mock_logger.info.call_args[0][0]
        assert "targeted_function" in log_msg

    def test_logger_info_contains_timing(self):
        @self.timed
        def timed_function():
            pass

        timed_function()
        log_msg = self.mock_logger.info.call_args[0][0]
        # Message ends with a seconds value like "0.0001 s" (narrow space U+2009 or regular)
        assert "took" in log_msg and log_msg.rstrip().endswith("s")

    # 5. Exceptions from the wrapped function propagate unaltered
    def test_exception_propagates(self):
        @self.timed
        def explode():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            explode()

    def test_exception_type_preserved(self):
        @self.timed
        def raise_key_error():
            raise KeyError("missing_key")

        with pytest.raises(KeyError):
            raise_key_error()

    def test_logger_not_called_on_exception(self):
        """Logging must NOT fire if the function raises — no partial log noise."""
        @self.timed
        def explode():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            explode()

        self.mock_logger.info.assert_not_called()
