import json
import logging


from monitoring.logging_config import (
    JSONFormatter,
    setup_logging
)


def test_json_formatter():

    formatter = JSONFormatter()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=10,
        msg="Test Message",
        args=(),
        exc_info=None
    )

    result = json.loads(formatter.format(record))

    assert result["level"] == "INFO"
    assert result["message"] == "Test Message"
    assert result["logger"] == "test_logger"


def test_logging_initialization():

    setup_logging()

    logger = logging.getLogger("application")

    assert logger.isEnabledFor(logging.INFO)
