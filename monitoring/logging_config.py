import json
import logging
import logging.config
from datetime import datetime, timezone

from config.settings import get_settings



class JSONFormatter(logging.Formatter):

    def format(self, record):

        log_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(
                record.exc_info
            )

        return json.dumps(log_record)


def setup_logging():

    settings = get_settings()

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "json": {
                "()": JSONFormatter
            }
        },

        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level.upper(),
                "formatter": "json"
            }
        },

        "root": {
            "handlers": ["console"],
            "level": settings.log_level.upper()
        }
    }

    logging.config.dictConfig(logging_config)