import json
import logging

from luber_shared.logging import configure_logging


def test_log_lines_are_json_with_context_fields(capsys):
    configure_logging(service="test-service", level="INFO")
    logger = logging.getLogger("luber.test")

    logger.info(
        "generation queued",
        extra={"request_id": "req-1", "generation_id": "gen-1"},
    )

    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["message"] == "generation queued"
    assert payload["service"] == "test-service"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "req-1"
    assert payload["generation_id"] == "gen-1"
    assert "timestamp" in payload


def test_configure_logging_is_idempotent(capsys):
    configure_logging(service="svc")
    configure_logging(service="svc")
    logging.getLogger("luber.idempotent").warning("once")

    out = capsys.readouterr().out
    occurrences = [ln for ln in out.strip().splitlines() if "once" in ln]
    assert len(occurrences) == 1
