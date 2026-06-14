#!/usr/bin/env python3
from waitress import serve

from metatranscribe.config import (
    load_settings,
    validate_polish_credentials,
    validate_provider_credentials,
    validate_reconciler_credentials,
    validate_web_credentials,
)
from metatranscribe.logging_utils import configure_logging
from metatranscribe.web.app import create_app


def main() -> None:
    settings = load_settings()
    configure_logging(settings.logs_dir / "web.log", settings.log_level)
    validate_web_credentials(settings)
    # Fail fast if the pipeline itself isn't configured -- the worker needs these.
    validate_provider_credentials(settings)
    validate_reconciler_credentials(settings)
    validate_polish_credentials(settings)

    app = create_app(settings)
    print(f"Serving metatranscribe web interface on http://{settings.web_host}:{settings.web_port}")
    serve(app, host=settings.web_host, port=settings.web_port)


if __name__ == "__main__":
    main()
