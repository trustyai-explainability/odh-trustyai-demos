"""Run the TrustyAI application on an externally reachable local HTTP port."""

from __future__ import annotations

import asyncio
import os

from hypercorn.asyncio import serve
from hypercorn.config import Config
from trustyai_service.main import app


async def run() -> None:
    """Serve the image from the container on the local demo port."""
    config = Config()
    config.bind = [f"0.0.0.0:{os.environ.get('TRUSTYAI_HTTP_PORT', '8081')}"]
    config.accesslog = "-"
    config.errorlog = "-"
    config.use_reloader = False
    await serve(app, config)


if __name__ == "__main__":
    asyncio.run(run())
