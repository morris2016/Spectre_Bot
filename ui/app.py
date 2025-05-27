#!/usr/bin/env python3
"""UI Service for QuantumSpectre Elite Trading System."""

import asyncio
import os
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from common.logger import get_logger
from common.metrics import MetricsCollector


class UIService:
    """Service responsible for running the web based user interface."""

    def __init__(self, config: Any, loop: Optional[asyncio.AbstractEventLoop] = None,
                 redis_client: Any = None, db_client: Any = None) -> None:
        self.config = config
        self.loop = loop or asyncio.get_event_loop()
        self.redis_client = redis_client
        self.db_client = db_client
        self.logger = get_logger("UIService")
        self.metrics = MetricsCollector("ui")

        self._server: Optional[uvicorn.Server] = None
        self.task: Optional[asyncio.Task] = None
        self.running = False

        # FastAPI application instance
        self.app = FastAPI(title="QuantumSpectre UI")
        self.app.add_api_route("/health", self.health_endpoint, methods=["GET"])

        static_dir = os.path.abspath(self.config.ui.get("static_dir", "./ui/dist"))
        index_file = self.config.ui.get("index_file", "index.html")
        self.index_path = os.path.join(static_dir, index_file)

        if os.path.isdir(static_dir):
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        else:
            self.logger.warning("UI static directory %s not found; static files will not be served", static_dir)
            if not os.path.isfile(self.index_path):
                self.index_path = None

        self.app.add_api_route("/{full_path:path}", self.index, methods=["GET"])

    async def start(self) -> None:
        """Start the UI service using Uvicorn."""
        if self.running:
            return

        host = self.config.ui.get("host", "0.0.0.0")
        port = int(self.config.ui.get("port", 3000))
        log_level = self.config.logging.get("ui_level", "info").lower()

        config = uvicorn.Config(self.app, host=host, port=port, log_level=log_level,
                                loop="asyncio")
        self._server = uvicorn.Server(config)
        self.task = self.loop.create_task(self._server.serve())
        self.running = True
        self.logger.info("UI Service started on %s:%s", host, port)

    async def stop(self) -> None:
        """Stop the UI service."""
        if not self.running:
            return
        if self._server and self._server.should_exit is False:
            self._server.should_exit = True
        if self.task:
            await self.task
        self.running = False
        self.logger.info("UI Service stopped")

    async def health_check(self) -> bool:
        """Return True if the service is running."""
        return self.running

    async def health_endpoint(self) -> dict:
        """Simple health check endpoint for FastAPI."""
        return {"status": "ok"}

    async def index(self, full_path: str) -> FileResponse:
        """Serve the React application's index file for all routes."""
        if self.index_path and os.path.isfile(self.index_path):
            return FileResponse(self.index_path)
        return FileResponse(__file__, status_code=404)
