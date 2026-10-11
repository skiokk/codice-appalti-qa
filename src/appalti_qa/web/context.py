"""Per-application state shared by the routes, injected as a FastAPI dependency."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from fastapi import Depends, Request
from fastapi.templating import Jinja2Templates

from appalti_qa.config import Settings
from appalti_qa.web.asking import AskService
from appalti_qa.web.backend import Backend
from appalti_qa.web.clients import ClientHasher, TrustedProxies, client_address


@dataclass(frozen=True, slots=True)
class WebContext:
    settings: Settings
    templates: Jinja2Templates
    backend: Backend
    asker: AskService
    trusted_proxies: TrustedProxies
    hasher: ClientHasher
    results_path: Path

    def client_hash(self, request: Request) -> str:
        return self.hasher(client_address(request, self.trusted_proxies))


def get_context(request: Request) -> WebContext:
    return request.app.state.web


Context = Annotated[WebContext, Depends(get_context)]
