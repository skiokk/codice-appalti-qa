"""Polite, cached downloader of the Normattiva Akoma Ntoso snapshots of D.Lgs. 36/2023.

Normattiva serves the consolidated text in force on a date (``dataVigenza``) as Akoma Ntoso
XML once a session cookie has been obtained from the act's page. The downloader first fetches
the text in force today, reads the amendment dates from its lifecycle and then fetches the text
in force on each of them: the original act on the day after its publication (its entry into
force), every amendment on the date recorded in the lifecycle. Requests are sequential and
spaced by ``delay_s`` seconds; files already present are not downloaded again.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import httpx
from lxml import etree

from appalti_qa.ingest.akn import read_events
from appalti_qa.ingest.xmlutil import AKN_NS
from appalti_qa.models import Event

logger = logging.getLogger(__name__)

ACT_CODE = "23G00044"  # codice redazionale of D.Lgs. 31 marzo 2023, n. 36
GU_DATE = "20230331"  # publication date in the Gazzetta Ufficiale
ACT_URN = "urn:nir:stato:decreto.legislativo:2023-03-31;36"
BASE_URL = "https://www.normattiva.it"
SESSION_URL = f"{BASE_URL}/uri-res/N2Ls?{ACT_URN}"
AKN_URL = f"{BASE_URL}/do/atto/caricaAKN"
USER_AGENT = (
    "codice-appalti-qa/0.1 (open-source study of D.Lgs. 36/2023; "
    "sequential requests, cached locally)"
)
TIMEOUT_S = 60.0


class DownloadError(RuntimeError):
    """Normattiva did not return a usable Akoma Ntoso document."""


def snapshot_path(raw_dir: Path, vigenza: date) -> Path:
    return raw_dir / f"{ACT_CODE}_{vigenza:%Y%m%d}.xml"


def vigenza_dates(events: Sequence[Event], today: date) -> list[date]:
    """Dates to download: entry into force of the original act, each amendment, today."""
    dates = {today}
    for event in events:
        day = event.date + timedelta(days=1) if event.kind == "original" else event.date
        if day <= today:
            dates.add(day)
    return sorted(dates)


class _PoliteSession:
    """Sequential GETs with a minimum delay, opening the Normattiva session on first use."""

    def __init__(self, client: httpx.Client, delay_s: float) -> None:
        self._client = client
        self._delay_s = delay_s
        self._last_request: float | None = None
        self._opened = False

    def get(self, url: str, params: dict[str, str] | None = None) -> httpx.Response:
        if self._last_request is not None:
            remaining = self._delay_s - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
        try:
            return self._client.get(url, params=params, headers={"User-Agent": USER_AGENT})
        finally:
            self._last_request = time.monotonic()

    def open(self, *, refresh: bool = False) -> None:
        if self._opened and not refresh:
            return
        response = self.get(SESSION_URL)
        if response.status_code >= 400:  # a redirect still sets the session cookie
            raise DownloadError(f"cannot open a Normattiva session: HTTP {response.status_code}")
        self._opened = True


def _akn_payload(response: httpx.Response, vigenza: date) -> bytes | None:
    """The XML body, None for an HTML page (expired session); raises on other failures."""
    if response.status_code != 200:
        raise DownloadError(f"HTTP {response.status_code} for dataVigenza={vigenza:%Y%m%d}")
    payload = response.content
    html_type = "html" in response.headers.get("content-type", "")
    if html_type or payload[:512].lstrip().lower().startswith((b"<!doctype", b"<html")):
        return None
    try:
        root = etree.fromstring(payload, parser=etree.XMLParser(huge_tree=True))
    except etree.XMLSyntaxError as exc:
        raise DownloadError(f"malformed XML for dataVigenza={vigenza:%Y%m%d}: {exc}") from exc
    if etree.QName(root).namespace != AKN_NS:
        raise DownloadError(f"not an Akoma Ntoso document for dataVigenza={vigenza:%Y%m%d}")
    return payload


def _fetch(session: _PoliteSession, path: Path, vigenza: date) -> Path:
    params = {"dataGU": GU_DATE, "codiceRedaz": ACT_CODE, "dataVigenza": f"{vigenza:%Y%m%d}"}
    session.open()
    payload = _akn_payload(session.get(AKN_URL, params), vigenza)
    if payload is None:  # the session expired: open a new one and try once more
        session.open(refresh=True)
        payload = _akn_payload(session.get(AKN_URL, params), vigenza)
    if payload is None:
        raise DownloadError(
            f"Normattiva returned an HTML page instead of Akoma Ntoso XML for "
            f"dataVigenza={vigenza:%Y%m%d}"
        )
    partial = path.with_name(path.name + ".part")
    partial.write_bytes(payload)
    partial.replace(path)
    logger.info("downloaded %s (%d bytes)", path.name, len(payload))
    return path


def download_snapshots(
    raw_dir: Path,
    *,
    delay_s: float = 2.0,
    force: bool = False,
    client: httpx.Client | None = None,
    today: date | None = None,
) -> list[Path]:
    """Download (or reuse) every snapshot of the act; returns the files sorted by date."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    today = today or date.today()
    owned = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S, follow_redirects=True)
    session = _PoliteSession(client, delay_s)
    fetched: set[date] = set()

    def ensure(vigenza: date) -> Path:
        path = snapshot_path(raw_dir, vigenza)
        if vigenza in fetched or (path.exists() and not force):
            return path
        fetched.add(vigenza)
        return _fetch(session, path, vigenza)

    try:
        latest = ensure(today)
        paths = [ensure(vigenza) for vigenza in vigenza_dates(read_events(latest), today)]
    finally:
        if owned:
            client.close()
    return sorted(set(paths))
