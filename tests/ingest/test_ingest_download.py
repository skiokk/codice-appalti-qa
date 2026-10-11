import argparse
from datetime import date
from pathlib import Path

import httpx
import pytest

from appalti_qa.ingest import cli_download, download_snapshots
from appalti_qa.ingest import commands as ingest_commands
from appalti_qa.ingest.akn import read_events
from appalti_qa.ingest.download import (
    AKN_URL,
    SESSION_URL,
    USER_AGENT,
    DownloadError,
    vigenza_dates,
)

TODAY = date(2025, 4, 10)
HTML_ERROR = b"<!DOCTYPE html><html><body>Sessione scaduta</body></html>"
HTML = {"content-type": "text/html"}


class FakeNormattiva:
    """Serves the session page and AKN exports; AKN requests need the session cookie."""

    def __init__(self, akn: bytes, *, akn_status: int = 200, html_always: bool = False) -> None:
        self.akn = akn
        self.akn_status = akn_status
        self.html_always = html_always
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        if url == str(httpx.URL(SESSION_URL)):
            return httpx.Response(200, headers={"Set-Cookie": "JSESSIONID=abc; Path=/"})
        if url.startswith(AKN_URL):
            if self.html_always or "JSESSIONID=abc" not in request.headers.get("cookie", ""):
                return httpx.Response(200, content=HTML_ERROR, headers=HTML)
            return httpx.Response(
                self.akn_status, content=self.akn, headers={"content-type": "application/xml"}
            )
        return httpx.Response(404)

    @property
    def akn_dates(self) -> list[str]:
        akn = [r for r in self.requests if str(r.url).startswith(AKN_URL)]
        return [r.url.params["dataVigenza"] for r in akn]

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


@pytest.fixture
def export(fixture_paths: list[Path]) -> bytes:
    return fixture_paths[2].read_bytes()  # lifecycle: original act + 11 amendments


def test_vigenza_dates(fixture_paths: list[Path]) -> None:
    dates = vigenza_dates(read_events(fixture_paths[2]), TODAY)
    assert dates[0] == date(2023, 4, 1)  # the day after the original publication
    assert date(2024, 12, 31) in dates and date(2025, 3, 14) in dates
    assert dates[-1] == TODAY
    assert len(dates) == 13
    assert vigenza_dates(read_events(fixture_paths[2]), date(2024, 1, 1))[-1] == date(2024, 1, 1)


def test_downloads_every_vigenza_date(tmp_path: Path, export: bytes) -> None:
    site = FakeNormattiva(export)
    paths = download_snapshots(tmp_path, delay_s=0, client=site.client(), today=TODAY)
    assert [p.name for p in paths][:2] == ["23G00044_20230401.xml", "23G00044_20230510.xml"]
    assert paths[-1].name == "23G00044_20250410.xml"
    assert len(paths) == 13
    assert all(p.read_bytes() == export for p in paths)
    assert str(site.requests[0].url) == str(httpx.URL(SESSION_URL))
    assert site.akn_dates[0] == "20250410"  # the latest text comes first
    assert len(site.akn_dates) == 13
    first_akn = site.requests[1]
    assert first_akn.url.params["codiceRedaz"] == "23G00044"
    assert first_akn.url.params["dataGU"] == "20230331"
    assert all(r.headers["user-agent"] == USER_AGENT for r in site.requests)
    assert not list(tmp_path.glob("*.part"))


def test_cached_files_are_not_downloaded_again(tmp_path: Path, export: bytes) -> None:
    download_snapshots(tmp_path, delay_s=0, client=FakeNormattiva(export).client(), today=TODAY)
    site = FakeNormattiva(export)
    paths = download_snapshots(tmp_path, delay_s=0, client=site.client(), today=TODAY)
    assert len(paths) == 13
    assert site.requests == []

    forced = FakeNormattiva(export)
    download_snapshots(tmp_path, delay_s=0, force=True, client=forced.client(), today=TODAY)
    assert len(forced.akn_dates) == len(set(forced.akn_dates)) == 13


def test_html_error_page_is_detected(tmp_path: Path, export: bytes) -> None:
    site = FakeNormattiva(export, html_always=True)
    with pytest.raises(DownloadError, match="HTML page"):
        download_snapshots(tmp_path, delay_s=0, client=site.client(), today=TODAY)
    sessions = [r for r in site.requests if str(r.url) == str(httpx.URL(SESSION_URL))]
    assert len(sessions) == 2  # the session is renewed once before giving up
    assert list(tmp_path.iterdir()) == []


def test_http_errors_and_non_akn_payloads(tmp_path: Path, export: bytes) -> None:
    with pytest.raises(DownloadError, match="HTTP 500"):
        download_snapshots(
            tmp_path, delay_s=0, client=FakeNormattiva(export, akn_status=500).client(), today=TODAY
        )
    with pytest.raises(DownloadError, match="Akoma Ntoso"):
        download_snapshots(
            tmp_path, delay_s=0, client=FakeNormattiva(b"<root/>").client(), today=TODAY
        )


def test_cli_download(monkeypatch, tmp_path: Path, capsys) -> None:
    args = argparse.Namespace(raw_dir=str(tmp_path), delay=0.5, force=True)
    calls = []

    def fake_download(raw_dir: Path, *, delay_s: float, force: bool) -> list[Path]:
        calls.append((raw_dir, delay_s, force))
        return [raw_dir / "23G00044_20250401.xml"]

    monkeypatch.setattr(ingest_commands, "download_snapshots", fake_download)
    assert cli_download(args) == 0
    assert calls == [(tmp_path, 0.5, True)]
    assert "1 snapshots available" in capsys.readouterr().out

    def failing(*_args, **_kwargs):
        raise DownloadError("HTTP 503 for dataVigenza=20250401")

    monkeypatch.setattr(ingest_commands, "download_snapshots", failing)
    assert cli_download(args) == 1
    assert "HTTP 503" in capsys.readouterr().err


def test_session_page_errors(tmp_path: Path, export: bytes) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(DownloadError, match="session: HTTP 503"):
        download_snapshots(tmp_path, delay_s=0, client=client, today=TODAY)


def test_requests_are_spaced(monkeypatch, tmp_path: Path, export: bytes) -> None:
    pauses: list[float] = []
    monkeypatch.setattr("appalti_qa.ingest.download.time.sleep", pauses.append)
    site = FakeNormattiva(export)
    download_snapshots(tmp_path, delay_s=2.0, client=site.client(), today=TODAY)
    assert len(pauses) == len(site.requests) - 1
    assert all(0 < pause <= 2.0 for pause in pauses)
