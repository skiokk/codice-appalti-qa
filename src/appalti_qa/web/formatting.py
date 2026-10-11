"""Italian formatting of dates, validity intervals and numbers for the templates."""

from __future__ import annotations

from datetime import date, timedelta

_MONTHS = (
    "gennaio",
    "febbraio",
    "marzo",
    "aprile",
    "maggio",
    "giugno",
    "luglio",
    "agosto",
    "settembre",
    "ottobre",
    "novembre",
    "dicembre",
)


def format_date(day: date) -> str:
    """2024-12-31 -> '31/12/2024'."""
    return f"{day:%d/%m/%Y}"


def format_date_long(day: date) -> str:
    """2024-12-31 -> '31 dicembre 2024'."""
    return f"{day.day} {_MONTHS[day.month - 1]} {day.year}"


def validity(valid_from: date, valid_to: date | None) -> str:
    """The validity interval in words (`valid_to` is exclusive: the last day is the one before)."""
    if valid_to is None:
        return f"In vigore dal {format_date(valid_from)}"
    last = valid_to - timedelta(days=1)
    return f"In vigore dal {format_date(valid_from)} al {format_date(last)}"


def format_int(value: float) -> str:
    """12345 -> '12.345'."""
    return f"{round(value):,}".replace(",", ".")


def format_decimal(value: float, digits: int = 1) -> str:
    """0.7125 -> '0,7' (digits=1)."""
    return f"{value:.{digits}f}".replace(".", ",")


def format_percent(ratio: float) -> str:
    """0.725 -> '72,5%'."""
    return f"{format_decimal(ratio * 100)}%"


def format_seconds(milliseconds: int) -> str:
    """1234 -> '1,2 s'."""
    return f"{format_decimal(milliseconds / 1000)} s"
