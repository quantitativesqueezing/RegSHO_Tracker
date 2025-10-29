"""
Script to fetch the latest NASDAQ and NYSE Regulation SHO
threshold securities lists and compare them with the prior day’s lists.

This program downloads the daily threshold list for Nasdaq from
NasdaqTrader and for each of the NYSE markets (NYSE, NYSE American,
and NYSE Arca) from NYSE’s public API.  It then determines which
tickers have been added to or removed from the threshold list
relative to the previous trading day.

Usage: simply run the script.  It will determine the appropriate
dates based on the current Eastern time.  If the latest list is not
available for the current day (for example, because the script is run
before the nightly update), it will automatically step back to the
most recent available date.

Requirements: Python 3.10+ (for zoneinfo), requests

Note:  This script only fetches the threshold lists for Nasdaq and
the three NYSE markets specified on the NYSE regulation site.  Other
exchanges (e.g., CBOE) publish their own lists separately and are
outside the scope of this program.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import calendar
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Set, Dict, List

try:
    # Python 3.9+ provides zoneinfo in the stdlib
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback: zoneinfo is available in backports.zoneinfo for older versions
    from backports.zoneinfo import ZoneInfo  # type: ignore

import requests


NASDAQ_URL_TEMPLATE = (
    "https://www.nasdaqtrader.com/dynamic/symdir/regsho/nasdaqth{date}.txt"
)
NYSE_API_ENDPOINT = (
    "https://www.nyse.com/api/regulatory/threshold-securities/download"
)

# Markets to query on the NYSE API.  According to the NYSE website,
# threshold securities lists are segregated by market.  Leaving the
# market parameter blank returns no useful data.
NYSE_MARKETS = ["NYSE", "NYSE American", "NYSE Arca"]


@dataclass(frozen=True)
class ThresholdList:
    """Container for a single day’s threshold securities data."""

    date: _dt.date
    tickers: Set[str]
    raw_lines: List[str]

    def diff(self, other: "ThresholdList") -> Dict[str, Set[str]]:
        """Return tickers added to, removed from, and unchanged relative to another list.

        This method compares the current list of tickers (`self`) against
        another day's list (`other`) and categorizes the symbols into three
        groups:

        * ``added``:  symbols present in ``self`` but absent from ``other``;
        * ``removed``:  symbols present in ``other`` but absent from ``self``;
        * ``unchanged``:  symbols present in both lists (persisted from
          ``other`` to ``self``).

        Parameters
        ----------
        other: ThresholdList
            The list to compare against.

        Returns
        -------
        dict
            A mapping with keys ``'added'``, ``'removed'``, and ``'unchanged'``.
            Each value is the set of symbols falling into that category.
        """
        # Tickers that newly appear on the threshold list compared to the previous day.
        added = self.tickers - other.tickers
        # Tickers that were present yesterday but no longer appear today.
        removed = other.tickers - self.tickers
        # Tickers that remain on the list across both days.
        unchanged = self.tickers & other.tickers
        return {"added": added, "removed": removed, "unchanged": unchanged}


def fetch_nasdaq_list(date: _dt.date) -> ThresholdList | None:
    """Fetch the Nasdaq threshold list for the given date.

    Parameters
    ----------
    date : datetime.date
        The settlement/trade date.

    Returns
    -------
    ThresholdList | None
        A ThresholdList object on success, or None if the file is not found.
    """
    url_date = date.strftime("%Y%m%d")
    url = NASDAQ_URL_TEMPLATE.format(date=url_date)
    try:
        resp = requests.get(url, timeout=10)
    except Exception as exc:
        raise RuntimeError(f"Network error fetching Nasdaq list: {exc}") from exc
    if resp.status_code == 200:
        lines = resp.text.strip().splitlines()
        # The first line is the header; subsequent lines may include a trailing timestamp.
        tickers: Set[str] = set()
        parsed_lines: List[str] = []
        for line in lines[1:]:
            fields = line.split("|")
            # Some lines may just contain a timestamp (e.g., 20250806230015).  Skip if no symbol.
            if fields and fields[0].isalpha():
                tickers.add(fields[0])
                parsed_lines.append(line)
        return ThresholdList(date=date, tickers=tickers, raw_lines=parsed_lines)
    elif resp.status_code == 404:
        # File not found; return None so the caller can step back a day.
        return None
    else:
        raise RuntimeError(
            f"Unexpected HTTP status {resp.status_code} fetching Nasdaq list for {date}: {resp.text}"
        )


def fetch_nyse_list(date: _dt.date, markets: Iterable[str] = NYSE_MARKETS) -> ThresholdList:
    """Fetch the combined NYSE threshold list for the given date across specified markets.

    Parameters
    ----------
    date : datetime.date
        The settlement/trade date.
    markets : Iterable[str]
        An iterable of market identifiers (e.g., "NYSE", "NYSE American", "NYSE Arca").

    Returns
    -------
    ThresholdList
        A ThresholdList object containing the combined tickers from all requested markets.

    Raises
    ------
    RuntimeError
        If any request fails with a non‑200 HTTP status other than 404.
    """
    selected_date = date.strftime("%Y-%m-%d")
    combined_tickers: Set[str] = set()
    combined_lines: List[str] = []
    for market in markets:
        params = {"selectedDate": selected_date, "market": market}
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
        try:
            resp = requests.get(NYSE_API_ENDPOINT, params=params, headers=headers, timeout=10)
        except Exception as exc:
            raise RuntimeError(f"Network error fetching NYSE list for {market}: {exc}") from exc
        if resp.status_code == 200:
            text = resp.text.strip()
            lines = text.splitlines()
            # Skip header; last line may be timestamp.
            for line in lines[1:]:
                fields = line.split("|")
                if fields and fields[0].isalpha():
                    combined_tickers.add(fields[0])
                    combined_lines.append(line)
        elif resp.status_code == 404:
            # 404 would indicate no data for the date; skip this market.
            continue
        elif resp.status_code == 406:
            # The NYSE API returns 406 Not Acceptable if Accept header is not acceptable; this
            # should not occur given our headers.  Handle explicitly for clarity.
            raise RuntimeError(
                f"NYSE API returned 406 Not Acceptable for {selected_date} market {market}."
            )
        else:
            raise RuntimeError(
                f"Unexpected HTTP status {resp.status_code} fetching NYSE list for {selected_date} market {market}: {resp.text}"
            )
    return ThresholdList(date=date, tickers=combined_tickers, raw_lines=combined_lines)


def get_latest_dates(reference: _dt.date, max_lookback: int = 7) -> tuple[_dt.date, _dt.date]:
    """Determine the latest two dates for which Nasdaq data files exist.

    Nasdaq publishes one file per settlement date.  This helper walks
    backwards from the reference date to find the most recent available
    file and the file from the previous day.  A maximum lookback of
    `max_lookback` days is used to avoid looping indefinitely when
    markets are closed (e.g., weekends or holidays).

    Parameters
    ----------
    reference : datetime.date
        The starting date for the search (typically today).
    max_lookback : int
        Maximum number of days to search backwards.

    Returns
    -------
    tuple of two datetime.date
        A pair `(latest_date, previous_date)` corresponding to the most
        recent file and the file immediately before it.  Raises an
        exception if suitable files are not found within the lookback.
    """
    today = reference
    latest: _dt.date | None = None
    previous: _dt.date | None = None
    # Iterate backward up to max_lookback days to find two dates that have non‑empty Nasdaq lists.
    for offset in range(max_lookback):
        candidate = today - _dt.timedelta(days=offset)
        # Fetch the Nasdaq list for this candidate date
        result = fetch_nasdaq_list(candidate)
        # Skip dates where there is no file or the file contains no tickers
        if result is None or not result.tickers:
            continue
        if latest is None:
            # First valid date with threshold securities becomes the 'latest' date
            latest = candidate
        elif previous is None:
            # Second valid date becomes the 'previous' date
            previous = candidate
            break
    if latest is None or previous is None:
        raise RuntimeError(
            f"Could not find two consecutive Nasdaq threshold files with data within {max_lookback} days starting from {reference}."
        )
    return latest, previous


def load_env_file(path: Path) -> None:
    """Populate os.environ with values from a simple KEY=VALUE .env file."""
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Failed to read environment file {path}: {exc}") from exc
    index = 0
    total = len(lines)
    while index < total:
        raw_line = lines[index]
        stripped = raw_line.strip()
        index += 1
        if not stripped or stripped.startswith("#") or "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.startswith("[") and not value.rstrip().endswith("]"):
            buffer = [value]
            while index < total:
                continuation = lines[index].strip()
                buffer.append(continuation)
                index += 1
                if continuation.endswith("]"):
                    break
            value = "\n".join(buffer)
        if (
            (value.startswith('"') and value.endswith('"'))
            or (value.startswith("'") and value.endswith("'"))
        ):
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def format_human_date(date: _dt.date) -> str:
    """Return dates such as 'Tuesday, 10/28/2025'."""
    return date.strftime("%A, %m/%d/%Y")


def format_display_date(date: _dt.date) -> str:
    """Return dates in display format with a trailing parenthesis."""
    return f"{format_human_date(date)})"


def format_symbols_for_embed(symbols: Iterable[str]) -> str:
    """Render a collection of tickers for Discord embed fields."""
    sorted_symbols = sorted(symbols)
    return ", ".join(sorted_symbols) if sorted_symbols else "None"


def load_post_log(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_post_log(path: Path, data: Dict[str, str]) -> None:
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Failed to write webhook post log {path}: {exc}") from exc


def apply_template_replacements(template: object, replacements: Dict[str, str]) -> object:
    """Recursively replace placeholder strings within a template structure."""
    if isinstance(template, str):
        result = template
        for placeholder, value in replacements.items():
            if placeholder in result:
                result = result.replace(placeholder, value)
        return result
    if isinstance(template, list):
        return [apply_template_replacements(item, replacements) for item in template]
    if isinstance(template, dict):
        return {
            key: apply_template_replacements(value, replacements) for key, value in template.items()
        }
    return template


US_MARKET_HOLIDAYS_CACHE: Dict[int, Set[_dt.date]] = {}


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> _dt.date:
    first = _dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    day = 1 + offset + (n - 1) * 7
    return _dt.date(year, month, day)


def _last_weekday_of_month(year: int, month: int, weekday: int) -> _dt.date:
    last_day = calendar.monthrange(year, month)[1]
    last_date = _dt.date(year, month, last_day)
    offset = (last_date.weekday() - weekday) % 7
    return last_date - _dt.timedelta(days=offset)


def _calculate_easter(year: int) -> _dt.date:
    """Return the Gregorian Easter date using Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return _dt.date(year, month, day)


def _observed_holiday(date: _dt.date) -> _dt.date:
    if date.weekday() == 5:  # Saturday
        return date - _dt.timedelta(days=1)
    if date.weekday() == 6:  # Sunday
        return date + _dt.timedelta(days=1)
    return date


def _register_holiday(date: _dt.date) -> None:
    US_MARKET_HOLIDAYS_CACHE.setdefault(date.year, set()).add(date)


def _populate_market_holidays(year: int) -> None:
    years_to_generate = {year - 1, year, year + 1}
    for target_year in years_to_generate:
        if target_year in US_MARKET_HOLIDAYS_CACHE:
            continue
        holidays_for_year: Set[_dt.date] = set()

        def add(date: _dt.date) -> None:
            holidays_for_year.add(date)
            _register_holiday(date)

        # New Year's Day (observed)
        new_year = _dt.date(target_year, 1, 1)
        add(_observed_holiday(new_year))
        # Martin Luther King Jr. Day (3rd Monday of January)
        add(_nth_weekday_of_month(target_year, 1, 0, 3))
        # Presidents' Day (3rd Monday of February)
        add(_nth_weekday_of_month(target_year, 2, 0, 3))
        # Good Friday (two days before Easter Sunday)
        add(_calculate_easter(target_year) - _dt.timedelta(days=2))
        # Memorial Day (last Monday of May)
        add(_last_weekday_of_month(target_year, 5, 0))
        # Juneteenth (observed)
        add(_observed_holiday(_dt.date(target_year, 6, 19)))
        # Independence Day (observed)
        add(_observed_holiday(_dt.date(target_year, 7, 4)))
        # Labor Day (first Monday of September)
        add(_nth_weekday_of_month(target_year, 9, 0, 1))
        # Thanksgiving Day (fourth Thursday of November)
        add(_nth_weekday_of_month(target_year, 11, 3, 4))
        # Christmas Day (observed)
        add(_observed_holiday(_dt.date(target_year, 12, 25)))

        US_MARKET_HOLIDAYS_CACHE.setdefault(target_year, set()).update(holidays_for_year)


def is_market_holiday(date: _dt.date) -> bool:
    _populate_market_holidays(date.year)
    if date.weekday() >= 5:
        return True
    return date in US_MARKET_HOLIDAYS_CACHE.get(date.year, set())


def next_trading_day(date: _dt.date) -> _dt.date:
    candidate = date
    while True:
        candidate += _dt.timedelta(days=1)
        if is_market_holiday(candidate):
            continue
        return candidate


@dataclass(frozen=True)
class WebhookTarget:
    url: str
    thread_id: str | None = None


def _parse_custom_mapping(text: str) -> List[WebhookTarget]:
    """Parse square-bracketed entries that use the \"url\" => \"thread\" format."""
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        stripped = stripped[1:-1]
    targets: List[WebhookTarget] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip().rstrip(",")
        if not line or "=>" not in line:
            continue
        left, right = line.split("=>", 1)
        url = left.strip().strip('"').strip("'")
        thread_raw = right.strip().strip('"').strip("'")
        thread_id = thread_raw or None
        if url:
            targets.append(WebhookTarget(url=url, thread_id=thread_id))
    return targets


def _coerce_webhook_targets(value: str, *, require_array: bool = False) -> List[WebhookTarget]:
    """Coerce CLI/env inputs into webhook targets with optional thread IDs."""
    value = value.strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = None
    targets: List[WebhookTarget] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                url = item.get("url") or item.get("webhook_url") or item.get("target")
                thread = item.get("thread_id") or item.get("thread")
                if url:
                    url_str = str(url).strip()
                    if url_str:
                        thread_str = str(thread).strip() if thread else None
                        targets.append(WebhookTarget(url=url_str, thread_id=thread_str or None))
            elif isinstance(item, (list, tuple)) and item:
                url_str = str(item[0]).strip()
                if url_str:
                    thread_val = item[1] if len(item) > 1 else None
                    thread_str = str(thread_val).strip() if thread_val else None
                    targets.append(WebhookTarget(url=url_str, thread_id=thread_str or None))
            elif isinstance(item, str):
                url_str = item.strip()
                if url_str:
                    targets.append(WebhookTarget(url=url_str))
        return targets
    if isinstance(parsed, dict):
        for key, value in parsed.items():
            url_str = str(key).strip()
            if not url_str:
                continue
            thread_str = str(value).strip() if value else None
            targets.append(WebhookTarget(url=url_str, thread_id=thread_str or None))
        return targets
    if isinstance(parsed, str) and not require_array:
        url_str = parsed.strip()
        return [WebhookTarget(url=url_str)] if url_str else []
    if "=>" in value:
        targets = _parse_custom_mapping(value)
        if targets:
            return targets
    if not require_array:
        if "|" in value:
            url_part, thread_part = value.split("|", 1)
            url_str = url_part.strip()
            if url_str:
                thread_str = thread_part.strip()
                return [WebhookTarget(url=url_str, thread_id=thread_str or None)]
        if value:
            return [WebhookTarget(url=value)]
    raise ValueError("expected a JSON array/object or associative mapping of webhook URLs")


def parse_webhook_inputs(values: Iterable[str], *, require_array: bool = False) -> List[WebhookTarget]:
    """Normalize CLI/environment webhook strings into a list of WebhookTarget."""
    targets: List[WebhookTarget] = []
    for value in values:
        if not value:
            continue
        targets.extend(_coerce_webhook_targets(value, require_array=require_array))
    return targets


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and compare the latest RegSHO Threshold securities lists."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Emit the comparison report as JSON.",
    )
    parser.add_argument(
        "--post-combined",
        action="store_true",
        dest="post_combined",
        help="Post the combined exchange JSON payload to a Discord webhook.",
    )
    parser.add_argument(
        "--webhook-url",
        dest="webhook_urls",
        action="append",
        help="Discord webhook URL optionally followed by '|thread_id'. Repeat to send to multiple. Defaults to arrays in DISCORD_WEBHOOK_URL / DISCORD_WEBHOOK_TEST.",
    )
    parser.add_argument(
        "--webhook-template",
        dest="webhook_template",
        default=os.getenv("DISCORD_WEBHOOK_TEMPLATE"),
        help="Path to Discord webhook JSON template. Defaults to discord_webhook_object.json beside the script when omitted.",
    )
    return parser


def main(argv: List[str] | None = None) -> None:
    load_env_file(Path(__file__).with_name(".env"))
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    webhook_targets: List[WebhookTarget] = []
    eastern_zone = ZoneInfo("America/New_York")
    now_eastern = _dt.datetime.now(eastern_zone).date()
    if args.post_combined:
        if args.webhook_urls:
            try:
                webhook_targets.extend(parse_webhook_inputs(args.webhook_urls))
            except ValueError as exc:
                raise RuntimeError(f"Invalid --webhook-url value: {exc}") from exc
        else:
            array_keys = ("DISCORD_WEBHOOK_URL", "DISCORD_WEBHOOK_TEST")
            for key in array_keys:
                raw = os.getenv(key)
                if not raw:
                    continue
                try:
                    webhook_targets.extend(parse_webhook_inputs([raw], require_array=True))
                except ValueError as exc:
                    raise RuntimeError(
                        f"Environment variable {key} must be a JSON array of webhook URLs: {exc}"
                    ) from exc
            if not webhook_targets:
                legacy_keys = (
                    "DISCORD_WEBHOOK_URLS",
                    "DISCORD_WEBHOOK_URL",
                    "DISCORD_WEBHOOK_TEST_URL",
                )
                for key in legacy_keys:
                    raw = os.getenv(key)
                    if not raw:
                        continue
                    try:
                        webhook_targets.extend(parse_webhook_inputs([raw]))
                    except ValueError as exc:
                        raise RuntimeError(f"Failed to parse {key}: {exc}") from exc
        if not webhook_targets:
            print("Error: --post-combined requires at least one Discord webhook URL.", file=sys.stderr)
            sys.exit(1)
    template_path: Path | None = None
    if args.post_combined:
        if args.webhook_template:
            template_path = Path(args.webhook_template)
        else:
            template_path = Path(__file__).with_name("discord_webhook_object.json")
        if not template_path.exists():
            raise RuntimeError(f"Discord webhook template not found: {template_path}")
        post_log_path = Path(__file__).with_name("webhook_post_log.json")
        post_log = load_post_log(post_log_path)
        post_log_updated = False
        today_log_token = now_eastern.isoformat()
    # Determine current date in America/New_York.  The lists update in the late
    # evening (11 pm – midnight) Eastern time, so using the local date provides a
    # sensible default.
    try:
        latest_date, previous_date = get_latest_dates(now_eastern)
    except Exception as exc:
        print(f"Error locating Nasdaq lists: {exc}")
        sys.exit(1)
    # Fetch lists
    latest_nasdaq = fetch_nasdaq_list(latest_date)
    previous_nasdaq = fetch_nasdaq_list(previous_date)
    # The fetch functions can return None only when not found; by this point
    # latest and previous have been verified to exist.
    assert latest_nasdaq is not None and previous_nasdaq is not None

    latest_nyse = fetch_nyse_list(latest_date)
    previous_nyse = fetch_nyse_list(previous_date)

    # Compute differences
    nasdaq_diff = latest_nasdaq.diff(previous_nasdaq)
    nyse_diff = latest_nyse.diff(previous_nyse)

    def as_sorted_list(symbols: Iterable[str]) -> List[str]:
        return sorted(symbols)

    nasdaq_current = latest_nasdaq.tickers
    nyse_current = latest_nyse.tickers
    combined_current = nasdaq_current | nyse_current
    combined_added = nasdaq_diff["added"] | nyse_diff["added"]
    combined_removed = nasdaq_diff["removed"] | nyse_diff["removed"]

    display_latest_date = next_trading_day(latest_date)
    display_previous_date = next_trading_day(previous_date)

    human_latest = format_display_date(display_latest_date)
    human_previous = format_display_date(display_previous_date)

    payload = {
        "latest_date": human_latest,
        "previous_date": human_previous,
        "nasdaq": {
            "current": as_sorted_list(nasdaq_current),
            "added": as_sorted_list(nasdaq_diff["added"]),
            "removed": as_sorted_list(nasdaq_diff["removed"]),
        },
        "nyse": {
            "current": as_sorted_list(nyse_current),
            "added": as_sorted_list(nyse_diff["added"]),
            "removed": as_sorted_list(nyse_diff["removed"]),
        },
        "combined": {
            "current": as_sorted_list(combined_current),
            "added": as_sorted_list(combined_added),
            "removed": as_sorted_list(combined_removed),
        },
    }

    replacements = {
        "{previous_date}": human_previous,
        "{latest_date}": human_latest,
        "{nasdaq.current}": format_symbols_for_embed(payload["nasdaq"]["current"]),
        "{nasdaq.added}": format_symbols_for_embed(payload["nasdaq"]["added"]),
        "{nasdaq.removed}": format_symbols_for_embed(payload["nasdaq"]["removed"]),
        "{nyse.current}": format_symbols_for_embed(payload["nyse"]["current"]),
        "{nyse.added}": format_symbols_for_embed(payload["nyse"]["added"]),
        "{nyse.removed}": format_symbols_for_embed(payload["nyse"]["removed"]),
    }

    if args.output_json:
        print(json.dumps(payload, indent=2))

    if args.post_combined:
        assert template_path is not None
        try:
            template_text = template_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Failed to read Discord webhook template: {exc}") from exc
        try:
            template_json = json.loads(template_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse Discord webhook template {template_path}: {exc}"
            ) from exc
        webhook_payload = apply_template_replacements(template_json, replacements)
        for target in webhook_targets:
            log_key = f"{target.url}|{target.thread_id or ''}"
            if post_log.get(log_key) == today_log_token:
                print(f"Skipping Discord webhook {target.url} (already posted today).")
                continue
            params = {"thread_id": target.thread_id} if target.thread_id else None
            try:
                response = requests.post(
                    target.url,
                    json=webhook_payload,
                    params=params,
                    timeout=10,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to post to Discord webhook {target.url}: {exc}") from exc
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Discord webhook {target.url} returned {response.status_code}: {response.text}"
                )
            post_log[log_key] = today_log_token
            post_log_updated = True
        if post_log_updated:
            save_post_log(post_log_path, post_log)

    if args.output_json:
        return

    # Report results
    print(f"RegSHO Threshold List Comparison for {human_latest} vs {human_previous}\n")

    def format_symbol_list(symbols: Iterable[str]) -> str:
        """Return a comma‑separated string of sorted symbols or 'None' if empty."""
        return ", ".join(sorted(symbols)) if symbols else "None"

    # NASDAQ results
    print("NASDAQ:")
    print(f"  Current: ({len(nasdaq_diff['unchanged']) + len(nasdaq_diff['added']):2d}): {format_symbol_list(nasdaq_diff['unchanged']), format_symbol_list(nasdaq_diff['added'])}\n")
    print(f"  Added:    ({len(nasdaq_diff['added']):2d} ): {format_symbol_list(nasdaq_diff['added'])}")
    print(f"  Removed:  ({len(nasdaq_diff['removed']):2d} ): {format_symbol_list(nasdaq_diff['removed'])}\n\n")

    # NYSE results (all markets combined)
    print("NYSE (combined markets):")
    print(f"  Current: ({len(nyse_diff['unchanged']) + len(nyse_diff['added']):2d}): {format_symbol_list(nyse_diff['unchanged']), format_symbol_list(nyse_diff['added'])}\n")
    print(f"  Added:    ({len(nyse_diff['added']):2d} ): {format_symbol_list(nyse_diff['added'])}")
    print(f"  Removed:  ({len(nyse_diff['removed']):2d} ): {format_symbol_list(nyse_diff['removed'])}\n\n")


if __name__ == "__main__":
    main()
