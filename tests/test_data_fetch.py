"""CelesTrak and Space-Track clients: caching, throttling, auth handling.

No network. ``requests`` is monkeypatched, the caches are redirected into
``tmp_path``, and the rate limiter is driven with a fake clock, so the
sliding-window arithmetic is tested without anyone sleeping.
"""
from __future__ import annotations

import json
import time

import pytest
import requests

from orbital.sgp4tools import celestrak, spacetrack
from orbital.sgp4tools.spacetrack import RateLimiter, SpaceTrack, credentials, load_env

ISS_TEXT = (
    "ISS (ZARYA)\n"
    "1 25544U 98067A   24117.51782528  .00016717  00000-0  30074-3 0  9992\n"
    "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49309239448471\n"
)


class FakeResponse:
    def __init__(self, text="", status=200):
        self.text = text
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


@pytest.fixture
def tle_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(celestrak, "CACHE_DIR", tmp_path / "tle_cache")
    return tmp_path / "tle_cache"


class TestCelesTrakCaching:
    def test_fetches_then_serves_from_cache(self, tle_cache, monkeypatch):
        calls = []

        def fake_get(url, params=None, **kwargs):
            calls.append(params)
            return FakeResponse(ISS_TEXT)

        monkeypatch.setattr(celestrak.requests, "get", fake_get)
        assert celestrak.fetch_gp(catnr=25544) == ISS_TEXT
        assert celestrak.fetch_gp(catnr=25544) == ISS_TEXT      # second: cached
        assert len(calls) == 1
        assert calls[0] == {"CATNR": "25544", "FORMAT": "tle"}

    def test_force_refetches_even_when_fresh(self, tle_cache, monkeypatch):
        calls = []
        monkeypatch.setattr(celestrak.requests, "get",
                            lambda *a, **k: (calls.append(1), FakeResponse(ISS_TEXT))[1])
        celestrak.fetch_gp(catnr=25544)
        celestrak.fetch_gp(catnr=25544, force=True)
        assert len(calls) == 2

    def test_stale_cache_triggers_a_refetch(self, tle_cache, monkeypatch):
        calls = []
        monkeypatch.setattr(celestrak.requests, "get",
                            lambda *a, **k: (calls.append(1), FakeResponse(ISS_TEXT))[1])
        celestrak.fetch_gp(catnr=25544)
        path = next(tle_cache.iterdir())
        import os
        old = time.time() - 3 * 3600
        os.utime(path, (old, old))
        celestrak.fetch_gp(catnr=25544, max_age_hours=2.0)
        assert len(calls) == 2

    def test_stale_cache_beats_a_network_failure(self, tle_cache, monkeypatch):
        """CelesTrak throttles aggressively, so a stale copy is preferable to
        failing the run.
        """
        monkeypatch.setattr(celestrak.requests, "get", lambda *a, **k: FakeResponse(ISS_TEXT))
        celestrak.fetch_gp(catnr=25544)

        def boom(*args, **kwargs):
            raise requests.ConnectionError("no network")

        monkeypatch.setattr(celestrak.requests, "get", boom)
        assert celestrak.fetch_gp(catnr=25544, force=True) == ISS_TEXT

    def test_network_failure_with_no_cache_raises(self, tle_cache, monkeypatch):
        def boom(*args, **kwargs):
            raise requests.ConnectionError("no network")

        monkeypatch.setattr(celestrak.requests, "get", boom)
        with pytest.raises(requests.ConnectionError):
            celestrak.fetch_gp(catnr=25544)

    def test_throttle_page_is_rejected_not_cached(self, tle_cache, monkeypatch):
        """A throttle or empty match arrives as HTTP 200 with plain text."""
        monkeypatch.setattr(celestrak.requests, "get",
                            lambda *a, **k: FakeResponse("Max daily requests exceeded"))
        with pytest.raises(RuntimeError, match="no TLE data"):
            celestrak.fetch_gp(catnr=25544)
        assert not tle_cache.exists() or not list(tle_cache.iterdir())

    def test_catnr_and_group_are_mutually_exclusive(self, tle_cache):
        with pytest.raises(ValueError, match="exactly one"):
            celestrak.fetch_gp()
        with pytest.raises(ValueError, match="exactly one"):
            celestrak.fetch_gp(catnr=25544, group="stations")

    def test_group_queries_use_the_group_parameter(self, tle_cache, monkeypatch):
        seen = {}

        def fake_get(url, params=None, **kwargs):
            seen.update(params)
            return FakeResponse(ISS_TEXT)

        monkeypatch.setattr(celestrak.requests, "get", fake_get)
        celestrak.fetch_gp(group="stations")
        assert seen == {"GROUP": "stations", "FORMAT": "tle"}

    def test_fetch_tle_parses_a_single_object(self, tle_cache, monkeypatch):
        monkeypatch.setattr(celestrak.requests, "get", lambda *a, **k: FakeResponse(ISS_TEXT))
        t = celestrak.fetch_tle(25544)
        assert t.catalog_number == 25544
        assert t.name == "ISS (ZARYA)"

    def test_separate_queries_use_separate_cache_files(self, tle_cache, monkeypatch):
        monkeypatch.setattr(celestrak.requests, "get", lambda *a, **k: FakeResponse(ISS_TEXT))
        celestrak.fetch_gp(catnr=25544)
        celestrak.fetch_gp(group="stations")
        assert len(list(tle_cache.iterdir())) == 2


class TestRateLimiter:
    def test_allows_the_first_burst_without_waiting(self, monkeypatch):
        slept = []
        monkeypatch.setattr(spacetrack.time, "sleep", slept.append)
        limiter = RateLimiter(per_minute=5, per_hour=100)
        for _ in range(5):
            limiter.acquire(verbose=False)
        assert slept == []

    def test_waits_when_the_minute_cap_is_reached(self, monkeypatch):
        clock = {"t": 1000.0}
        slept = []
        monkeypatch.setattr(spacetrack.time, "time", lambda: clock["t"])
        monkeypatch.setattr(spacetrack.time, "sleep",
                            lambda s: (slept.append(s), clock.__setitem__("t", clock["t"] + s)))
        limiter = RateLimiter(per_minute=3, per_hour=100)
        for _ in range(4):
            limiter.acquire(verbose=False)
        assert len(slept) == 1
        assert slept[0] == pytest.approx(60.0, abs=1.0)

    def test_the_window_slides(self, monkeypatch):
        clock = {"t": 1000.0}
        slept = []
        monkeypatch.setattr(spacetrack.time, "time", lambda: clock["t"])
        monkeypatch.setattr(spacetrack.time, "sleep", slept.append)
        limiter = RateLimiter(per_minute=2, per_hour=100)
        limiter.acquire(verbose=False)
        limiter.acquire(verbose=False)
        clock["t"] += 61.0            # the first two have aged out
        limiter.acquire(verbose=False)
        assert slept == []

    def test_hourly_cap_also_applies(self, monkeypatch):
        clock = {"t": 0.0}
        slept = []
        monkeypatch.setattr(spacetrack.time, "time", lambda: clock["t"])

        def fake_sleep(s):
            slept.append(s)
            clock["t"] += s

        monkeypatch.setattr(spacetrack.time, "sleep", fake_sleep)
        limiter = RateLimiter(per_minute=1000, per_hour=3)
        for _ in range(4):
            limiter.acquire(verbose=False)
            clock["t"] += 1.0
        assert slept and max(slept) > 1000.0        # had to wait out the hour

    def test_defaults_stay_under_the_published_limits(self):
        """Published caps are 30/min and 300/hour; ours must be lower."""
        limiter = RateLimiter()
        assert limiter.per_minute < 30
        assert limiter.per_hour < 300

    def test_records_every_call(self, monkeypatch):
        monkeypatch.setattr(spacetrack.time, "sleep", lambda s: None)
        limiter = RateLimiter(per_minute=100, per_hour=1000)
        for _ in range(7):
            limiter.acquire(verbose=False)
        assert len(limiter.calls) == 7


class TestCredentials:
    def test_env_file_parsing(self, tmp_path):
        path = tmp_path / ".env"
        path.write_text(
            "# a comment\n\n"
            "SPACETRACK_IDENTITY=user@example.com\n"
            'SPACETRACK_PASSWORD="quoted secret"\n'
            "MALFORMED\n"
        )
        env = load_env(path)
        assert env == {"SPACETRACK_IDENTITY": "user@example.com",
                       "SPACETRACK_PASSWORD": "quoted secret"}

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_env(tmp_path / "absent") == {}

    def test_environment_takes_precedence(self, monkeypatch, tmp_path):
        monkeypatch.setattr(spacetrack, "ENV_PATH", tmp_path / "absent")
        monkeypatch.setenv("SPACETRACK_IDENTITY", "env@example.com")
        monkeypatch.setenv("SPACETRACK_PASSWORD", "s3cret")
        assert credentials() == ("env@example.com", "s3cret")

    def test_missing_credentials_explain_the_fix(self, monkeypatch, tmp_path):
        monkeypatch.setattr(spacetrack, "ENV_PATH", tmp_path / "absent")
        monkeypatch.delenv("SPACETRACK_IDENTITY", raising=False)
        monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
        with pytest.raises(RuntimeError, match=".env.example"):
            credentials()

    def test_placeholder_password_is_rejected(self, monkeypatch, tmp_path):
        """Copying .env.example without editing it must fail loudly."""
        monkeypatch.setattr(spacetrack, "ENV_PATH", tmp_path / "absent")
        monkeypatch.setenv("SPACETRACK_IDENTITY", "user@example.com")
        monkeypatch.setenv("SPACETRACK_PASSWORD", "your-password-here")
        with pytest.raises(RuntimeError, match="placeholder"):
            credentials()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(spacetrack, "CACHE_DIR", tmp_path / "st_cache")
    monkeypatch.setattr(spacetrack, "ENV_PATH", tmp_path / "absent")
    monkeypatch.setenv("SPACETRACK_IDENTITY", "user@example.com")
    monkeypatch.setenv("SPACETRACK_PASSWORD", "s3cret")
    st = SpaceTrack()
    monkeypatch.setattr(st.limiter, "acquire", lambda verbose=True: None)
    return st


class TestSpaceTrackClient:
    def test_url_building(self, client):
        url = client._build_path("gp", {"NORAD_CAT_ID": [25544, 48274], "limit": 5,
                                        "skipped": None}, "json")
        assert url.endswith("/class/gp/NORAD_CAT_ID/25544,48274/limit/5/format/json")
        assert "skipped" not in url

    def test_login_detects_a_failure_returned_as_http_200(self, client, monkeypatch):
        monkeypatch.setattr(client.session, "post",
                            lambda *a, **k: FakeResponse("Failed: login invalid"))
        with pytest.raises(RuntimeError, match="login failed"):
            client.login()

    def test_query_caches_and_reuses(self, client, monkeypatch):
        posts, gets = [], []
        monkeypatch.setattr(client.session, "post",
                            lambda *a, **k: (posts.append(1), FakeResponse("ok"))[1])
        payload = json.dumps([{"NORAD_CAT_ID": "25544"}])
        monkeypatch.setattr(client.session, "get",
                            lambda *a, **k: (gets.append(1), FakeResponse(payload))[1])

        first = client.query("gp", NORAD_CAT_ID=25544)
        second = client.query("gp", NORAD_CAT_ID=25544)
        assert first == second == [{"NORAD_CAT_ID": "25544"}]
        assert len(gets) == 1               # served from cache the second time
        assert len(posts) == 1              # logged in once

    def test_expired_session_is_retried_once(self, client, monkeypatch):
        monkeypatch.setattr(client.session, "post", lambda *a, **k: FakeResponse("ok"))
        responses = [FakeResponse("denied", status=401), FakeResponse("[]")]
        monkeypatch.setattr(client.session, "get", lambda *a, **k: responses.pop(0))
        assert client.query("gp", NORAD_CAT_ID=1) == []
        assert responses == []

    def test_non_json_format_returns_text(self, client, monkeypatch):
        monkeypatch.setattr(client.session, "post", lambda *a, **k: FakeResponse("ok"))
        monkeypatch.setattr(client.session, "get", lambda *a, **k: FakeResponse("1 25544U"))
        assert client.query("gp", fmt="tle", NORAD_CAT_ID=1) == "1 25544U"

    def test_context_manager_logs_in_and_out(self, client, monkeypatch):
        events = []
        monkeypatch.setattr(client.session, "post",
                            lambda *a, **k: (events.append("login"), FakeResponse("ok"))[1])
        monkeypatch.setattr(client.session, "get",
                            lambda *a, **k: (events.append("logout"), FakeResponse("ok"))[1])
        with client:
            pass
        assert events == ["login", "logout"]
