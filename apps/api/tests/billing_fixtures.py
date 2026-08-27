"""PayApp doubles for the API suite.

In a module with a distinct name rather than in `conftest`, following the
same convention as `plan_fixtures` and `asset_fixtures`: a whole-repository
pytest run puts several packages' `conftest` modules on one import path,
and `from conftest import x` resolves to whichever was imported first.

The important thing here is what these fixtures make impossible. The fake
client is installed on `app.state` before any test runs, so there is no
code path in the suite that constructs a real `HttpPayAppClient` — an
automated run cannot reach api.payapp.kr even if someone puts real
credentials in the environment. That is deliberate: a test that could
call `rebillRegist` for real could register a live recurring contract
against a real person's phone number.

The credentials below are obviously fake and exist so the notification
endpoints have something to validate against. They are not secrets and
never were.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from luber_billing.payapp.fake import FakePayAppClient

from luber_api.settings import get_settings

#: Recognisable as test values at a glance, so nobody mistakes a failing
#: assertion for a leaked credential.
TEST_PAYAPP_USERID = "boorda-test"
TEST_PAYAPP_LINKKEY = "test-linkkey"
TEST_PAYAPP_LINKVAL = "test-linkval"


@pytest.fixture(autouse=True)
def payapp_settings(monkeypatch: pytest.MonkeyPatch):
    """Configure billing for every API test, and clear the settings cache.

    Autouse because `billing_available()` gates the notification
    endpoints: without credentials they answer 503, and a test asserting
    a forged notification is refused would pass for the wrong reason.
    """
    monkeypatch.setenv("PAYAPP_USERID", TEST_PAYAPP_USERID)
    monkeypatch.setenv("PAYAPP_LINKKEY", TEST_PAYAPP_LINKKEY)
    monkeypatch.setenv("PAYAPP_LINKVAL", TEST_PAYAPP_LINKVAL)
    monkeypatch.setenv("PAYAPP_PUBLIC_BASE_URL", "https://api.boorda.test")
    monkeypatch.setenv("PAYAPP_RETURN_BASE_URL", "https://boorda.test")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def payapp(app: FastAPI) -> FakePayAppClient:
    """The provider double, already installed and inspectable.

    Autouse, and that is the safety property rather than a convenience:
    with the fake on `app.state` for every test, there is no path in this
    suite that constructs a real `HttpPayAppClient`, even if someone puts
    live credentials in the environment. A test that could call
    `rebillRegist` for real could register a live recurring contract
    against a real person's phone number.

    Tests assert against `payapp.registrations` — chiefly to prove the
    amount reaching the provider came from the plan table rather than
    from anything a browser sent.
    """
    client = FakePayAppClient()
    app.state.payapp_client = client
    return client


__all__ = [
    "TEST_PAYAPP_LINKKEY",
    "TEST_PAYAPP_LINKVAL",
    "TEST_PAYAPP_USERID",
    "payapp",
    "payapp_settings",
]
