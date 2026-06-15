from pathlib import Path

import pytest

from metatranscribe.web.app import create_app
from tests.test_web_app import _settings


@pytest.fixture
def client(tmp_path):
    app = create_app(_settings(tmp_path), start_worker=False)
    app.config["TESTING"] = True
    return app.test_client()


def test_wrong_password_is_rejected(client):
    resp = client.post("/login", data={"password": "wrong"})
    assert resp.status_code == 200
    assert b"Incorrect password" in resp.data
    # Still locked out of protected routes.
    assert client.get("/").status_code == 302


def test_correct_password_grants_access(client):
    client.post("/login", data={"password": "hunter2"})
    assert client.get("/").status_code == 200


def test_logout_clears_session(client):
    client.post("/login", data={"password": "hunter2"})
    assert client.get("/").status_code == 200
    client.post("/logout")
    resp = client.get("/")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_login_next_must_be_relative(client):
    resp = client.post("/login?next=http://evil.example/", data={"password": "hunter2"})
    assert resp.status_code == 302
    assert "evil.example" not in resp.headers["Location"]


def test_missing_credentials_raises():
    from metatranscribe.config import validate_web_credentials

    settings = _settings(Path("/tmp/does-not-matter"))
    settings.web_password = None
    with pytest.raises(ValueError, match="WEB_PASSWORD"):
        validate_web_credentials(settings)
