import pytest


@pytest.fixture(autouse=True)
def isolated_tutor_environment(monkeypatch):
    for name in list(__import__("os").environ):
        if name.startswith("TUTOR_"):
            monkeypatch.delenv(name, raising=False)
