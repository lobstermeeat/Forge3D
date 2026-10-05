import os
import socket
import sys

import pytest

# Import painter_worker from the worker folder without installing it
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Every test fails if anything it runs opens a network connection or looks up a host."""

    def refuse(*args, **kwargs):
        raise RuntimeError("a painter test tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
