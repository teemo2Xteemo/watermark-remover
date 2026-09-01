from __future__ import annotations

import socket

import pytest
from pytest_socket import SocketBlockedError


def test_tcp_socket_construction_is_blocked() -> None:
    with pytest.warns(UserWarning, match="socket"):
        with pytest.raises(SocketBlockedError):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_urllib_urlopen_cannot_open_the_network() -> None:
    import urllib.request

    with pytest.warns(UserWarning, match="socket"):
        with pytest.raises(SocketBlockedError):
            urllib.request.urlopen("http://127.0.0.1:1", timeout=0.1)
