"""Tests for the TelegramChannel adapter."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from openjarvis.channels._stubs import ChannelStatus
from openjarvis.channels.telegram import TelegramChannel
from openjarvis.core.events import EventBus, EventType
from openjarvis.core.registry import ChannelRegistry


@pytest.fixture(autouse=True)
def _register_telegram():
    """Re-register after any registry clear."""
    if not ChannelRegistry.contains("telegram"):
        ChannelRegistry.register_value("telegram", TelegramChannel)


class TestRegistration:
    def test_registry_key(self):
        assert ChannelRegistry.contains("telegram")

    def test_channel_id(self):
        ch = TelegramChannel(bot_token="test-token")
        assert ch.channel_id == "telegram"


class TestInit:
    def test_defaults(self):
        ch = TelegramChannel()
        assert ch._token == ""
        assert ch._parse_mode == "Markdown"
        assert ch._status == ChannelStatus.DISCONNECTED

    def test_constructor_token(self):
        ch = TelegramChannel(bot_token="my-token")
        assert ch._token == "my-token"

    def test_env_var_fallback(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "env-token"}):
            ch = TelegramChannel()
            assert ch._token == "env-token"

    def test_constructor_overrides_env(self):
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "env-token"}):
            ch = TelegramChannel(bot_token="explicit-token")
            assert ch._token == "explicit-token"


class TestSend:
    def test_send_success(self):
        ch = TelegramChannel(bot_token="123:ABC")

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.post", return_value=mock_response) as mock_post:
            result = ch.send("12345678", "Hello!")
            assert result is True
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            url = call_args[0][0]
            assert "api.telegram.org" in url
            assert "bot123:ABC" in url
            assert "sendMessage" in url
            payload = call_args[1]["json"]
            assert payload["chat_id"] == "12345678"
            assert payload["text"] == "Hello!"
            assert payload["parse_mode"] == "Markdown"

    def test_send_failure(self):
        ch = TelegramChannel(bot_token="123:ABC")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        with patch("httpx.post", return_value=mock_response):
            result = ch.send("12345678", "Hello!")
            assert result is False

    def test_send_exception(self):
        ch = TelegramChannel(bot_token="123:ABC")

        with patch("httpx.post", side_effect=ConnectionError("refused")):
            result = ch.send("12345678", "Hello!")
            assert result is False

    def test_send_no_token(self):
        ch = TelegramChannel()
        result = ch.send("12345678", "Hello!")
        assert result is False

    def test_send_publishes_event(self):
        bus = EventBus(record_history=True)
        ch = TelegramChannel(bot_token="123:ABC", bus=bus)

        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.post", return_value=mock_response):
            ch.send("12345678", "Hello!")

        event_types = [e.event_type for e in bus.history]
        assert EventType.CHANNEL_MESSAGE_SENT in event_types


class TestListChannels:
    def test_list_channels(self):
        ch = TelegramChannel(bot_token="123:ABC")
        assert ch.list_channels() == ["telegram"]


class TestStatus:
    def test_disconnected_initially(self):
        ch = TelegramChannel(bot_token="123:ABC")
        assert ch.status() == ChannelStatus.DISCONNECTED

    def test_no_token_connect_error(self):
        ch = TelegramChannel()
        ch.connect()
        assert ch.status() == ChannelStatus.ERROR


class TestOnMessage:
    def test_on_message(self):
        ch = TelegramChannel(bot_token="123:ABC")
        handler = MagicMock()
        ch.on_message(handler)
        assert handler in ch._handlers


class TestDisconnect:
    def test_disconnect(self):
        ch = TelegramChannel(bot_token="123:ABC")
        ch._status = ChannelStatus.CONNECTED
        ch.disconnect()
        assert ch.status() == ChannelStatus.DISCONNECTED
