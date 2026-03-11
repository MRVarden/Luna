"""Provider factory — lazy import of the selected LLM backend.

Only the SDK for the chosen provider is loaded at runtime.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import urllib.parse
from typing import TYPE_CHECKING

from luna.llm_bridge.bridge import LLMBridge, LLMBridgeError

if TYPE_CHECKING:
    from luna.core.config import LLMSection

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSRF protection — base_url validation for cloud providers (M-01)
# ---------------------------------------------------------------------------

_BLOCKED_NETWORKS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("::1/128"),
)


def _validate_provider_base_url(url: str, provider_name: str) -> None:
    """Validate a provider base_url against SSRF attacks.

    Cloud providers (openai, deepseek) must not resolve to private/loopback IPs.
    The ``local`` provider is exempt — localhost is its legitimate target.

    Raises:
        LLMBridgeError: If the URL resolves to a blocked network.
    """
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        raise LLMBridgeError(
            f"Failed to parse base_url for {provider_name}: {url!r}",
            provider=provider_name,
        )

    if parsed.scheme not in ("http", "https"):
        raise LLMBridgeError(
            f"Unsupported scheme {parsed.scheme!r} in base_url for {provider_name}",
            provider=provider_name,
        )

    hostname = parsed.hostname
    if not hostname:
        raise LLMBridgeError(
            f"No hostname in base_url for {provider_name}: {url!r}",
            provider=provider_name,
        )

    try:
        addrinfo = socket.getaddrinfo(
            hostname, parsed.port or 443, proto=socket.IPPROTO_TCP
        )
    except socket.gaierror:
        raise LLMBridgeError(
            f"DNS resolution failed for {provider_name} base_url hostname {hostname!r}",
            provider=provider_name,
        )

    for _family, _type, _proto, _canonname, sockaddr in addrinfo:
        ip_str = sockaddr[0]
        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            raise LLMBridgeError(
                f"Invalid resolved IP {ip_str!r} for {provider_name} base_url",
                provider=provider_name,
            )

        for network in _BLOCKED_NETWORKS:
            if addr in network:
                raise LLMBridgeError(
                    f"SSRF blocked: {provider_name} base_url {url!r} resolved to "
                    f"{addr} in private range {network}",
                    provider=provider_name,
                )

    log.debug("SSRF check passed for %s base_url: %s", provider_name, url)


def create_provider(config: LLMSection) -> LLMBridge:
    """Instantiate the correct provider from config.

    Args:
        config: The ``[llm]`` section of ``LunaConfig``.

    Returns:
        A concrete ``LLMBridge`` implementation.

    Raises:
        LLMBridgeError: If the provider name is unknown.
    """
    # Load .env if present — ensures API keys are available even when
    # launched from contexts that don't inherit the user's shell env.
    try:
        from dotenv import load_dotenv
        from pathlib import Path

        env_path = Path(__file__).resolve().parents[3] / ".env"
        if env_path.is_file():
            load_dotenv(env_path, override=False)
            log.debug("Loaded .env from %s", env_path)
    except ImportError:
        pass  # python-dotenv not installed — rely on env vars

    provider = config.provider.lower()

    if provider == "anthropic":
        from luna.llm_bridge.providers.anthropic import AnthropicProvider

        return AnthropicProvider(
            model=config.model,
            api_key=config.api_key,
        )

    if provider == "openai":
        if config.base_url:
            _validate_provider_base_url(config.base_url, "openai")
        from luna.llm_bridge.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "deepseek":
        if config.base_url:
            _validate_provider_base_url(config.base_url, "deepseek")
        from luna.llm_bridge.providers.deepseek import DeepSeekProvider

        return DeepSeekProvider(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
        )

    if provider == "local":
        from luna.llm_bridge.providers.local import LocalProvider

        return LocalProvider(
            model=config.model,
            base_url=config.base_url or "http://localhost:11434/v1",
            api_key=config.api_key or "ollama",
        )

    raise LLMBridgeError(
        f"Unknown LLM provider: {config.provider!r}. "
        f"Supported: anthropic, openai, deepseek, local.",
        provider=config.provider,
    )
