"""
Tor Manager — handles SOCKS5 proxy configuration and dynamic IP rotation via stem.
"""

import requests
from stem import Signal
from stem.control import Controller

from app import config


class TorManager:
    """Manages Tor proxy sessions and identity rotation."""

    @staticmethod
    def get_session() -> requests.Session:
        """
        Create a requests Session routed through the local Tor SOCKS proxy.
        If USE_TOR is false, returns a standard unproxied session.
        """
        session = requests.Session()
        if config.USE_TOR:
            session.proxies = {
                "http": config.TOR_PROXY_URL,
                "https": config.TOR_PROXY_URL
            }
        return session

    @staticmethod
    def renew_ip() -> bool:
        """
        Request a new circuit (new IP address) from the Tor control port.
        Returns True if successful, False otherwise.
        """
        if not config.USE_TOR:
            return False

        try:
            with Controller.from_port(port=config.TOR_CONTROL_PORT) as controller:
                if config.TOR_PASSWORD:
                    controller.authenticate(password=config.TOR_PASSWORD)
                else:
                    controller.authenticate()
                
                controller.signal(Signal.NEWNYM)
                return True
        except Exception as e:
            print(f"\n[!] Failed to renew Tor IP via control port {config.TOR_CONTROL_PORT}: {e}")
            return False
