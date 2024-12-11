from copy import deepcopy
from dataclasses import dataclass
import os
import sys
from browsergym.webarena.instance import WebArenaInstance


class BaseServer:
    """Base class for server instances.

    Behaves like an identity function for running in parallel on servers that don't need multiple
    instances.
    """

    def init(self):
        pass


@dataclass
class WebArenaInstanceVars(BaseServer):
    base_url: str
    shopping: str
    shopping_admin: str
    reddit: str
    gitlab: str
    wikipedia: str
    map: str
    homepage: str
    full_reset: str
    module_name: str = "webarena"
    prefix: str = "WA_"

    def make_env_vars(self):
        """Return a dictionary of environment variables"""
        return {
            f"{self.prefix}SHOPPING": f"{self.base_url}:{self.shopping}",
            f"{self.prefix}SHOPPING_ADMIN": f"{self.base_url}:{self.shopping_admin}",
            f"{self.prefix}REDDIT": f"{self.base_url}:{self.reddit}",
            f"{self.prefix}GITLAB": f"{self.base_url}:{self.gitlab}",
            f"{self.prefix}WIKIPEDIA": f"{self.base_url}:{self.wikipedia}",
            f"{self.prefix}MAP": f"{self.base_url}:{self.map}",
            f"{self.prefix}HOMEPAGE": f"{self.base_url}:{self.homepage}",
            f"{self.prefix}FULL_RESET": f"{self.base_url}:{self.full_reset}",
        }

    def init(self):
        # necessary for webarena to re-import the env vars
        unimport_modules(self.module_name)
        for key, value in self.make_env_vars().items():
            os.environ[key] = value

        # this is just a dynamic check to see that the env vars are set correctly
        bgym_instance = WebArenaInstance()
        base_url, _ = _split_url(bgym_instance.urls["reddit"])
        assert base_url == self.base_url, f"Expected {self.base_url}, got {base_url}"

    @staticmethod
    def from_env_vars(prefix="WA_", module_name="webarena"):
        kwargs = {"module_name": module_name}
        base_urls = set()
        for key, url in os.environ.items():
            if key.startswith(prefix):
                base_url, url_tail = _split_url(url)
                base_urls.add(base_url)
                kwargs[key[len(prefix) :].lower()] = url_tail

        if len(base_urls) > 1:
            raise ValueError("Multiple base urls found in environment variables")

        kwargs["base_url"] = base_urls.pop()
        return WebArenaInstanceVars(**kwargs)

    def clone(self):
        """Return a deep copy of the instance"""
        return deepcopy(self)


def unimport_modules(base_name):
    """un-import any module starting with base_name"""
    for module in sys.modules.copy():
        if module.startswith(base_name):
            del sys.modules[module]


def _split_url(url: str):
    """Extract the base url and the port/page from a url"""
    parts = url.split(":")
    base_url = ":".join(parts[0:2])
    url_tail = ":".join(parts[2:])
    return base_url, url_tail
