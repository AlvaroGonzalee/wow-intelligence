from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler

from wow_intelligence.rag.validation.metadata import _require_url


class ImportError(ValueError):
    # Describe an invalid API response or unsafe import destination.
    pass


def _ahledger_url(value: object) -> str:
    _require_url(value, "AHledger URL")
    if urlsplit(value).scheme != "https" or urlsplit(value).hostname != "ahledger.com":
        raise ImportError("Item and attribution links must point to https://ahledger.com.")
    return value


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ImportError("API redirects are not followed. Review the new endpoint first.")
