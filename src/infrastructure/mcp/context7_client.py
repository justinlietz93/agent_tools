"""
Context7 MCP Client (HTTP transport; optional Docker auto-start)

Responsibilities:
- Detect whether a Context7 MCP server is reachable over HTTP (SSE-based MCP).
- Provide lightweight HTTP methods used by tools:
  - health_check(timeout=2.0) -> bool
  - check_docs(path, recursive, options) -> dict         (legacy; posts to /doc-check; may do local fallback)
  - check_sites(sites) -> dict                           (legacy; posts to /sites-check; no direct-HTTP fallback)
  - docs_completeness(path, recursive, include, exclude, required_sections, options) -> dict
  - docs_sites(path, sites, options) -> dict
  - docs_links(path, recursive, include, exclude, options) -> dict
  - docs_anchors(path, recursive, include, exclude, options) -> dict
  - docs_frontmatter(path, recursive, include, exclude, options) -> dict
- Optionally attempt to start a Docker container and wait until ready.

Environment variables:
- CONTEXT7_BASE_URL         (preferred; default: http://localhost:8080)
- CONTEXT7_HTTP_URL         (legacy; used if BASE_URL not set)
- CONTEXT7_CONTAINER_NAME   (default: context7-mcp)
- CONTEXT7_DOCKER_IMAGE     (default: mcp/context7:latest)
- CONTEXT7_PORT             (default: 8080 - host port for container)
- CONTEXT7_TIMEOUT_SECONDS  (default: 60)

Notes:
- health_check treats HTTP 200 as healthy.
- All public methods return structured dicts on error; they do not raise.
"""

from __future__ import annotations

import os
import time
import subprocess
from typing import Any, Dict, List, Optional, Union

import requests


class Context7MCPClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        container_name: Optional[str] = None,
        image: Optional[str] = None,
        port: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self.port = int(os.getenv("CONTEXT7_PORT", str(port or 8080)))
        # Prefer CONTEXT7_BASE_URL, then legacy CONTEXT7_HTTP_URL, then constructor/default
        # Default points to root; health_check will probe '/mcp' variants if needed.
        self.base_url = (
            os.getenv("CONTEXT7_BASE_URL")
            or os.getenv("CONTEXT7_HTTP_URL")
            or base_url
            or f"http://localhost:{self.port}"
        )
        self.container_name = os.getenv("CONTEXT7_CONTAINER_NAME", container_name or "context7-mcp")
        self.image = os.getenv("CONTEXT7_DOCKER_IMAGE", image or "mcp/context7:latest")
        self.timeout_seconds = int(os.getenv("CONTEXT7_TIMEOUT_SECONDS", str(timeout_seconds or 60)))

    # ---------- Public API ----------

    def is_available(self) -> bool:
        """
        Backwards-compatible availability check (alias for health_check with 3s timeout).
        """
        return self.health_check(timeout=3.0)

    def ensure_running(self) -> bool:
        """
        Ensure a Context7 server is up:
        - If available: return True.
        - Else: attempt to start via Docker (create or start existing), then wait until available.
        - Return True when available, else False after timeout.
        """
        if self.is_available():
            return True

        # Attempt to start container
        try:
            if not self._docker_available():
                return False

            if self._container_exists():
                self._docker(["start", self.container_name], check=False)
            else:
                # Run a new container with HTTP transport exposed on host port
                self._docker(
                    [
                        "run",
                        "-d",
                        "--name",
                        self.container_name,
                        "-p",
                        f"{self.port}:8080",
                        "-e",
                        "MCP_TRANSPORT=http",
                        self.image,
                    ],
                    check=False,
                )
        except Exception:
            # If docker itself raises, we'll fail availability check below
            pass

        # Wait for readiness
        deadline = time.time() + self.timeout_seconds
        while time.time() < deadline:
            if self.is_available():
                return True
            time.sleep(1)

        return False

    # ---------- HTTP surface ----------
    def health_check(self, timeout: float = 2.0) -> bool:
        """
        Healthy if any of the following return HTTP 200:
        - base_url /
        - base_url /health
        - base_url /healthz
        - base_url /mcp (some deployments mount under /mcp)
        - base_url /sse (legacy SSE endpoint returns 200 with event-stream)

        Robustness:
        - If base_url does NOT end with '/mcp', also probe the '/mcp' variants.
        - If base_url DOES end with '/mcp', also probe the parent path (strip '/mcp').
        - Probe SSE endpoints ('/mcp' and '/sse' variants) with Accept: text/event-stream.
        - On success, normalizes self.base_url to the working base path actually serving 200.
        """
        base = self.base_url.rstrip("/")

        # Build (url, needs_sse_accept_header) candidates
        candidates: List[tuple[str, bool]] = [
            (f"{base}/", False),
            (f"{base}/health", False),
            (f"{base}/healthz", False),
            (f"{base}/mcp", True),  # many images expose SSE on /mcp
            (f"{base}/sse", True),  # legacy SSE endpoint
        ]

        if base.endswith("/mcp"):
            parent = base[: -len("/mcp")]
            if parent:
                candidates.extend([
                    (f"{parent}/", False),
                    (f"{parent}/health", False),
                    (f"{parent}/healthz", False),
                    (f"{parent}/mcp", True),
                    (f"{parent}/sse", True),
                ])
        else:
            mcp = f"{base}/mcp"
            candidates.extend([
                (f"{mcp}/", False),
                (f"{mcp}/health", False),
                (f"{mcp}/healthz", False),
                (f"{mcp}", True),
                (f"{mcp}/sse", True),
            ])

        for url, needs_sse in candidates:
            try:
                headers = {"Accept": "text/event-stream"} if needs_sse else None
                # stream=True avoids long reads on SSE; we only need headers
                r = requests.get(url, timeout=timeout, stream=needs_sse, headers=headers)
                if r.status_code == 200:
                    # Normalize base_url to the parent path of the probe we used
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        path = parsed.path or ""
                        if path.endswith("/healthz"):
                            base_path = path[: -len("/healthz")]
                        elif path.endswith("/health"):
                            base_path = path[: -len("/health")]
                        elif path.endswith("/sse"):
                            base_path = path[: -len("/sse")]
                        elif path.endswith("/"):
                            base_path = path[: -1]
                        else:
                            base_path = path
                        normalized = f"{parsed.scheme}://{parsed.netloc}{base_path}".rstrip("/")
                        self.base_url = normalized or self.base_url
                    except Exception:
                        pass
                    return True
            except requests.RequestException:
                continue
        return False

    def check_docs(self, path: str, recursive: bool = True, options: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Attempt to POST to /doc-check with payload; if endpoint is missing (404/405),
        fallback to a local docs check (no remote HTTP scraping).
        Always returns a dict, with keys like: status, files_checked, details, error.
        """
        payload: Dict[str, Any] = {"path": path, "recursive": bool(recursive), "options": options or {}}
        if not self.health_check(timeout=2.0):
            return {
                "status": "unavailable",
                "error": f"Context7 MCP is unavailable at {self.base_url}",
                "base_url": self.base_url,
            }

        url = self._join("/doc-check")
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            # If endpoint exists and returns JSON, use it
            if 200 <= resp.status_code < 300:
                try:
                    data = resp.json()
                except ValueError:
                    data = {"status": "ok", "raw": resp.text}
                return data
            # If the endpoint doesn't exist, fall back to local check
            if resp.status_code in (404, 405):
                return self._local_docs_check(path, recursive, options or {})
            # Other HTTP errors -> structured error
            return {
                "status": "error",
                "error": f"/doc-check returned HTTP {resp.status_code}",
                "base_url": self.base_url,
            }
        except requests.RequestException as e:
            # Network/timeout -> structured error, but still attempt local fallback
            fallback = self._local_docs_check(path, recursive, options or {})
            fallback.setdefault("note", "Performed local docs check due to HTTP error")
            fallback.setdefault("http_error", str(e))
            return fallback

    def check_sites(self, sites: Union[str, List[str]], timeout: float = 10.0) -> Dict[str, Any]:
        """
        Attempt to POST to /sites-check with {'sites': [...]}.
        Does NOT perform direct-HTTP scraping fallback; returns a clear status if endpoint is missing.
        """
        site_list: List[str] = [sites] if isinstance(sites, str) else list(sites or [])
        if not self.health_check(timeout=2.0):
            return {
                "status": "unavailable",
                "error": f"Context7 MCP is unavailable at {self.base_url}",
                "base_url": self.base_url,
            }

        url = self._join("/sites-check")
        try:
            resp = requests.post(url, json={"sites": site_list}, timeout=timeout)
            if 200 <= resp.status_code < 300:
                try:
                    return resp.json()
                except ValueError:
                    return {"status": "ok", "raw": resp.text}
            if resp.status_code in (404, 405):
                return {
                    "status": "skipped",
                    "reason": "Context7 MCP '/sites-check' endpoint not available",
                    "sites": site_list,
                    "base_url": self.base_url,
                }
            return {
                "status": "error",
                "error": f"/sites-check returned HTTP {resp.status_code}",
                "base_url": self.base_url,
            }
        except requests.RequestException as e:
            return {"status": "unavailable", "error": str(e), "sites": site_list, "base_url": self.base_url}

    # ---------- New high-level docs operations (no local fallback here) ----------

    def docs_completeness(
        self,
        path: str,
        recursive: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        required_sections: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        POST to /doc-check (compat) with extended payload. No local fallback; returns structured error if unavailable.
        """
        if not self.health_check(timeout=2.0):
            return {
                "status": "unavailable",
                "error": f"Context7 MCP not reachable at {self.base_url}",
                "base_url": self.base_url,
            }
        payload: Dict[str, Any] = {
            "path": path,
            "recursive": bool(recursive),
            "include": include or [],
            "exclude": exclude or [],
            "required_sections": required_sections or [],
            "options": options or {},
        }
        url = self._join("/doc-check")
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if 200 <= r.status_code < 300:
                try:
                    data = r.json()
                except ValueError:
                    data = {"raw": r.text}
                data.setdefault("base_url", self.base_url)
                return data
            return {
                "status": "error",
                "error": f"/doc-check returned HTTP {r.status_code}",
                "base_url": self.base_url,
            }
        except requests.RequestException as e:
            return {"status": "unavailable", "error": str(e), "base_url": self.base_url}

    def docs_sites(
        self,
        path: Optional[str] = None,
        sites: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        POST to /sites-check with {'sites': [...], 'options': {...}}. No fallback.
        """
        if not self.health_check(timeout=2.0):
            return {
                "status": "unavailable",
                "error": f"Context7 MCP not reachable at {self.base_url}",
                "base_url": self.base_url,
            }
        payload = {"sites": list(sites or []), "options": options or {}}
        url = self._join("/sites-check")
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if 200 <= r.status_code < 300:
                try:
                    data = r.json()
                except ValueError:
                    data = {"raw": r.text}
                data.setdefault("base_url", self.base_url)
                return data
            if r.status_code in (404, 405):
                return {
                    "status": "unsupported",
                    "error": "Context7 endpoint '/sites-check' not available",
                    "base_url": self.base_url,
                }
            return {
                "status": "error",
                "error": f"/sites-check returned HTTP {r.status_code}",
                "base_url": self.base_url,
            }
        except requests.RequestException as e:
            return {"status": "unavailable", "error": str(e), "base_url": self.base_url}

    def docs_links(
        self,
        path: str,
        recursive: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: float = 20.0,
    ) -> Dict[str, Any]:
        """
        POST to /links-check with payload. If endpoint missing, return 'unsupported'.
        """
        if not self.health_check(timeout=2.0):
            return {
                "status": "unavailable",
                "error": f"Context7 MCP not reachable at {self.base_url}",
                "base_url": self.base_url,
            }
        payload: Dict[str, Any] = {
            "path": path,
            "recursive": bool(recursive),
            "include": include or [],
            "exclude": exclude or [],
            "options": options or {},
        }
        url = self._join("/links-check")
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if 200 <= r.status_code < 300:
                try:
                    data = r.json()
                except ValueError:
                    data = {"raw": r.text}
                data.setdefault("base_url", self.base_url)
                return data
            if r.status_code in (404, 405):
                return {
                    "status": "unsupported",
                    "error": "Context7 endpoint '/links-check' not available",
                    "base_url": self.base_url,
                }
            return {
                "status": "error",
                "error": f"/links-check returned HTTP {r.status_code}",
                "base_url": self.base_url,
            }
        except requests.RequestException as e:
            return {"status": "unavailable", "error": str(e), "base_url": self.base_url}

    def docs_anchors(
        self,
        path: str,
        recursive: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: float = 20.0,
    ) -> Dict[str, Any]:
        """
        POST to /anchors-check with payload.
        """
        if not self.health_check(timeout=2.0):
            return {
                "status": "unavailable",
                "error": f"Context7 MCP not reachable at {self.base_url}",
                "base_url": self.base_url,
            }
        payload: Dict[str, Any] = {
            "path": path,
            "recursive": bool(recursive),
            "include": include or [],
            "exclude": exclude or [],
            "options": options or {},
        }
        url = self._join("/anchors-check")
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if 200 <= r.status_code < 300:
                try:
                    data = r.json()
                except ValueError:
                    data = {"raw": r.text}
                data.setdefault("base_url", self.base_url)
                return data
            if r.status_code in (404, 405):
                return {
                    "status": "unsupported",
                    "error": "Context7 endpoint '/anchors-check' not available",
                    "base_url": self.base_url,
                }
            return {
                "status": "error",
                "error": f"/anchors-check returned HTTP {r.status_code}",
                "base_url": self.base_url,
            }
        except requests.RequestException as e:
            return {"status": "unavailable", "error": str(e), "base_url": self.base_url}

    def docs_frontmatter(
        self,
        path: str,
        recursive: bool = False,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout: float = 20.0,
    ) -> Dict[str, Any]:
        """
        POST to /frontmatter-check with payload.
        """
        if not self.health_check(timeout=2.0):
            return {
                "status": "unavailable",
                "error": f"Context7 MCP not reachable at {self.base_url}",
                "base_url": self.base_url,
            }
        payload: Dict[str, Any] = {
            "path": path,
            "recursive": bool(recursive),
            "include": include or [],
            "exclude": exclude or [],
            "options": options or {},
        }
        url = self._join("/frontmatter-check")
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if 200 <= r.status_code < 300:
                try:
                    data = r.json()
                except ValueError:
                    data = {"raw": r.text}
                data.setdefault("base_url", self.base_url)
                return data
            if r.status_code in (404, 405):
                return {
                    "status": "unsupported",
                    "error": "Context7 endpoint '/frontmatter-check' not available",
                    "base_url": self.base_url,
                }
            return {
                "status": "error",
                "error": f"/frontmatter-check returned HTTP {r.status_code}",
                "base_url": self.base_url,
            }
        except requests.RequestException as e:
            return {"status": "unavailable", "error": str(e), "base_url": self.base_url}

    # ---------- Library docs (MCP tool semantics via HTTP client facade) ----------
 
    def resolve_library_id(self, library_name: str) -> str:
        """
        Resolve a human-readable library name to a Context7-compatible library ID.
        NOTE: This HTTP client does not directly speak MCP tool protocol over SSE,
        so this method returns the input as-is by default.
 
        Callers may pass already-resolved IDs like '/org/project' or '/org/project/version'.
        """
        if not isinstance(library_name, str) or not library_name.strip():
            raise ValueError("library_name must be a non-empty string")
        return library_name.strip()
 
    def get_docs(
        self,
        library_id: str,
        topic: Optional[str] = None,
        tokens: int = 4000,
        timeout: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Retrieve documentation text for a given library via Context7.

        This HTTP client does not implement the SSE-based MCP tool invocation
        ('get-library-docs'). To keep runtime deps minimal, this method returns a
        structured 'unsupported' result. An SSE-capable client can be injected that
        overrides this method.
        """
        if not isinstance(library_id, str) or not library_id.strip():
            return {"status": "error", "error": "library_id is required", "base_url": self.base_url}
        return {
            "status": "unsupported",
            "error": "get-library-docs over raw HTTP is not supported by Context7MCPClient; inject an SSE-capable client",
            "base_url": self.base_url,
            "library_id": library_id,
            "topic": topic,
            "tokens": tokens,
        }

    # ---------- Private helpers ----------

    def _join(self, path: str) -> str:
        base = self.base_url.rstrip("/")
        p = path if path.startswith("/") else f"/{path}"
        return f"{base}{p}"

    def _docker_available(self) -> bool:
        try:
            self._docker(["version"], check=True)
            return True
        except Exception:
            return False

    def _container_exists(self) -> bool:
        try:
            out = self._docker(
                ["ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                check=False,
                capture_output=True,
            )
            names = (out or "").strip().splitlines()
            return any(n.strip() == self.container_name for n in names if n.strip())
        except Exception:
            return False

    def _docker(self, args: list[str], check: bool, capture_output: bool = False) -> Optional[str]:
        cmd = ["docker"] + args
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True,
        )
        if capture_output:
            return (result.stdout or "") + (result.stderr or "")
        return None

    # ---------- Local fallback for docs check ----------
    def _local_docs_check(self, path: str, recursive: bool, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimal local markdown 'completeness' check used as a fallback when /doc-check
        is unavailable. Only validates required_sections presence in headings.
        """
        import os as _os
        required: List[str] = list(options.get("required_sections") or [])
        files: List[str] = []

        apath = _os.path.abspath(path)
        if _os.path.isfile(apath):
            if apath.lower().endswith(".md"):
                files.append(apath)
        elif _os.path.isdir(apath):
            for root, _dirs, fns in _os.walk(apath):
                for fn in fns:
                    if fn.lower().endswith(".md"):
                        files.append(_os.path.join(root, fn))
                if not recursive:
                    break

        details: List[Dict[str, Any]] = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    txt = fh.read()
                headings = self._extract_headings(txt)
                missing = [sec for sec in required if sec not in headings]
                details.append({"file": f, "missing_sections": missing})
            except Exception as e:
                details.append({"file": f, "error": str(e)})

        return {
            "status": "ok",
            "files_checked": len(files),
            "details": details,
            "base_url": self.base_url,
            "mode": "local_fallback",
        }

    def _extract_headings(self, text: str) -> List[str]:
        import re as _re
        heads: List[str] = []
        for line in (text or "").splitlines():
            m = _re.match(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", line)
            if m:
                name = m.group(1).strip().strip("#").strip()
                if name:
                    heads.append(name)
        return heads