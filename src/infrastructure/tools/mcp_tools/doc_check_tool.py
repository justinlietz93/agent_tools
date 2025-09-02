"""
DocCheckTool — Context7 MCP adapter (first-class, library-centric)

Objective:
- Provide a clean, thin adapter to Context7 MCP with explicit, versionable input schema.
- No Anthropic/LLM coupling and no local fallback logic.
- Standardized output shape across all checks.
- ≤ 500 LOC, dependency-injected Context7 client.

Important:
- Context7 serves documentation for requested libraries (not your local codebase).
- This tool uses Context7 MCP tools semantics (resolve library → get docs).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict, Literal

from src.infrastructure.tools.tool_base import Tool

_ALLOWED_CHECKS = {"completeness", "sites", "links", "anchors", "frontmatter"}


class DocCheckInputV1(TypedDict, total=False):
    check_type: Literal["completeness", "sites", "links", "anchors", "frontmatter"]
    library: str
    libraries: List[str]
    required_sections: List[str]
    topic: str
    tokens: int
    options: Dict[str, Any]


class DocCheckTool(Tool):
    """
    Context7 Documentation Check Tool (strict MCP adapter).
    - completeness: fetch docs for a library and verify required_sections are present
    - sites: fetch docs for each provided library and report availability
    - links/anchors/frontmatter: currently provider-unsupported (returns standardized error)
    """

    def __init__(self, context7_client: Optional[Any] = None) -> None:
        # Dependency injection for clean architecture and testing
        self._client = context7_client

    # Tool interface

    @property
    def name(self) -> str:
        return "documentation_check"

    @property
    def description(self) -> str:
        return (
            "Validate third-party documentation through Context7 MCP. "
            "Checks include: completeness (by required sections) and sites (library availability). "
            "Links/anchors/frontmatter are not yet supported by Context7 and return a standardized error."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        # Pydantic-lite JSON schema exposed in tool metadata
        return {
            "type": "object",
            "properties": {
                "check_type": {
                    "type": "string",
                    "description": "Which validation to run on provider-returned docs",
                    "enum": ["completeness", "sites", "links", "anchors", "frontmatter"],
                },
                "library": {
                    "type": "string",
                    "description": "Library name or Context7-compatible ID (/org/project or /org/project/version).",
                },
                "libraries": {
                    "type": "array",
                    "description": "Multiple library names or IDs (used by 'sites'; can also be used for completeness with the first item).",
                    "items": {"type": "string"},
                },
                "required_sections": {
                    "type": "array",
                    "description": "Required headings/sections to look for in the fetched documentation (completeness).",
                    "items": {"type": "string"},
                },
                "topic": {
                    "type": "string",
                    "description": "Optional topic to focus the provider docs on (provider-specific).",
                },
                "tokens": {
                    "type": "integer",
                    "description": "Maximum number of tokens to retrieve from provider (provider-specific).",
                    "default": 4000,
                },
                "options": {
                    "type": "object",
                    "description": "Future flags; kept for forward-compatibility.",
                    "additionalProperties": True,
                },
            },
            "required": ["check_type"],
            "additionalProperties": True,
        }

    def run(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the tool with validated parameters and standardized output.

        Standardized output:
        {
          "status": "success" | "error",
          "check_type": str,
          "summary": str,
          "data": Dict[str, Any],
          "base_url": str
        }
        """
        try:
            params = self._validate_and_normalize(input)
        except ValueError as ve:
            return self._wrap_error(
                check_type=str(input.get("check_type") or ""),
                summary=f"Input validation failed: {ve}",
                base_url=self._get_base_url_fallback(),
                data={"error": str(ve)},
            )

        client = self._get_client()
        base_url = getattr(client, "base_url", self._get_base_url_fallback())

        # Server availability first
        try:
            healthy = bool(client.health_check(timeout=2.0))
        except Exception as e:
            return self._wrap_error(
                check_type=params["check_type"],
                summary=f"Context7 health_check failed: {e}",
                base_url=base_url,
                data={"error": str(e)},
            )

        if not healthy:
            return self._wrap_error(
                check_type=params["check_type"],
                summary=f"Context7 MCP not reachable at {base_url}",
                base_url=base_url,
                data={"reason": "unavailable"},
            )

        ct = params["check_type"]
        if ct == "completeness":
            return self._handle_completeness(client, base_url, params)
        if ct == "sites":
            return self._handle_sites(client, base_url, params)

        # Unsupported operations for now
        return self._wrap_error(
            check_type=ct,
            summary=f"{ct} check is not supported by Context7 provider at this time",
            base_url=base_url,
            data={"error": "unsupported"},
        )

    # Internal handlers

    def _handle_completeness(self, client: Any, base_url: str, p: DocCheckInputV1) -> Dict[str, Any]:
        lib = (p.get("library") or (p.get("libraries") or [None])[0] or "").strip()
        if not lib:
            return self._wrap_error(
                check_type="completeness",
                summary="Missing 'library' (or first item in 'libraries') for completeness check",
                base_url=base_url,
                data={"error": "missing_library"},
            )

        try:
            lib_id = client.resolve_library_id(lib)
        except Exception as e:
            return self._wrap_error("completeness", f"Failed to resolve library id: {e}", base_url, {"error": str(e)})

        try:
            resp: Dict[str, Any] = client.get_docs(
                lib_id,
                topic=p.get("topic"),
                tokens=int(p.get("tokens") or 4000),
            )
        except Exception as e:
            return self._wrap_error("completeness", f"get_docs failed: {e}", base_url, {"error": str(e)})

        status = str(resp.get("status", "") or "").lower()
        if status in {"error", "unavailable", "unsupported"}:
            return self._wrap_error(
                "completeness",
                f"Provider returned {status} for {lib_id}",
                base_url,
                {"provider": resp, "library": lib, "library_id": lib_id},
            )

        text = self._extract_text(resp)
        req = list(p.get("required_sections") or [])
        missing = self._missing_sections(text, req) if req else []

        analysis = {"library": lib, "library_id": lib_id, "required_sections": req, "missing_sections": missing}
        data = {"provider": resp, "analysis": analysis}
        if missing:
            summary = f"Completeness found missing sections: {', '.join(missing)}"
        else:
            summary = "Completeness check succeeded; all required sections present" if req else "Completeness executed with no required sections"
        return {
            "status": "success",
            "check_type": "completeness",
            "summary": summary,
            "data": data,
            "base_url": base_url,
        }

    def _handle_sites(self, client: Any, base_url: str, p: DocCheckInputV1) -> Dict[str, Any]:
        libs: List[str] = list(p.get("libraries") or [])
        if not libs and p.get("library"):
            libs = [str(p["library"])]

        if not libs:
            return self._wrap_error(
                "sites", "No 'libraries' provided for sites check", base_url, {"error": "missing_libraries"}
            )

        results: List[Dict[str, Any]] = []
        ok = 0
        for name in libs:
            try:
                lib_id = client.resolve_library_id(name)
                resp: Dict[str, Any] = client.get_docs(
                    lib_id, topic=p.get("topic"), tokens=int(p.get("tokens") or 4000)
                )
                st = str(resp.get("status", "") or "").lower()
                if st in {"error", "unavailable", "unsupported"}:
                    results.append({"library": name, "library_id": lib_id, "status": st, "error": resp.get("error")})
                else:
                    ok += 1
                    results.append({"library": name, "library_id": lib_id, "status": "ok"})
            except Exception as e:
                results.append({"library": name, "status": "error", "error": str(e)})

        summary = f"Sites check: {ok} OK, {len(libs) - ok} with issues"
        return {
            "status": "success",
            "check_type": "sites",
            "summary": summary,
            "data": {"results": results},
            "base_url": base_url,
        }

    # Internal helpers

    def _get_client(self):
        if self._client is not None:
            return self._client
        # Lazy import to avoid hard coupling at import time
        from src.infrastructure.mcp.context7_client import Context7MCPClient  # type: ignore
        self._client = Context7MCPClient()
        return self._client

    def _get_base_url_fallback(self) -> str:
        # Matches client default if not constructed: http://localhost:8080
        return "http://localhost:8080"

    def _validate_and_normalize(self, raw: Dict[str, Any]) -> DocCheckInputV1:
        # Required
        check_type_raw = str((raw.get("check_type") or "")).strip().lower()
        if check_type_raw not in _ALLOWED_CHECKS:
            raise ValueError(f"check_type must be one of {sorted(_ALLOWED_CHECKS)}")

        # Optional with defaults
        library = str(raw.get("library") or "").strip()
        libraries = [str(x).strip() for x in (raw.get("libraries") or []) if isinstance(x, str) and str(x).strip()]
        required_sections = [str(x).strip() for x in (raw.get("required_sections") or []) if isinstance(x, str) and str(x).strip()]
        topic = str(raw.get("topic") or "").strip() or None
        tokens_val = raw.get("tokens", 4000)
        try:
            tokens = int(tokens_val)
        except Exception:
            raise ValueError("tokens must be an integer")
        options = dict(raw.get("options") or {})

        # Basic shape guards
        if not isinstance(options, dict):
            raise ValueError("options must be an object/dict")

        return DocCheckInputV1(
            check_type=check_type_raw,  # type: ignore[assignment]
            library=library,
            libraries=libraries,
            required_sections=required_sections,
            topic=topic or "",
            tokens=tokens,
            options=options,
        )

    def _extract_text(self, provider_resp: Dict[str, Any]) -> str:
        # Best-effort extraction from provider-native payloads
        for key in ("text", "raw", "content", "docs"):
            v = provider_resp.get(key)
            if isinstance(v, str) and v.strip():
                return v
        # Fallback: stringify known fields
        return str(provider_resp)

    def _missing_sections(self, text: str, required: List[str]) -> List[str]:
        if not text:
            return list(required or [])
        lower = text.lower()
        missing: List[str] = []
        for sec in required:
            if sec and sec.lower() not in lower:
                missing.append(sec)
        return missing

    def _wrap_error(self, check_type: str, summary: str, base_url: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "error",
            "check_type": check_type,
            "summary": summary,
            "data": data,
            "base_url": base_url,
        }