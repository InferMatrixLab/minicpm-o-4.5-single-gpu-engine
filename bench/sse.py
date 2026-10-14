"""Incremental Server-Sent Events parser tolerant to arbitrary TCP fragmentation."""

from __future__ import annotations


class SSEParser:
    """Feed raw bytes; get complete ``data:`` payloads (strings) back.

    Events are separated by a blank line.  Multiple ``data:`` lines in one
    event are joined with newlines.  Comment lines (``:`` prefix) and other
    fields are ignored.
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> list[str]:
        self._buf.extend(chunk)
        out: list[str] = []
        while True:
            idx, sep_len = self._find_boundary()
            if idx < 0:
                break
            raw = bytes(self._buf[:idx])
            del self._buf[: idx + sep_len]
            data = self._parse_event(raw)
            if data is not None:
                out.append(data)
        return out

    def flush(self) -> list[str]:
        """Parse whatever remains (a stream that ended without a blank line)."""
        raw = bytes(self._buf)
        self._buf.clear()
        if not raw.strip():
            return []
        data = self._parse_event(raw)
        return [] if data is None else [data]

    def _find_boundary(self) -> tuple[int, int]:
        best = (-1, 0)
        for sep in (b"\r\n\r\n", b"\n\n", b"\r\r"):
            i = self._buf.find(sep)
            if i >= 0 and (best[0] < 0 or i < best[0]):
                best = (i, len(sep))
        return best

    @staticmethod
    def _parse_event(raw: bytes) -> str | None:
        lines = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n").split("\n")
        data_lines = []
        for line in lines:
            if not line or line.startswith(":"):
                continue
            if line.startswith("data:"):
                value = line[5:]
                if value.startswith(" "):
                    value = value[1:]
                data_lines.append(value)
        if not data_lines:
            return None
        return "\n".join(data_lines)
