"""Export the OpenAPI document.

python scripts/export_openapi.py docs/api/openapi.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    """Write the schema to the given path, defaulting to docs/api/openapi.json."""
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/api/openapi.json")
    from emrag.api.main import create_app

    schema = create_app().openapi()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    sys.stdout.write(f"wrote {target} ({len(schema.get('paths', {}))} paths)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
