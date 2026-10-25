"""Generate an API key and its Argon2 hash.

The plaintext key is printed once; only the hash is ever stored.

    python scripts/make_api_key.py --tenant acme --role writer --group finance
"""

from __future__ import annotations

import argparse
import json
import sys

from emrag.security.auth import generate_api_key


def main() -> int:
    """Print a key plus the tenant config fragment to paste into configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--role", choices=["reader", "writer", "admin"], default="reader")
    parser.add_argument("--group", action="append", default=[], help="repeatable ACL group")
    parser.add_argument("--key-id", default=None)
    args = parser.parse_args()

    token, digest = generate_api_key(args.tenant)
    entry = {
        "key_id": args.key_id or f"{args.tenant}-{args.role}",
        "role": args.role,
        "groups": args.group or ["all"],
        "argon2_hash": digest,
    }
    sys.stdout.write("API KEY (store it now, it is not recoverable):\n")
    sys.stdout.write(f"  {token}\n\n")
    sys.stdout.write("Add this entry to the tenant's `keys` list:\n")
    sys.stdout.write(json.dumps(entry, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
