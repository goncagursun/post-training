"""Explicit admin actions: initialize schema and onboard the two demo tenants."""
import argparse
from pathlib import Path

from post_training.auth import issue_api_key
from post_training.database import Base, DatasetRecord, Tenant, make_engine, make_session_factory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["init-db", "onboard-demo"])
    parser.add_argument("--keys-file", default=".env.demo", help="New local file for demo customer keys")
    args = parser.parse_args()
    engine = make_engine()
    try:
        if args.action == "init-db":
            Base.metadata.create_all(engine)
            print("Database schema initialized.")
            return
        keys_file = Path(args.keys_file)
        # Exclusive creation prevents overwriting keys. Mode 0600 limits local access.
        import os
        fd = os.open(keys_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w") as output:
                with make_session_factory(engine).begin() as session:
                    lines = []
                    for letter in ("a", "b"):
                        tenant_id = f"tenant-{letter}"
                        if session.get(Tenant, tenant_id):
                            raise ValueError("Demo tenants already exist; onboarding will not rotate keys")
                        session.add(Tenant(id=tenant_id))
                        session.flush()
                        session.add(DatasetRecord(id=f"dataset_{letter}", tenant_id=tenant_id,
                                                  uri=f"demo://tenant-{letter}/mini-qa-v1"))
                        key = issue_api_key(session, tenant_id)
                        lines.append(f"export TENANT_{letter.upper()}_API_KEY={key}\n")
                    output.writelines(lines)
                    output.flush()
                    os.fsync(output.fileno())
        except BaseException:
            keys_file.unlink(missing_ok=True)
            raise
        print(f"Created two demo tenants and dataset registrations. Keys saved once to {keys_file}.")
        print("Demo dataset references resolve to separate two-example subsets in the dispatcher.")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
