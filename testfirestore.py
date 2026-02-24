import json
import os
from pathlib import Path

from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore


def resolve_credentials() -> credentials.Certificate | None:
    """Build a credentials object from FIREBASE_CREDENTIALS."""
    source = os.environ.get("FIREBASE_CREDENTIALS")
    if not source:
        print("FIREBASE_CREDENTIALS is not set.")
        return None

    path = Path(source)
    if path.exists():
        print(f"Using service account file: {path}")
        return credentials.Certificate(str(path))

    try:
        data = json.loads(source)
    except json.JSONDecodeError as exc:
        print(f"FIREBASE_CREDENTIALS is neither a readable file nor valid JSON: {exc}")
        return None

    print("Using service account JSON provided via environment variable.")
    return credentials.Certificate(data)


def ensure_firebase_app(cred: credentials.Certificate | None) -> firebase_admin.App | None:
    try:
        return firebase_admin.get_app()
    except ValueError:
        pass

    try:
        return firebase_admin.initialize_app(cred)
    except Exception as exc:
        print(f"Firebase initialisation failed: {exc}")
        return None


def test_firestore_connection(app: firebase_admin.App | None) -> None:
    if app is None:
        print("Firestore test skipped because the Firebase app was not initialised.")
        return

    try:
        client = firestore.client(app=app)
    except Exception as exc:
        print(f"Unable to build Firestore client: {exc}")
        return

    try:
        docs = list(client.collection("test_connection").limit(1).stream())
    except Exception as exc:
        print(f"Firestore access failed: {exc}")
        return

    print("Firestore client initialised successfully.")
    if docs:
        for doc in docs:
            print(f"Found document {doc.id}: {doc.to_dict()}")
    else:
        print("No documents found in test_connection collection (this is expected if it is empty).")


def main() -> None:
    load_dotenv()
    cred = resolve_credentials()
    app = ensure_firebase_app(cred)
    test_firestore_connection(app)


if __name__ == "__main__":
    main()