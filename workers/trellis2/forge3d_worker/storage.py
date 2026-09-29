"""Uploads finished assets to Cloudflare R2 (S3-compatible)."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Protocol


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> dict: ...


@dataclass
class R2Storage:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    public_base_url: str | None = None

    def __post_init__(self) -> None:
        import boto3  # imported lazily so tests don't need it

        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{self.account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name="auto",
        )

    def put(self, key: str, data: bytes, content_type: str) -> dict:
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
            # Keys include the seed, so a key's content never changes
            CacheControl="public, max-age=31536000, immutable",
        )
        url = f"{self.public_base_url.rstrip('/')}/{key}" if self.public_base_url else None
        return {"key": key, "url": url}


class InlineStorage:
    """Local development without R2: returns the file in the job output (small files only)."""

    LIMIT = 8 * 1024 * 1024

    def put(self, key: str, data: bytes, content_type: str) -> dict:
        if len(data) > self.LIMIT:
            raise RuntimeError("asset too large to return inline; configure R2")
        return {"key": key, "url": None, "base64": base64.b64encode(data).decode("ascii")}


def storage_from_env() -> Storage:
    names = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    values = [os.environ.get(n) for n in names]
    if all(values):
        return R2Storage(*values, public_base_url=os.environ.get("R2_PUBLIC_BASE_URL"))  # type: ignore[arg-type]
    if any(values):
        missing = [n for n, v in zip(names, values) if not v]
        raise RuntimeError(f"R2 is partly configured; missing {', '.join(missing)}")
    return InlineStorage()
