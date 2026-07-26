from pathlib import Path

import pytest

from training_sdk import objects


def test_put_and_get_round_trip(s3_base: str) -> None:
    uri = f"{s3_base}/round/trip.json"
    objects.put_bytes(uri, b'{"value": 1}')
    assert objects.get_bytes(uri) == b'{"value": 1}'
    assert objects.exists(uri) is True


def test_missing_object_is_not_reported_as_present(s3_base: str) -> None:
    assert objects.exists(f"{s3_base}/absent") is False


def test_put_file_uploads_contents(s3_base: str, tmp_path: Path) -> None:
    source = tmp_path / "episode.rrd"
    source.write_bytes(b"rrd-bytes")
    objects.put_file(f"{s3_base}/episodes/episode.rrd", source)
    assert objects.get_bytes(f"{s3_base}/episodes/episode.rrd") == b"rrd-bytes"


def test_split_uri_rejects_non_s3_uri() -> None:
    with pytest.raises(ValueError, match="not an s3 uri"):
        objects.split_uri("https://example.com/key")
