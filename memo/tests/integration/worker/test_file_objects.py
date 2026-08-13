from pathlib import Path

from worker import objects


def test_file_object_store_round_trip(tmp_path: Path) -> None:
    uri = (tmp_path / "nested" / "payload.json").resolve().as_uri()

    objects.put_bytes(uri, b'{"value": 1}')

    assert objects.get_bytes(uri) == b'{"value": 1}'
    assert objects.exists(uri) is True


def test_file_object_store_uploads_a_file(tmp_path: Path) -> None:
    source = tmp_path / "source.rrd"
    source.write_bytes(b"rerun")
    destination = (tmp_path / "artifacts" / "episode.rrd").resolve().as_uri()

    objects.put_file(destination, source)

    assert objects.get_bytes(destination) == b"rerun"
