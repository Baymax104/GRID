import pytest

from src.data.components.readers import TFRecordReader


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("file://E:/data/a%20b.gz", "E:/data/a b.gz"),
        ("file:///E:/data/a.gz", "E:/data/a.gz"),
        ("file://server/share/a.gz", "//server/share/a.gz"),
        ("file://localhost/data/a.gz", "/data/a.gz"),
        ("E:/data/a.gz", "E:/data/a.gz"),
    ],
)
def test_tfrecord_local_uri_normalization(uri, expected):
    assert TFRecordReader._normalize_example_loader_path(uri) == expected
