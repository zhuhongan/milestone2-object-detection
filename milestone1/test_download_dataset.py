"""Focused checks for acquisition completeness and immutable-byte validation."""

import hashlib
from pathlib import Path
import tempfile
import threading
import unittest

from download_dataset import DriveReader, download_one, safe_name


class Response:
    def __init__(self, value=None, content=b""):
        self.value = value
        self.content = content

    def json(self):
        return self.value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, _):
        yield self.content


class AcquisitionTests(unittest.TestCase):
    def reader_with_pages(self, pages):
        reader = object.__new__(DriveReader)
        calls = []

        def get(route, params):
            calls.append((route, params.copy()))
            return Response(pages[len(calls) - 1])

        reader.get = get
        return reader, calls

    def test_pagination_reads_every_page(self):
        reader, calls = self.reader_with_pages([
            {"files": [{"id": "a", "name": "first"}], "nextPageToken": "p2"},
            {"files": [{"id": "b", "name": "second"}]},
        ])
        self.assertEqual([f["id"] for f in reader.children("root")], ["a", "b"])
        self.assertNotIn("pageToken", calls[0][1])
        self.assertEqual(calls[1][1]["pageToken"], "p2")

    def test_incomplete_listing_fails(self):
        reader, _ = self.reader_with_pages([{"files": [], "incompleteSearch": True}])
        with self.assertRaisesRegex(RuntimeError, "incompleteSearch"):
            reader.children("root")

    def test_repeated_pagination_token_fails(self):
        reader, _ = self.reader_with_pages([
            {"files": [], "nextPageToken": "again"},
            {"files": [], "nextPageToken": "again"},
        ])
        with self.assertRaisesRegex(RuntimeError, "repeated a token"):
            reader.children("root")

    def test_repeated_file_id_fails(self):
        reader, _ = self.reader_with_pages([
            {"files": [{"id": "a", "name": "first"}], "nextPageToken": "p2"},
            {"files": [{"id": "a", "name": "first"}]},
        ])
        with self.assertRaisesRegex(RuntimeError, "repeated file IDs"):
            reader.children("root")

    def test_names_cannot_escape_snapshot(self):
        for name in ["", ".", "..", "../outside", "/absolute", "a\\b", "\x00"]:
            with self.assertRaises(RuntimeError):
                safe_name(name)
        self.assertEqual(safe_name("images_OBJ001"), "images_OBJ001")

    def test_403_is_reported_once_without_retry(self):
        calls = []

        class Denied:
            ok = False
            status_code = 403
            text = "Your computer or network may be sending automated queries"

            def close(self):
                pass

        class Session:
            def get(self, *args, **kwargs):
                calls.append(args)
                return Denied()

        reader = object.__new__(DriveReader)
        reader.cancelled = threading.Event()
        reader.api_key = None
        reader.session = lambda: Session()
        with self.assertRaisesRegex(RuntimeError, "automated-query rate protection"):
            reader.get("files/test", {"alt": "media"})
        self.assertEqual(len(calls), 1)

    def test_cancelled_reader_does_not_start_new_requests(self):
        reader = object.__new__(DriveReader)
        reader.cancelled = threading.Event()
        reader.cancelled.set()
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            reader.get("files/test")

    def test_download_verifies_and_reuses_but_never_overwrites_bad_cache(self):
        content = b"test image contents"
        calls = []

        class Reader:
            def get(self, *args, **kwargs):
                calls.append(args)
                return Response(content=content)

        item = {"id": "a", "relative_path": "images_OBJ001/image.jpg",
                "size": str(len(content)), "md5Checksum": hashlib.md5(content).hexdigest()}
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            first = download_one(Reader(), staging, item)
            self.assertEqual(first["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(first, download_one(Reader(), staging, item))
            self.assertEqual(len(calls), 1)
            path = staging / "images" / item["relative_path"]
            path.write_bytes(b"changed local content")
            with self.assertRaisesRegex(RuntimeError, "Cached file mismatch"):
                download_one(Reader(), staging, item)
            self.assertEqual(path.read_bytes(), b"changed local content")

    def test_wrong_bytes_are_never_published_as_raw_image(self):
        class Reader:
            def get(self, *args, **kwargs):
                return Response(content=b"wrong data")

        item = {"id": "a", "relative_path": "images_OBJ001/image.jpg",
                "size": "3", "md5Checksum": hashlib.md5(b"yes").hexdigest()}
        with tempfile.TemporaryDirectory() as tmp:
            staging = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "size/MD5 mismatch"):
                download_one(Reader(), staging, item)
            self.assertFalse((staging / "images" / item["relative_path"]).exists())


if __name__ == "__main__":
    unittest.main()
