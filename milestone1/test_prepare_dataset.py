"""Behavioral checks for immutable sources, exact duplicates, and split leakage."""

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin

from prepare_dataset import DatasetPreparationError, inspect_image, prepare_dataset, read_mapping, sha256_file, split_counts


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "locked_raw"
        self.raw.mkdir()
        self.mapping = self.root / "official_mapping.csv"
        self.mapping.write_text("folder,object_id,object_name\nBeta,010,Second object\nAlpha,001,First object\n", encoding="utf-8")
        self.output = self.root / "prepared"
        for folder, offset in (("Alpha", 0), ("Beta", 50)):
            (self.raw / folder).mkdir()
            for index in range(11):
                self.make_image(self.raw / folder / f"image_{index:03d}.png", index + offset)

    @staticmethod
    def make_image(path, value, size=None):
        image = Image.new("RGB", size or (8 + value % 3, 9 + value % 5), (value % 256, value * 31 % 256, value * 73 % 256))
        image.save(path)
        image.close()

    def records(self, name):
        with (self.output / f"{name}.csv").open(newline="") as handle:
            return list(csv.DictReader(handle))

    def snapshot_raw(self):
        return {path.relative_to(self.raw).as_posix(): sha256_file(path) for path in self.raw.rglob("*") if path.is_file() and not path.is_symlink()}

    def test_duplicates_excluded_before_reproducible_stratified_split(self):
        canonical = self.raw / "Alpha" / "image_000.png"
        shutil.copyfile(canonical, self.raw / "Alpha" / "raw_duplicate.png")
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("comment", "Different bytes, same decoded image")
        with Image.open(canonical) as source:
            source.save(self.raw / "Alpha" / "pixel_duplicate.png", pnginfo=metadata)
        (self.raw / "Alpha" / "broken.jpg").write_bytes(b"this is not an image")
        (self.raw / ".DS_Store").write_bytes(b"metadata")
        before = self.snapshot_raw()
        audit = prepare_dataset(self.raw, self.mapping, self.output)
        self.assertEqual(audit["split_counts"], {"train": 18, "val": 2, "test": 2})
        self.assertEqual(audit["retained_images"], 22)
        self.assertEqual(audit["exclusion_counts"], {"exact_file_duplicate": 1, "exact_rgb_pixel_duplicate": 1, "invalid_or_unsupported_image": 1, "unsupported_nonimage_file": 1})
        self.assertEqual(before, self.snapshot_raw())
        seen_pixels, seen_bytes = set(), set()
        for split in ("train", "val", "test"):
            records = self.records(split)
            self.assertEqual({row["object_id"] for row in records}, {"001", "010"})
            for row in records:
                self.assertNotIn(row["pixel_sha256"], seen_pixels)
                self.assertNotIn(row["sha256"], seen_bytes)
                seen_pixels.add(row["pixel_sha256"])
                seen_bytes.add(row["sha256"])
        first_manifests = {name: (self.output / f"{name}.csv").read_bytes() for name in ("train", "val", "test")}
        prepare_dataset(self.raw, self.mapping, self.output, seed=42)
        self.assertEqual(first_manifests, {name: (self.output / f"{name}.csv").read_bytes() for name in first_manifests})
        prepare_dataset(self.raw, self.mapping, self.output, seed=43)
        self.assertNotEqual(first_manifests["train"], (self.output / "train.csv").read_bytes())
        classes = json.loads((self.output / "class_mapping.json").read_text())
        self.assertEqual([(row["object_id"], row["class_index"]) for row in classes], [("001", 0), ("010", 1)])

    def test_cross_label_duplicate_stops_and_marks_old_manifests_unusable(self):
        prepare_dataset(self.raw, self.mapping, self.output)
        shutil.copyfile(self.raw / "Alpha" / "image_000.png", self.raw / "Beta" / "conflicting_label.png")
        before = self.snapshot_raw()
        with self.assertRaisesRegex(DatasetPreparationError, "conflicting Object IDs"):
            prepare_dataset(self.raw, self.mapping, self.output)
        self.assertEqual(before, self.snapshot_raw())
        audit = json.loads((self.output / "audit.json").read_text())
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["cross_label_duplicate_groups"], 1)
        conflicts = json.loads((self.output / "conflicts.json").read_text())
        self.assertEqual({row["object_id"] for row in conflicts[0]["images"]}, {"001", "010"})

    def test_pixel_identity_honors_exif_orientation(self):
        original = self.root / "original.png"
        oriented = self.root / "oriented.png"
        image = Image.new("RGB", (2, 3))
        image.putdata([(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0), (255, 255, 255), (80, 90, 100)])
        exif = Image.Exif()
        exif[274] = 6
        image.save(original, exif=exif)
        image.transpose(Image.Transpose.ROTATE_270).save(oriented)
        image.close()
        self.assertNotEqual(sha256_file(original), sha256_file(oriented))
        self.assertEqual(inspect_image(original)["pixel_sha256"], inspect_image(oriented)["pixel_sha256"])

    def test_rgb_and_jpeg_extensions_and_large_dimension_heuristic(self):
        self.make_image(self.raw / "Alpha" / "different_dimensions.jpeg", 99, (40, 30))
        image = Image.new("L", (13, 29), 182)
        image.save(self.raw / "Beta" / "grayscale.JPG")
        image.close()
        old_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = 16
        try:
            audit = prepare_dataset(self.raw, self.mapping, self.output)
            self.assertEqual(Image.MAX_IMAGE_PIXELS, 16)
        finally:
            Image.MAX_IMAGE_PIXELS = old_limit
        self.assertEqual(audit["retained_images"], 24)
        self.assertEqual(audit["exclusion_counts"], {})

    def test_multiframe_mpo_jpeg_uses_primary_photo_not_auxiliary_frame(self):
        path = self.raw / "Alpha" / "phone_primary.jpg"
        primary = Image.new("RGB", (32, 24), (80, 120, 200))
        auxiliary = Image.new("RGB", (8, 6), (10, 10, 10))
        primary.save(path, format="MPO", save_all=True, append_images=[auxiliary])
        primary.close()
        auxiliary.close()
        with Image.open(path) as image:
            self.assertEqual(image.format, "MPO")
            self.assertEqual(image.n_frames, 2)
            image.seek(0)
            image.convert("RGB").save(self.root / "primary_decoded.png")
        before = self.snapshot_raw()
        inspected = inspect_image(path)
        self.assertEqual(inspected["source_frames"], 2)
        self.assertEqual(inspected["selected_frame"], 0)
        self.assertEqual((inspected["width"], inspected["height"]), (32, 24))
        self.assertEqual(inspected["pixel_sha256"], inspect_image(self.root / "primary_decoded.png")["pixel_sha256"])
        audit = prepare_dataset(self.raw, self.mapping, self.output)
        self.assertEqual(audit["retained_images"], 23)
        self.assertEqual(audit["valid_multiframe_jpeg_mpo_primary_images"], 1)
        self.assertEqual(audit["exclusion_counts"], {})
        self.assertEqual(self.snapshot_raw(), before)

    def test_other_multiframe_format_has_explicit_exclusion_reason(self):
        first = Image.new("RGB", (10, 10), "red")
        second = Image.new("RGB", (10, 10), "blue")
        first.save(self.raw / "Alpha" / "animation.gif", save_all=True, append_images=[second])
        first.close()
        second.close()
        audit = prepare_dataset(self.raw, self.mapping, self.output)
        self.assertEqual(audit["retained_images"], 22)
        self.assertEqual(audit["exclusion_counts"], {"unsupported_multiframe_format": 1})

    def test_more_than_100_valid_images_are_retained(self):
        self.mapping.write_text("folder,object_id,object_name\nAlpha,001,First object\n", encoding="utf-8")
        shutil.rmtree(self.raw / "Beta")
        for index in range(11, 103):
            self.make_image(self.raw / "Alpha" / f"image_{index:03d}.png", index)
        audit = prepare_dataset(self.raw, self.mapping, self.output)
        self.assertEqual(audit["retained_images"], 103)
        self.assertEqual(audit["split_counts"], {"train": 83, "val": 10, "test": 10})
        self.assertEqual(audit["exclusion_counts"], {})

    def test_source_output_and_symlink_escape_protection(self):
        before = self.snapshot_raw()
        with self.assertRaisesRegex(DatasetPreparationError, "outside the locked"):
            prepare_dataset(self.raw, self.mapping, self.raw / "prepared")
        self.assertFalse((self.raw / "prepared").exists())
        outside = self.root / "outside.jpg"
        self.make_image(outside, 192)
        (self.raw / "Alpha" / "outside_link.jpg").symlink_to(outside)
        audit = prepare_dataset(self.raw, self.mapping, self.output)
        self.assertEqual(audit["exclusion_counts"], {"symlink_outside_raw_root": 1})
        self.assertEqual(self.snapshot_raw(), before)

    def test_unmapped_images_or_nonunique_ids_block(self):
        (self.raw / "Unknown").mkdir()
        self.make_image(self.raw / "Unknown" / "valid.png", 221)
        with self.assertRaisesRegex(DatasetPreparationError, "no official"):
            prepare_dataset(self.raw, self.mapping, self.output)
        self.assertEqual(json.loads((self.output / "audit.json").read_text())["unmapped_image_paths"], ["Unknown/valid.png"])
        self.mapping.write_text("folder,object_id,object_name\nAlpha,001,First\nBeta,001,Second\n", encoding="utf-8")
        with self.assertRaisesRegex(DatasetPreparationError, "object_id values must be unique"):
            read_mapping(self.mapping)

    def test_too_few_unique_samples_blocks_instead_of_silent_class_loss(self):
        for path in sorted((self.raw / "Alpha").glob("*.png"))[3:]:
            path.unlink()
        with self.assertRaisesRegex(DatasetPreparationError, "nonempty"):
            prepare_dataset(self.raw, self.mapping, self.output)
        self.assertEqual(split_counts(100), {"train": 80, "val": 10, "test": 10})
        self.assertEqual(split_counts(101), {"train": 81, "val": 10, "test": 10})


if __name__ == "__main__":
    unittest.main()
