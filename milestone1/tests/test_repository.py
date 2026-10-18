"""Checks for the published manifest bundle and portable Colab launcher."""
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_frozen_manifest_integrity_and_separation(self):
        prepared=ROOT/'prepared'
        audit=json.loads((prepared/'audit.json').read_text())
        mapping=json.loads((prepared/'class_mapping.json').read_text())
        self.assertEqual(audit['seed'],42)
        self.assertEqual(audit['split_counts'],{'train':5851,'val':735,'test':729})
        self.assertEqual(len(mapping),73)
        self.assertEqual(hashlib.sha256((prepared/'class_mapping.json').read_bytes()).hexdigest(),audit['class_mapping_sha256'])
        source_mapping=ROOT/'metadata/object_mapping.csv'
        provenance=json.loads((ROOT/'metadata/object_mapping.provenance.json').read_text())
        self.assertEqual(hashlib.sha256(source_mapping.read_bytes()).hexdigest(),provenance['mapping_sha256'])
        seen={field:set() for field in ('path','sha256','pixel_sha256')}
        for split,count in audit['split_counts'].items():
            content=(prepared/f'{split}.csv').read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(),audit['manifest_sha256'][f'{split}.csv'])
            rows=list(csv.DictReader(io.StringIO(content.decode())))
            self.assertEqual(len(rows),count)
            self.assertEqual({int(row['class_index']) for row in rows},set(range(73)))
            for field in seen:
                values={row[field] for row in rows}
                self.assertEqual(len(values),count)
                self.assertFalse(values&seen[field])
                seen[field].update(values)

    def test_portable_launcher_help_without_gpu_or_private_config(self):
        result=subprocess.run([sys.executable,str(ROOT/'benchmark/run_colab_benchmark.py'),'--help'],
                              cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertIn('--prepared-dir',result.stdout)
        self.assertIn('--raw-snapshot',result.stdout)


if __name__=='__main__':unittest.main()
