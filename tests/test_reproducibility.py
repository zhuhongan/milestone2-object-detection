"""A compact seeded generation check against the actual finalized source pool."""
import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generate_dataset import generate,sha256
HERE=Path(__file__).resolve().parents[1]
class SeededGeneration(unittest.TestCase):
 @unittest.skipUnless((HERE.parent/'data/raw/final-local/images').is_dir(),'Finalized local source pool not present')
 def test_source_cycles_cover_every_eligible_photo(self):
  with tempfile.TemporaryDirectory() as temp:
   output=Path(temp)/'coverage'
   with contextlib.redirect_stdout(io.StringIO()):
    generate(HERE.parent/'data/raw/final-local/images',HERE/'annotations/source_annotations.json',HERE/'class_mapping.json',output,counts=(219,73,73),cover_sources=True)
   rows=[json.loads(x) for x in (output/'manifest.jsonl').read_text().splitlines()]
   eligible=json.loads((output/'eligible_sources.json').read_text())
   for split in ['train','val','test']:
    used={s['path'] for r in rows if r['split']==split for s in r['sources']}
    self.assertEqual(used,set(eligible[split]))
   from validate_dataset import validate
   self.assertTrue(validate(output)['passed'])
 @unittest.skipUnless((HERE.parent/'data/raw/final-local/images').is_dir(),'Finalized local source pool not present')
 def test_insufficient_coverage_budget_fails_before_writing(self):
  with tempfile.TemporaryDirectory() as temp:
   output=Path(temp)/'insufficient'
   with self.assertRaises(ValueError):
    generate(HERE.parent/'data/raw/final-local/images',HERE/'annotations/source_annotations.json',HERE/'class_mapping.json',output,counts=(73,73,73),cover_sources=True)
   self.assertFalse(output.exists())
 @unittest.skipUnless((HERE.parent/'data/raw/final-local/images').is_dir(),'Finalized local source pool not present')
 def test_identical_seed_reproduces_composites_and_manifest(self):
  with tempfile.TemporaryDirectory() as temp:
   a=Path(temp)/'first';b=Path(temp)/'second'
   with contextlib.redirect_stdout(io.StringIO()):
    for output in [a,b]:generate(HERE.parent/'data/raw/final-local/images',HERE/'annotations/source_annotations.json',HERE/'class_mapping.json',output,counts=(3,3,3))
   self.assertEqual((a/'manifest.jsonl').read_bytes(),(b/'manifest.jsonl').read_bytes())
   rows=[json.loads(x) for x in (a/'manifest.jsonl').read_text().splitlines()]
   for r in rows:self.assertEqual(sha256(a/r['image']),sha256(b/r['image']))
if __name__=='__main__':unittest.main()
