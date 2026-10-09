"""A compact seeded generation check against the actual finalized source pool."""
import contextlib,io,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generate_dataset import generate,sha256
HERE=Path(__file__).resolve().parents[1]
class SeededGeneration(unittest.TestCase):
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
