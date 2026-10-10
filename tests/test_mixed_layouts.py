"""Verify occupied grid cells, exact mixed-layout budgets and deterministic geometry."""
import contextlib
import io
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generate_mixed_dataset import cell_rectangle, grid_geometry, generate, layout_counts
from generate_dataset import sha256
from validate_dataset import validate


class MixedLayouts(unittest.TestCase):
    def test_exact_layout_budgets(self):
        self.assertEqual(layout_counts(8000), {'2x2': 3200, '3x3': 3200, 'random': 1600})
        self.assertEqual(layout_counts(1000), {'2x2': 400, '3x3': 400, 'random': 200})

    def test_cells_partition_canvas_and_letterboxing_stays_inside(self):
        for side in (2, 3):
            cells = [cell_rectangle(side, i, 640) for i in range(side*side)]
            self.assertEqual(sum((r-l)*(b-t) for l,t,r,b in cells), 640*640)
            for i, cell in enumerate(cells):
                for width, height in [(224,224),(400,100),(100,400),(1024,768)]:
                    pw, ph, ox, oy = grid_geometry(width,height,cell,random.Random(42+i))
                    self.assertTrue(cell[0] <= ox < ox+pw <= cell[2])
                    self.assertTrue(cell[1] <= oy < oy+ph <= cell[3])
                    self.assertLessEqual(abs(pw/width-ph/height),1/width+1/height)

    def test_complete_generation_reproducibility_and_reject_invalid_grid(self):
        # Independent synthetic fixtures exercise every real class and all three splits.
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);raw=root/'raw';raw.mkdir();mapping=[];sources=[]
            for c in range(73):
                mapping.append({'class_index':c,'object_id':f'OBJ{c+1:03d}','object_name':f'Object {c+1}'})
                for j,split in enumerate(('train','val','test')):
                    name=f'{split}_{c}.png';im=Image.new('RGB',(32,24),(c*3%256,j*75,100));im.save(raw/name)
                    digest=sha256(raw/name)
                    sources.append({**mapping[c],'path':name,'sha256':digest,'pixel_sha256':digest,'split':split,'boxes_xyxy_normalized':[[.1,.2,.8,.9]]})
            classes=root/'classes.json';classes.write_text(json.dumps(mapping))
            annotations=root/'annotations.json';annotations.write_text(json.dumps({'selected_sources':sources}))
            first,second=root/'first',root/'second'
            with contextlib.redirect_stdout(io.StringIO()):
                for output in (first,second):generate(raw,annotations,classes,output,counts=(365,365,365))
            self.assertEqual((first/'manifest.jsonl').read_bytes(),(second/'manifest.jsonl').read_bytes())
            self.assertTrue(validate(first)['passed'])
            rows=[json.loads(s) for s in (first/'manifest.jsonl').read_text().splitlines()]
            row=next(r for r in rows if r['layout']=='3x3')
            self.assertEqual(len(row['sources']),9)
            self.assertEqual(len(row['targets']),9)
            row['sources'][0]['cell_index']=row['sources'][1]['cell_index']
            (first/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError,'Duplicate or missing grid cell'):validate(first)


if __name__=='__main__':unittest.main()
