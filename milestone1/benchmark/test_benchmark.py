"""GPU behavioral checks on all four architectures; no dataset test images used."""
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmark.train_benchmark import Classifier, MODELS, loaders_for
from test_training import fixture
import train
from types import SimpleNamespace

@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for four-backbone behavioral checks')
class BenchmarkTests(unittest.TestCase):
    def test_all_backbones_freeze_then_update(self):
        for name in MODELS:
            with self.subTest(model=name):
                train.seed_everything(42)
                model=Classifier(name,2,pretrained=False).cuda()
                reference=next(model.backbone.parameters())
                before=reference.detach().clone()
                head=next(model.head.parameters()); head_before=head.detach().clone()
                x=torch.randn(2,3,224,224,device='cuda'); y=torch.tensor([0,1],device='cuda')
                for frozen in (True,False):
                    model.set_phase(frozen); model.train()
                    self.assertEqual(model.backbone.training,not frozen)
                    self.assertTrue(model.head.training)
                    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss=torch.nn.functional.cross_entropy(model(x),y)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
                    if frozen:
                        self.assertTrue(torch.equal(reference,before));self.assertFalse(torch.equal(head,head_before))
                    else:self.assertFalse(torch.equal(reference,before))
                del model,optimizer,x,y,loss,reference,head,before,head_before
                gc.collect();torch.cuda.empty_cache()

    def test_two_phase_budget_selection_resume_and_data_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw,prepared,_=fixture(root)
            cmd=[sys.executable,str(Path(__file__).with_name('train_benchmark.py')),
                 '--raw-root',str(raw),'--prepared-dir',str(prepared),'--output',str(root/'run'),
                 '--epochs','2','--frozen-epochs','1','--batch-size','16','--workers','2',
                 '--models','mobilenetv2_140','--smoke']
            subprocess.run(cmd,check=True)
            run=root/'run'; history=json.loads((run/'mobilenetv2_140/history.json').read_text())
            self.assertEqual([r['phase'] for r in history],['frozen','finetuned'])
            self.assertTrue(all(r['optimizer_updates']==1 and r['skipped_optimizer_steps']==0 for r in history))
            self.assertFalse(json.loads((run/'selection.json').read_text())['test_used_for_selection'])
            self.assertFalse(json.loads((run/'completed.json').read_text())['pretrained'])
            previous=train.sha256(run/'mobilenetv2_140/test_predictions.csv')
            subprocess.run([*cmd,'--resume'],check=True)
            self.assertEqual(previous,train.sha256(run/'mobilenetv2_140/test_predictions.csv'))
            args=SimpleNamespace(raw_root=raw,prepared_dir=prepared,seed=42,batch_size=16,workers=2,device='cuda')
            _,manifests,_=train.read_inputs(args)
            first=next(iter(loaders_for(args,manifests,7)['train']))
            second=next(iter(loaders_for(args,manifests,7)['train']))
            self.assertTrue(torch.equal(first[0],second[0]));self.assertTrue(torch.equal(first[2],second[2]))

if __name__=='__main__':unittest.main()
