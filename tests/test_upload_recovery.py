"""Temporary network failures retry the same part without starting another session."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_colab import upload_with_retries,run_cli_checked

class UploadRecovery(unittest.TestCase):
 def test_disconnected_client_is_terminated_instead_of_hanging(self):
  import time
  start=time.monotonic()
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(RuntimeError,'websocket disconnected'):
   run_cli_checked([sys.executable,'-u','-c'],"import time;print('RuntimeError: Connection was lost.');time.sleep(60)")
  self.assertLess(time.monotonic()-start,10)

 def test_retry_uses_identical_session_and_destination(self):
  calls=[]
  def cli(*args):
   calls.append(args)
   if len(calls)<3:raise RuntimeError('Transient transport failure')
   return 'uploaded'
  with contextlib.redirect_stdout(io.StringIO()):
   result=upload_with_retries(cli,'owned-session',Path('/tmp/part'),'/content/part',delay=lambda _:None)
  self.assertEqual(result,'uploaded');self.assertEqual(len(calls),3)
  self.assertEqual(calls,[calls[0]]*3)

 def test_exhausted_upload_still_fails(self):
  calls=[]
  def cli(*args):
   calls.append(args);raise RuntimeError('Transport unavailable')
  with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(RuntimeError):
   upload_with_retries(cli,'owned-session','/tmp/part','/content/part',attempts=2,delay=lambda _:None)
  self.assertEqual(len(calls),2)

if __name__=='__main__':unittest.main()
