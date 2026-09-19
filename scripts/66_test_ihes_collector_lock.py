"""Regression tests for transient and persistent Windows status-file locks."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('collector',ROOT/'scripts/37_collect_ihes_kaggle.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class LockTests(unittest.TestCase):
    def test_transient_lock_retries_then_publishes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/ihes_pdb') as td:
            real_replace=Path.replace;attempts=[]
            def flaky(path,target):
                attempts.append(1)
                if len(attempts)<3:raise PermissionError('sharing violation')
                return real_replace(path,target)
            with patch.object(c,'OUT',Path(td)),patch.object(Path,'replace',flaky),patch.object(c.time,'sleep'):
                self.assertTrue(c.save_status({'status':'RUNNING'}))
            self.assertEqual(len(attempts),3)
            self.assertEqual(json.loads((Path(td)/'collection_status.json').read_text()),{'status':'RUNNING'})
    def test_persistent_lock_preserves_both_snapshots(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/ihes_pdb') as td:
            folder=Path(td);(folder/'collection_status.json').write_text('{"status":"OLD"}')
            with patch.object(c,'OUT',folder),patch.object(Path,'replace',side_effect=PermissionError('locked')),patch.object(c.time,'sleep'):
                self.assertFalse(c.save_status({'status':'NEW'}))
            self.assertEqual(json.loads((folder/'collection_status.json').read_text())['status'],'OLD')
            self.assertEqual(json.loads((folder/'collection_status.tmp').read_text())['status'],'NEW')
    def test_other_errors_are_not_hidden(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/ihes_pdb') as td:
            with patch.object(c,'OUT',Path(td)),patch.object(Path,'replace',side_effect=OSError('disk error')):
                with self.assertRaises(OSError):c.save_status({'status':'NEW'})

if __name__=='__main__':unittest.main()
