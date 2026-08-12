"""Regression coverage for PyTorch distributed evaluation helpers."""

import multiprocessing
import os
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch.distributed as dist
except ImportError:
    raise unittest.SkipTest("Skipping PyTorch distributed tests because torch is unavailable.")

from faster_coco_eval.utils.pytorch.coco_eval import all_gather


def _all_gather_worker(rank, init_file, results):
    """Gather a rank-specific payload through a two-rank gloo group."""
    dist.init_process_group("gloo", rank=rank, world_size=2, init_method=Path(init_file).absolute().as_uri())
    try:
        results.put(all_gather({"rank": rank, "payload": "x" * (rank + 1)}))
    finally:
        dist.destroy_process_group()


class TestAllGather(unittest.TestCase):
    """Test distributed collection of picklable evaluator data."""

    def test_gloo_gathers_unequal_cpu_payloads(self):
        """Gather unequal payloads without initializing CUDA for gloo."""
        if not dist.is_gloo_available():
            self.skipTest("Gloo is unavailable in this PyTorch build.")

        with tempfile.NamedTemporaryFile(delete=False) as init_handle:
            init_file = init_handle.name

        # FileStore expects to create the rendezvous file itself.
        os.unlink(init_file)
        original_interface = os.environ.get("GLOO_SOCKET_IFNAME")
        if sys.platform == "darwin":
            os.environ["GLOO_SOCKET_IFNAME"] = "lo0"

        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        processes = [context.Process(target=_all_gather_worker, args=(rank, init_file, results)) for rank in range(2)]
        try:
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=30)
                self.assertFalse(process.is_alive(), "gloo worker did not exit")
                self.assertEqual(process.exitcode, 0)

            expected = [{"rank": 0, "payload": "x"}, {"rank": 1, "payload": "xx"}]
            self.assertEqual([results.get(timeout=5) for _ in processes], [expected, expected])
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join()
            results.close()
            if original_interface is None:
                os.environ.pop("GLOO_SOCKET_IFNAME", None)
            else:
                os.environ["GLOO_SOCKET_IFNAME"] = original_interface
            if os.path.exists(init_file):
                os.unlink(init_file)
