"""Separate two-rank additive BCE correctness test (not initial tiny subset)."""

import torch
import torch.nn.functional as F
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel

from uncertainty_retrieval.models.pair_confidence import PairConfidenceNetwork


def batch():
    generator=torch.Generator().manual_seed(42)
    return torch.randn(4,2,generator=generator),torch.randn(4,2,generator=generator),torch.tensor([0.,1.,0.,1.])


def worker(rank,init_file,output):
    torch.distributed.init_process_group("gloo",init_method=f"file://{init_file}",rank=rank,world_size=2)
    try:
        torch.manual_seed(42)
        model=PairConfidenceNetwork(2).eval()  # disable dropout for comparison
        ddp=DistributedDataParallel(model)
        q,x,target=batch()
        loss=F.binary_cross_entropy_with_logits(ddp(q[rank::2],x[rank::2]),target[rank::2])
        loss.backward()
        mean=loss.detach().clone()
        torch.distributed.all_reduce(mean)
        if rank==0:
            torch.save({"loss":mean/2,"gradients":{name:p.grad for name,p in model.named_parameters()}},output)
    finally:
        torch.distributed.destroy_process_group()


def test_pair_bce_ddp_matches_single_global_batch(tmp_path):
    output=tmp_path / "result.pt"
    spawn(worker,args=(str(tmp_path/"init"),str(output)),nprocs=2,join=True)
    distributed=torch.load(output,map_location="cpu",weights_only=True)
    torch.manual_seed(42)
    reference=PairConfidenceNetwork(2).eval()
    q,x,target=batch()
    loss=F.binary_cross_entropy_with_logits(reference(q,x),target)
    loss.backward()
    assert torch.allclose(distributed["loss"],loss,atol=1e-6)
    for name,p in reference.named_parameters():
        assert torch.allclose(distributed["gradients"][name],p.grad,atol=1e-6,rtol=1e-5)
