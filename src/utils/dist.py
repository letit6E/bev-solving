"""Distributed-training helpers used by v6/v7 multi-GPU notebooks."""
import os

import torch
import torch.distributed as dist


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank():
    return int(os.environ.get("RANK", "0"))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_dist_enabled(cfg=None):
    use_ddp = bool(cfg.get("use_ddp", False)) if cfg else False
    return use_ddp and get_world_size() > 1


def is_main_process():
    return get_rank() == 0


def setup_distributed(cfg=None):
    if not is_dist_enabled(cfg):
        return
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(get_local_rank())


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
