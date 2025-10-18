import os
import time
import argparse
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.amp import GradScaler
import timm
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp import MixedPrecision, BackwardPrefetch
from torch.distributed.fsdp.wrap import wrap
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.distributed.fsdp import StateDictType
import torch.distributed.checkpoint as dist_cp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullOptimStateDictConfig
import pickle
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function

from torchvision.datasets import ImageFolder
from torchvision import transforms
import os
from pickle import dump

import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import re
import glob

import torch.nn.functional as F

from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

# Training pipeline
@pipeline_def
def imagenet_train_pipeline(data_dir, crop=224, shard_id=0, num_shards=1, dali_cpu=False):
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=True,
        shard_id=shard_id,
        num_shards=num_shards,
        name="Reader"
    )

    decode_device = "cpu" if dali_cpu else "mixed"
    images = fn.decoders.image(jpegs, device=decode_device, output_type=types.RGB)

    images = fn.random_resized_crop(images, size=crop)
    images = fn.flip(images, horizontal=fn.random.coin_flip(probability=0.5))

    images = fn.crop_mirror_normalize(
        images,
        crop=(crop, crop),
        mean=[0.485*255, 0.456*255, 0.406*255],
        std=[0.229*255, 0.224*255, 0.225*255],
        dtype=types.FLOAT,
        output_layout="CHW"
    )
    return images, labels


# Validation pipeline
@pipeline_def
def imagenet_val_pipeline(data_dir, crop=224, shard_id=0, num_shards=1, dali_cpu=False):
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=False,
        shard_id=shard_id,
        num_shards=num_shards,
        name="Reader"
    )

    decode_device = "cpu" if dali_cpu else "mixed"
    images = fn.decoders.image(jpegs, device=decode_device, output_type=types.RGB)
    size = crop
    images = fn.resize(images, resize_shorter=size)
    images = fn.crop(images, crop=(crop, crop))

    images = fn.crop_mirror_normalize(
        images,
        crop=(crop, crop),
        mean=[0.485*255, 0.456*255, 0.406*255],
        std=[0.229*255, 0.224*255, 0.225*255],
        dtype=types.FLOAT,
        output_layout="CHW"
    )
    return images, labels

def get_dali_loader(data_dir, batch_size, num_threads, device_id, crop, is_train=True, world_size=1, rank=0):
    if is_train:
        pipe = imagenet_train_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            data_dir=data_dir,
            crop=crop,
            shard_id=rank,
            num_shards=world_size,
            prefetch_queue_depth=2
        )
    else:
        pipe = imagenet_val_pipeline(
            batch_size=batch_size,
            num_threads=num_threads,
            device_id=device_id,
            data_dir=data_dir,
            crop=crop,
            shard_id=rank,
            num_shards=world_size,
            prefetch_queue_depth=2
        )

    pipe.build()
    return DALIGenericIterator(
        pipelines=pipe,
        output_map=["data", "label"],
        reader_name="Reader",
        auto_reset=True
    )


def vit_mac_flops(
    H, W,
    patch=16,
    D=1024,            # ViT-L hidden size
    depth=24,          # ViT-L depth
    heads=16,          # ViT-L heads (not needed for FLOPs formula directly)
    include_cls_token=True,
    num_classes=6,
    count_flops=True   # True -> return FLOPs (2*MACs), False -> MACs
):
    """
    Returns total compute for ViT-L/16 at arbitrary resolution (H, W).
    By default returns FLOPs assuming 1 MAC = 2 FLOPs.
    """
    assert H % patch == 0 and W % patch == 0, "H and W must be divisible by the patch size."
    N = (H // patch) * (W // patch)
    if include_cls_token:
        N = N + 1

    # ---- Patch embedding (linear projection of flattened patches) ----
    # N_patches excludes cls token
    N_patches = (H // patch) * (W // patch)
    macs_patch_embed = N_patches * (3 * patch * patch) * D

    # ---- Transformer encoder (per layer) ----
    # QKV + proj: (3ND^2 + ND^2) = 4ND^2
    # Attention matmuls: QK^T + AV = 2 * N^2 * D
    # MLP: N * D * 4D + N * 4D * D = 8ND^2
    macs_per_layer = 12 * N * (D ** 2) + 2 * (N ** 2) * D
    macs_transformer = depth * macs_per_layer

    # ---- Classifier head (tiny; included for completeness) ----
    macs_head = D * num_classes

    est_flops = 2*N*patch*patch*3*D + depth*(2*N*3*D*D + 2*N*N*D + 3*heads*N*N + 2*N*N*D + 2*N*D*D + 4*N*D*4*D) + 2*D*N

    total_macs = macs_patch_embed + macs_transformer + macs_head
    if count_flops:
        old = 2 * total_macs  # FLOPs
        print(f"old: {old}, new: {est_flops}")
        return est_flops 
    return total_macs         # MACs


def human_readable(x):
    units = ["", "K", "M", "G", "T", "P"]
    i = 0
    while x >= 1000 and i < len(units)-1:
        x /= 1000.0
        i += 1
    return f"{x:.2f}{units[i]}"


def resize_pos_embed(old_posemb, new_posemb):
    # old_posemb/new_posemb: [1, num_patches+1, dim]
    cls_token = old_posemb[:, :1]
    old_grid = old_posemb[:, 1:]
    gs_old = int(old_grid.shape[1] ** 0.5)
    gs_new = int(new_posemb.shape[1] ** 0.5)

    old_grid = old_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)  # [1, dim, H, W]
    new_grid = F.interpolate(old_grid, size=(gs_new, gs_new), mode='bicubic', align_corners=False)
    new_grid = new_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)

    return torch.cat([cls_token, new_grid], dim=1)



def cleanup_old_checkpoints(output_dir, keep_last=3):
    """Remove older checkpoints, keeping only the most recent `keep_last` ones."""
    ckpts = sorted(
        glob.glob(os.path.join(output_dir, "*.pt")),
        key=lambda x: int(re.search(r"_epoch(\d+)\.pt", x).group(1))
    )
    if len(ckpts) > keep_last:
        for ckpt in ckpts[:-keep_last]:
            os.remove(ckpt)
            print(f"🗑 Removed old checkpoint: {ckpt}")

class FixedIndexImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Fixed class list: directories named 0 to 999
        self.classes = [str(i) for i in range(1000)]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Collect samples from non-empty dirs, but preserve class indices
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(root, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(cls_dir, fname)
                    label = self.class_to_idx[cls_name]
                    self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='vit-4k-fsdp')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--model-name', type=str, default='vit_large_patch16_384')
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--cpt-power', type=float, default=2.0)  # higher = more gradual decay
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--min-lr', type=float, default=1e-6, help='Minimum learning rate')


    return parser.parse_args()

import math  # For noise scale calculation

# 🚀 Utility: Gradient Norm
def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return math.sqrt(total_norm)

# 🚀 Utility: Parameter Norm
def compute_parameter_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        param_norm = p.data.norm(2)
        total_norm += param_norm.item() ** 2
    return math.sqrt(total_norm)

# 🚀 Utility: Optimizer State Norm
def compute_optimizer_state_norm(optimizer):
    total_norm = 0.0
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                total_norm += v.norm(2).item() ** 2
    return math.sqrt(total_norm)

# 🚀 Utility: Gradient Noise Scale (GNS)
def compute_gradient_noise_scale(loss, grad_norm):
    # GNS ~ loss / ||grad||^2 (very rough heuristic)
    if grad_norm > 0:
        return loss.item() / (grad_norm ** 2)
    return 0.0


def setup_ddp():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

import time
from torchvision import transforms

class TimedTransform:
    def __init__(self, transform):
        self.transform = transform
        self.last_time = 0.0

    def __call__(self, x):
        t0 = time.time()
        out = self.transform(x)
        self.last_time = time.time() - t0
        return out

from torch.utils.data.dataloader import default_collate

def timed_collate(batch):
    t0 = time.time()
    out = default_collate(batch)
    timed_collate.last_time = time.time() - t0
    return out

timed_collate.last_time = 0.0


def create_dataloaders_ddp(data_dir, image_size, batch_size, num_workers, rank, world_size):
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")

    transform = TimedTransform(transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ]))
    print(f"train_data_path: {train_path}")
    print(f"val_data_path: {val_path}")
    train_dataset = ImageFolder(train_path, transform=transform)
    val_dataset = ImageFolder(val_path, transform=transform)


    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    #device_id = local_rank
    #num_threads = 8
    #train_loader = get_dali_loader(train_path, batch_size, num_threads, device_id, crop=image_size,
    #                           is_train=True, world_size=world_size, rank=rank)
    #val_loader   = get_dali_loader(val_path, batch_size, num_threads, device_id, crop=image_size,
    #                           is_train=False, world_size=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=num_workers, collate_fn=timed_collate, pin_memory=True,prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler,
                            num_workers=num_workers,collate_fn=timed_collate,  pin_memory=True)

    return train_loader, val_loader, train_sampler

import time

def train_one_epoch(
    model, loader, optimizer, scaler, criterion, device, epoch, rank,
    scheduler, microbatch_size, use_amp=False, max_grad_norm=1.0, measure_timing=False, image_size=384
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    torch.cuda.synchronize()
    t = time.time()
    if measure_timing:
            torch.cuda.synchronize()
            tdata = time.time()
    for i, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        #images = data[0]["data"]    # Already on GPU (CHW, float32)
        #labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        #timings = {}
        '''print(images.is_pinned())
        torch.cuda.synchronize()
        images, labels = images.to(device), labels.to(device)
        if measure_timing:
            torch.cuda.synchronize()
            timings["time_data_load"] = time.time() - tdata
        optimizer.zero_grad()
        if measure_timing:
            timings["time_transforms"] = getattr(loader.dataset.transform, "last_time", 0.0)
            timings["time_collate"] = getattr(loader.collate_fn, "last_time", 0.0)'''

        timings = {}

        # -------------------------
        # Forward
        # -------------------------
        if measure_timing:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if measure_timing:
            torch.cuda.synchronize()
            timings["time_forward"] = time.time() - t0

        # -------------------------
        # Backward
        # -------------------------
        if measure_timing:
            torch.cuda.synchronize()
            t1 = time.time()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if measure_timing:
            torch.cuda.synchronize()
            timings["time_backward"] = time.time() - t1

        # -------------------------
        # Optimizer step
        # -------------------------
        if measure_timing:
            torch.cuda.synchronize()
            t2 = time.time()
        if not torch.isnan(loss):
            scaler.step(optimizer)
            scaler.update()
        if measure_timing:
            torch.cuda.synchronize()
            timings["time_optimizer"] = time.time() - t2

        # -------------------------
        # Scheduler step
        # -------------------------
        if measure_timing:
            torch.cuda.synchronize()
            t3 = time.time()
        scheduler.step()
        if measure_timing:
            torch.cuda.synchronize()
            timings["time_scheduler"] = time.time() - t3

        # -------------------------
        # Metrics
        # -------------------------
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total

        # -------------------------
        # Logging
        # -------------------------
        if rank == 0 and i % 1 == 0:
            torch.cuda.synchronize()
            elapsed_time_per_iter = (time.time() - t)/1 if i > 0 else (time.time() - t)
            flops_per_batch = vit_mac_flops(image_size, image_size)
            print(f"flops_per_batch: {flops_per_batch}")
            flops = 3 * microbatch_size * vit_mac_flops(image_size, image_size)

            print("ViT-L/16 FLOPs:", human_readable(flops), "image_size: ", image_size)
            flops_per_sec_per_gpu = flops / elapsed_time_per_iter
            print(f"ViT-L/16 FLOPs/sec/gpu: {human_readable(flops_per_sec_per_gpu)}, iteration time: {elapsed_time_per_iter} seconds")
            grad_norm_before = compute_gradient_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            grad_norm_after = compute_gradient_norm(model)
            param_norm = compute_parameter_norm(model)
            opt_state_norm = compute_optimizer_state_norm(optimizer)
            noise_scale = compute_gradient_noise_scale(loss, grad_norm_before)

            log_dict = {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "flops_per_gpu": flops_per_sec_per_gpu,
                "train_loss": loss,
                "train_acc": accuracy,
                "grad_norm_before_clip": grad_norm_before,
                "grad_norm_after_clip": grad_norm_after,
                "param_norm": param_norm,
                "opt_state_norm": opt_state_norm,
                "grad_noise_scale": noise_scale,
            }
            if measure_timing:
                log_dict.update(timings)

            print(f"[Epoch {epoch}] Batch {i}/{len(loader)} - Loss: {loss.item():.4f}")
            wandb.log(log_dict)
            torch.cuda.synchronize()
            t = time.time()
        
        if measure_timing:
            torch.cuda.synchronize()
            tdata = time.time()

    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy

import torch.profiler as profiler

def save_snapshot(prof_path):
    snapshot = torch.cuda.memory._snapshot()
    snapshot_path = os.path.join(prof_path)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    with open(os.path.join(snapshot_path, "mem_snapshot_flash.pickle"), "wb") as f:
        dump(snapshot, f)

def train_one_epoch_profile(
    model, loader, optimizer, scaler, criterion, device, epoch, rank,
    scheduler, microbatch_size, use_amp=False, max_grad_norm=1.0, measure_timing=True, image_size=384
    
):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    torch.cuda.synchronize()
    t = time.time()

    # -------------------------
    # Profiler setup
    # -------------------------
    torch.cuda.memory._record_memory_history(
            True,
            # keep a maximum 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=10000000,
            trace_alloc_record_context=True,
        )
    prof = profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        schedule=profiler.schedule(wait=0, warmup=1, active=5, repeat=1),  # only first 5 iters
        on_trace_ready=profiler.tensorboard_trace_handler("./profiler_logs"),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        with_flops=True,
    )
    prof_iters = 5
    prof.start()
    torch.cuda.cudart().cudaProfilerStart()

    for i, (images, labels) in enumerate(loader):
        torch.cuda.nvtx.range_push(f"Get batch")
        images, labels = images.to(device), labels.to(device)
        timings = {}
        '''print(images.is_pinned())
        torch.cuda.synchronize()
        images, labels = images.to(device), labels.to(device)
        if measure_timing:
            torch.cuda.synchronize()
            timings["time_data_load"] = time.time() - tdata
        optimizer.zero_grad()
        if measure_timing:
            timings["time_transforms"] = getattr(loader.dataset.transform, "last_time", 0.0)
            timings["time_collate"] = getattr(loader.collate_fn, "last_time", 0.0)'''

        timings = {}

        # -------------------------
        # Forward
        # -------------------------
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"Forward pass")
        if measure_timing:
            torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if measure_timing:
            torch.cuda.synchronize()
            timings["time_forward"] = time.time() - t0
        torch.cuda.nvtx.range_pop()

        # -------------------------
        # Backward
        # -------------------------
        torch.cuda.nvtx.range_push(f"Backward pass")
        if measure_timing:
            torch.cuda.synchronize()
            t1 = time.time()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm_before = compute_gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        grad_norm_after = compute_gradient_norm(model)
        if measure_timing:
            torch.cuda.synchronize()
            timings["time_backward"] = time.time() - t1
        torch.cuda.nvtx.range_pop()
        # -------------------------
        # Optimizer step
        # -------------------------
        torch.cuda.nvtx.range_push(f"Optimizer step")
        if measure_timing:
            torch.cuda.synchronize()
            t2 = time.time()
        if not torch.isnan(loss):
            scaler.step(optimizer)
            scaler.update()
        if measure_timing:
            torch.cuda.synchronize()
            timings["time_optimizer"] = time.time() - t2
        torch.cuda.nvtx.range_pop()

        # -------------------------
        # Scheduler step
        # -------------------------
        if measure_timing:
            torch.cuda.synchronize()
            t3 = time.time()
        scheduler.step()
        if measure_timing:
            torch.cuda.synchronize()
            timings["time_scheduler"] = time.time() - t3

        # -------------------------
        # Metrics
        # -------------------------
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        accuracy = 100. * correct / total

        # -------------------------
        # Logging
        # -------------------------
        if rank == 0 and i % 1 == 0:
            torch.cuda.synchronize()
            elapsed_time_per_iter = (time.time() - t)/1 if i > 0 else (time.time() - t)
            flops = 3 * microbatch_size * vit_mac_flops(image_size, image_size)

            print("ViT-L/16 FLOPs:", human_readable(flops), "image_size: ", image_size)
            flops_per_sec_per_gpu = flops / elapsed_time_per_iter
            print(f"ViT-L/16 FLOPs/sec/gpu: {human_readable(flops_per_sec_per_gpu)}, iteration time: {elapsed_time_per_iter} seconds")
            grad_norm = compute_gradient_norm(model)
            param_norm = compute_parameter_norm(model)
            opt_state_norm = compute_optimizer_state_norm(optimizer)
            noise_scale = compute_gradient_noise_scale(loss, grad_norm)

            log_dict = {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "flops_per_gpu": flops_per_sec_per_gpu,
                "train_loss": loss,
                "train_acc": accuracy,
                "grad_norm_before_clip": grad_norm_before,
                "grad_norm_after_clip": grad_norm_after,
                "param_norm": param_norm,
                "opt_state_norm": opt_state_norm,
                "grad_noise_scale": noise_scale,
            }
            if measure_timing:
                log_dict.update(timings)

            print(f"[Epoch {epoch}] Batch {i}/{len(loader)} - Loss: {loss.item():.4f}")
            wandb.log(log_dict)
            torch.cuda.synchronize()
            t = time.time()
        
        if measure_timing:
            torch.cuda.synchronize()
            tdata = time.time()

        # -------------------------
        # Profiler step
        # -------------------------
        if i <= prof_iters:
            prof.step()
            save_snapshot("memory_prof")
        if i == prof_iters:  # stop profiling after 5 iterations
            prof.stop()
            torch.cuda.memory._record_memory_history(enabled=None)
            if rank == 0:
                # Print a summary table of ops
                print("\n==== PyTorch Profiler Summary (first 5 iterations) ====")
                print(prof.key_averages().table(
                    sort_by="cuda_time_total",  # sort by CUDA time
                    row_limit=20                # top 20 ops
                ))
                print("=======================================================\n")

    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, accuracy

class CPTIterationScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters, power=2.0, min_lr=0.0, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.power = power
        self.min_lr = min_lr
        super(CPTIterationScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_iter = self.last_epoch + 1
        if current_iter < self.warmup_iters:
            return [
                max(self.min_lr, base_lr * (current_iter + 1) / self.warmup_iters)
                for base_lr in self.base_lrs
            ]
        else:
            progress = (current_iter - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            progress = min(progress, 1.0)
            decay = (torch.cos(torch.tensor(progress * torch.pi / 2)) ** self.power).item()
            return [
                max(self.min_lr, base_lr * decay)
                for base_lr in self.base_lrs
            ]

def save_checkpoint(path, model, optimizer, scaler, scheduler, epoch):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if True: #rank == 0:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            full_state_dict = model.state_dict()
        checkpoint = {
            "model": full_state_dict,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }
        torch.save(checkpoint, path)

    if dist.is_initialized():
        dist.barrier()

def save_checkpoint_rank0(path, model, optimizer, scaler, scheduler, epoch):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
            "optim": FSDP.optim_state_dict(model, optimizer),
            "scaler": scaler.state_dict() if scaler else None,
            "scheduler": scheduler.state_dict(),
            "epoch": epoch
        }

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=dist_cp.FileSystemWriter(path),
        )
def load_checkpoint(path, model, optimizer, scaler, scheduler, device):
    rank = dist.get_rank() if dist.is_initialized() else 0
    checkpoint = torch.load(path, map_location=device)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        state_dict = checkpoint["model"]

        # Handle positional embedding mismatch
        if "pos_embed" in state_dict and "pos_embed" in model.state_dict():
            old_posemb = state_dict["pos_embed"]
            new_posemb = model.state_dict()["pos_embed"]
            if old_posemb.shape != new_posemb.shape:
                if rank == 0:
                    print(f"⚠️ Resizing pos_embed from {old_posemb.shape} to {new_posemb.shape}")
                state_dict["pos_embed"] = resize_pos_embed(old_posemb, new_posemb)

        # Load into model
        model.load_state_dict(state_dict, strict=False)

    # Load optimizer, scheduler, scaler
    if scaler and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    #scheduler.load_state_dict(checkpoint["scheduler"])
    #optimizer.load_state_dict(checkpoint["optimizer"])
    #start_epoch = checkpoint["epoch"] + 1
    start_epoch=0

    if rank == 0:
        print(f"✅ Checkpoint loaded from {path}, resuming at epoch {start_epoch}.")

    return start_epoch

def load_checkpoint_old(path, model, optimizer, scaler, scheduler, device): 
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        state_dict = {
            "model": model.state_dict(),
            # cannot load the optimizer state_dict together with the model state_dict
        }
        state_dict = torch.load(path)

        model.load_state_dict(state_dict["model"])

        if scaler and state_dict["scaler"]:
            scaler.load_state_dict(state_dict["scaler"])
        scheduler.load_state_dict(state_dict["scheduler"])
        start_epoch = state_dict["epoch"] + 1

        '''optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optim",
            storage_reader=dist_cp.FileSystemReader(path),
        )'''

        #osd = FSDP.optim_state_dict_to_load(
        #    model, optimizer, state_dict["optimizer"]
        #)
        osd = state_dict["optimizer"]

        optimizer.load_state_dict(osd)
        start_epoch = state_dict["epoch"] + 1
    return start_epoch
'''
    checkpoint = {}
    cp.load(
        state_dict=checkpoint,
        checkpoint_id=CHECKPOINT_DIR,
    )
    set_state_dict(
            model,
            optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"]
        )
    if scaler and checkpoint["scaler"]:
        scaler.load_state_dict(checkpoint["scaler"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    return start_epoch
'''

def main():
    args = get_args()
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    if rank == 0:
        print(f"Distributed setup with world_size: {world_size}")

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        if args.use_wandb:
            wandb.init(project=args.wandb_project, config=vars(args))
        #torch.cuda.memory._record_memory_history(
        #    True, trace_alloc_max_entries=10000000, trace_alloc_record_context=True
        #)

    image_size =  4096 #768 * 2 * 2 # Adjust as needed
    train_loader, val_loader, train_sampler = create_dataloaders_ddp(
        args.data_dir, image_size, args.batch_size, args.num_workers, rank, world_size)

    model = timm.create_model(
        args.model_name,
        pretrained=True,
        num_classes=len(train_loader.dataset.classes),
        img_size=(image_size, image_size)
    )
    model.set_grad_checkpointing()  # 🔧 Enable checkpointing in timm
    model.to(device)

    # 🔧 Setup FSDP wrapping and mixed precision
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    ) if args.mixed_precision else None

    with enable_wrap(wrapper_cls=FSDP, mixed_precision=mp_policy, device_id=device):
        model = wrap(model)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_iters = len(train_loader) * args.epochs
    scheduler = CPTIterationScheduler(
        optimizer,
        warmup_iters=int(0.1 * total_iters),
        total_iters=total_iters,
        power=args.cpt_power,
        min_lr=args.min_lr
        )

    scaler = ShardedGradScaler() if args.mixed_precision else GradScaler()

    start_epoch = 1
    if args.resume:
        if rank == 0:
            print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler, scheduler, device)
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters()) * world_size
        print(f"Number of Parameters: {num_params} \n")
    epoch = 0
    train_loss,train_acc = 0,0
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    if rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
    for epoch in range(start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_loss,train_acc = 0,0
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device, epoch, rank, scheduler, args.batch_size,
            use_amp=args.mixed_precision, max_grad_norm=5.0, measure_timing = False, image_size=image_size  # ← Clip to 1.0
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if rank == 0:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            if args.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc
                })
        checkpoint_path = os.path.join(args.output_dir, f"{args.model_name}_epoch{epoch}.pt")
        save_checkpoint(checkpoint_path, model, optimizer, scaler, scheduler, epoch)
        if rank == 0:  # Only rank 0 handles cleanup
            cleanup_old_checkpoints(args.output_dir, keep_last=6)


    cleanup_ddp()

if __name__ == '__main__':
    main()
