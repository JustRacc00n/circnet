"""
train.py — Trening ScalarCorrectionNet z obsługą DDP (multi-GPU)

Uruchomienie na jednym GPU:
    python3 train.py \
        --data-root /app/data/RPI \
        --train-folders body1 body2 body3 \
        --val-folders body4 \
        --output-dir /app/test/mar_run

Uruchomienie z torchrun (wiele GPU):
    torchrun --nproc_per_node=4 train.py \
        --data-root /app/data/RPI \
        --train-folders body1 body2 body3 \
        --val-folders body4 \
        --output-dir /app/test/mar_run

Wznowienie treningu:
    python3 train.py ... --resume /app/test/mar_run/checkpoints/ckpt_epoch_010.pth
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dataset import PolarScalarDataset, build_paired_dataset, DEFAULT_POLAR_SIZE
from loss import ScalarCorrectionLoss
from model import ScalarCorrectionNet


# -------------------------------------------------------------------------
# Argumenty
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='MAR Scalar Correction Training')

    # Ścieżki — zgodne z konwencją run.sh kolegi
    p.add_argument('--data-root',     type=str, required=True,
                   help='Katalog główny z danymi, np. /app/data/RPI')
    p.add_argument('--train-folders', type=str, nargs='+', required=True,
                   help='Nazwy folderów treningowych, np. body1 body2 body3')
    p.add_argument('--val-folders',   type=str, nargs='+', required=True,
                   help='Nazwy folderów walidacyjnych, np. body4')
    p.add_argument('--output-dir',    type=str, required=True,
                   help='Katalog wyjściowy na modele, logi, checkpointy')
    p.add_argument('--run-name',      type=str, default='mar_run',
                   help='Nazwa eksperymentu (podkatalog w output-dir)')
    p.add_argument('--resume',        type=str, default=None,
                   help='Ścieżka do checkpointu do wznowienia treningu')

    # Dane
    p.add_argument('--polar-size',    type=str, default='512x1024',
                   help='Rozmiar siatki polarnej WxH, np. 512x1024 '
                        '(W=N_radii, H=N_angles) — zgodnie z cv2.warpPolar')

    # Trening
    p.add_argument('--epochs',        type=int,   default=50)
    p.add_argument('--batch-size',    type=int,   default=2,
                   help='Batch size PER GPU')
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--num-workers',   type=int,   default=0)

    # Loss
    p.add_argument('--l1-w',    type=float, default=0.65)
    p.add_argument('--mse-w',   type=float, default=0.20)
    p.add_argument('--zero-w',  type=float, default=0.10)
    p.add_argument('--range-w', type=float, default=0.05)

    # Model
    p.add_argument('--radius-features', type=int, default=32)

    return p.parse_args()


def parse_polar_size(s: str) -> tuple:
    """Parsuje '512x1024' → (512, 1024)."""
    w, h = s.lower().split('x')
    return (int(w), int(h))


# -------------------------------------------------------------------------
# DDP helpers
# -------------------------------------------------------------------------
def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main() -> bool:
    """Zwraca True tylko dla procesu rank=0 (lub gdy DDP nie jest używane)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


# -------------------------------------------------------------------------
# Jeden krok treningu / walidacji
# -------------------------------------------------------------------------
def run_epoch(model, loader, loss_fn, optimizer, device, training: bool):
    model.train() if training else model.eval()
    totals = {'total': 0., 'l1': 0., 'mse': 0., 'zero': 0., 'range': 0.}

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, c_tgt, amask, art_p, _, _ in loader:
            x        = x.to(device, non_blocking=True)
            c_tgt    = c_tgt.to(device, non_blocking=True)
            amask    = amask.to(device, non_blocking=True)
            art_p    = art_p.to(device, non_blocking=True)
            pol_mask = x[:, 1, :, :]  # [B, N_a, N_r] — kanał maski z x_pol[1]

            if training:
                optimizer.zero_grad()

            pred_c = model(x)
            loss, components = loss_fn(pred_c, c_tgt, amask, art_p, pol_mask)

            if training:
                loss.backward()
                optimizer.step()

            totals['total'] += loss.item()
            for k, v in components.items():
                totals[k] += v

    n = len(loader)
    return {k: v / n for k, v in totals.items()}


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args = parse_args()

    # DDP setup
    use_ddp = 'LOCAL_RANK' in os.environ
    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    # Katalogi wyjściowe (tworzy tylko rank 0)
    output_dir     = Path(args.output_dir)
    checkpoint_dir = output_dir / 'checkpoints'  # tymczasowe, nadpisane po modelu
    log_dir        = output_dir / 'logs'

    # -------------------------
    # Dane
    # -------------------------
    data_root  = Path(args.data_root)
    n_polar    = parse_polar_size(args.polar_size)
    train_dirs = [data_root / f for f in args.train_folders]
    val_dirs   = [data_root / f for f in args.val_folders]

    if is_main():
        print('Budowanie listy próbek...')
        print(f'  Train folders: {args.train_folders}')
        print(f'  Val folders:   {args.val_folders}')
        print(f'  Polar size:    {n_polar} (N_radii x N_angles)')
    train_samples = build_paired_dataset(train_dirs)
    val_samples   = build_paired_dataset(val_dirs)
    if is_main():
        print(f'  Train: {len(train_samples)} | Val: {len(val_samples)}')

    train_ds = PolarScalarDataset(train_samples, n_polar=n_polar)
    val_ds   = PolarScalarDataset(val_samples,   n_polar=n_polar)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if use_ddp else None
    val_sampler   = DistributedSampler(val_ds,  shuffle=False) if use_ddp else None

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # -------------------------
    # Model, optymalizator, loss
    # -------------------------
    model = ScalarCorrectionNet(
        n_radii=n_polar[0],
        radius_features=args.radius_features,
    ).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn   = ScalarCorrectionLoss(
        l1_w=args.l1_w, mse_w=args.mse_w,
        zero_w=args.zero_w, range_w=args.range_w,
    )

    # Katalog wyjściowy uwzględnia run-name
    output_dir = output_dir / args.run_name
    checkpoint_dir = output_dir / 'checkpoints'
    log_dir        = output_dir / 'logs'
    if is_main():
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()

    # -------------------------
    # Wznowienie z checkpointu
    # -------------------------
    start_epoch = 1
    best_val    = float('inf')

    if args.resume:
        if is_main():
            print(f'Wznawiam z {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        raw_model = model.module if use_ddp else model
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))

    # -------------------------
    # CSV log (tylko rank 0)
    # -------------------------
    csv_path = log_dir / 'training_metrics.csv'
    if is_main() and start_epoch == 1:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'Epoch',
                'Train_Total', 'Train_L1', 'Train_MSE', 'Train_Zero', 'Train_Range',
                'Val_Total',   'Val_L1',   'Val_MSE',   'Val_Zero',   'Val_Range',
                'LR',
            ])

    # -------------------------
    # Liczba parametrów
    # -------------------------
    if is_main():
        raw_model = model.module if use_ddp else model
        n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        world = dist.get_world_size() if use_ddp else 1
        print(f'\nDevice: {device} | GPUs: {world}')
        print(f'Parametry modelu: {n_params:,}')
        print(f'Batch size per GPU: {args.batch_size} | Global: {args.batch_size * world}')
        print(f'Epoki: {start_epoch} → {args.epochs}\n')

    # -------------------------
    # Pętla treningowa
    # -------------------------
    for epoch in range(start_epoch, args.epochs + 1):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        tr = run_epoch(model, train_dl, loss_fn, optimizer, device, training=True)
        vl = run_epoch(model, val_dl,   loss_fn, None,      device, training=False)

        # Uśrednij metryki walidacyjne po wszystkich rankach
        if use_ddp:
            for metrics in (tr, vl):
                for k in metrics:
                    t = torch.tensor(metrics[k], device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.AVG)
                    metrics[k] = t.item()

        scheduler.step(vl['total'])
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        if is_main():
            flag = ''
            if vl['total'] < best_val:
                best_val = vl['total']
                raw_model = model.module if use_ddp else model
                torch.save(raw_model.state_dict(), output_dir / 'best_model.pth')
                # Zapisz argumenty modelu obok best_model.pth (do późniejszej inferencji)
                model_args = {
                    'radius_features': args.radius_features,
                    'polar_size':      args.polar_size,
                    'n_radii':         n_polar[0],
                    'n_angles':        n_polar[1],
                }
                with open(output_dir / 'best_model_args.json', 'w') as _f:
                    json.dump(model_args, _f, indent=2)
                flag = ' ← BEST'

            if epoch % 5 == 0:
                raw_model = model.module if use_ddp else model
                ckpt_path = checkpoint_dir / f'ckpt_epoch_{epoch:03d}.pth'
                torch.save({
                    'epoch':                epoch,
                    'model_state_dict':     raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss':             vl['total'],
                    'best_val':             best_val,
                }, ckpt_path)
                flag += f' [ckpt]'

            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([
                    epoch,
                    tr['total'], tr['l1'], tr['mse'], tr['zero'], tr['range'],
                    vl['total'], vl['l1'], vl['mse'], vl['zero'], vl['range'],
                    lr,
                ])

            print(
                f'Ep {epoch:3d}/{args.epochs} '
                f'| Tr {tr["total"]:.4f} '
                f'(L1={tr["l1"]:.4f} MSE={tr["mse"]:.4f} '
                f'Zero={tr["zero"]:.4f} Rng={tr["range"]:.4f}) '
                f'| Vl {vl["total"]:.4f} '
                f'(L1={vl["l1"]:.4f} MSE={vl["mse"]:.4f} '
                f'Zero={vl["zero"]:.4f} Rng={vl["range"]:.4f}) '
                f'| LR {lr:.2e} | {elapsed:.0f}s{flag}'
            )

    if is_main():
        print(f'\nGotowe. Best val loss: {best_val:.4f}')
        print(f'Model zapisany w: {output_dir / "best_model.pth"}')

    if use_ddp:
        cleanup_ddp()


if __name__ == '__main__':
    main()
