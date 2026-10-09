"""
sanity_check.py — Wizualna weryfikacja wyników inferencji MAR

Tworzy pliki PNG z losowymi klatkami: oryginał (z artefaktem) | po korekcji
Uruchomienie:
    python3 sanity_check.py \
        --corrected-dir /net/storage/.../CircNet/mar_run_XXXXX/corrected \
        --baseline-dir  /net/storage/.../data/RPI/body/body4/Baseline \
        --output-dir    /net/storage/.../CircNet/mar_run_XXXXX/sanitycheck \
        --n-samples     16 \
        --cols          4
"""

import argparse
import re
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# CT okna wyświetlania
# -------------------------------------------------------------------------
WINDOWS = {
    'soft_tissue': (40,   400),   # center, width
    'lung':        (-600, 1500),
    'bone':        (400,  1800),
}


def apply_window(img_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Mapuje HU → [0, 1] przy zadanym oknie CT."""
    lo = center - width / 2
    hi = center + width / 2
    return np.clip((img_hu - lo) / (hi - lo), 0.0, 1.0)


def load_raw(path, shape=(512, 512), dtype=np.float32) -> np.ndarray:
    return np.fromfile(str(path), dtype=dtype).reshape(shape)


# -------------------------------------------------------------------------
# Argumenty
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--corrected-dir', type=str, required=True,
                   help='Folder z obrazami po korekcji (*.raw)')
    p.add_argument('--baseline-dir',  type=str, required=True,
                   help='Folder z oryginalnymi obrazami z artefaktem (*.raw)')
    p.add_argument('--output-dir',    type=str, required=True,
                   help='Folder wyjściowy na PNG')
    p.add_argument('--n-samples',     type=int, default=16,
                   help='Liczba losowych klatek')
    p.add_argument('--cols',          type=int, default=4,
                   help='Liczba kolumn w siatce (każda kolumna = 1 para)')
    p.add_argument('--window',        type=str, default='soft_tissue',
                   choices=list(WINDOWS.keys()),
                   help='Okno CT do wyświetlania')
    p.add_argument('--seed',          type=int, default=42)
    return p.parse_args()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    args = parse_args()
    random.seed(args.seed)

    corrected_dir = Path(args.corrected_dir)
    baseline_dir  = Path(args.baseline_dir)
    output_dir    = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Zbierz pary (baseline, corrected) po numerze idx
    idx_pat = re.compile(r'img(\d+)', re.IGNORECASE)

    corr_map = {}
    for f in corrected_dir.glob('*.raw'):
        m = idx_pat.search(f.name)
        if m:
            corr_map[int(m.group(1))] = f

    base_map = {}
    for f in baseline_dir.glob('*.raw'):
        m = idx_pat.search(f.name)
        if m:
            base_map[int(m.group(1))] = f

    common = sorted(set(corr_map) & set(base_map))
    if not common:
        print('Brak pasujących par! Sprawdź ścieżki.')
        return

    n = min(args.n_samples, len(common))
    chosen = sorted(random.sample(common, n))
    print(f'Znaleziono {len(common)} par, wizualizuję {n} losowych.')

    center, width = WINDOWS[args.window]
    cols  = args.cols
    rows  = n  # każda para to jeden wiersz: [oryginał | korekcja | diff]

    fig, axes = plt.subplots(
        nrows=rows, ncols=3,
        figsize=(cols * 3, rows * 3.2),
        gridspec_kw={'hspace': 0.05, 'wspace': 0.03},
    )
    if rows == 1:
        axes = [axes]

    for row_idx, img_idx in enumerate(chosen):
        base_hu = load_raw(base_map[img_idx])
        corr_hu = load_raw(corr_map[img_idx])
        diff    = corr_hu - base_hu   # wartości ujemne = odjęliśmy artefakt

        base_w  = apply_window(base_hu, center, width)
        corr_w  = apply_window(corr_hu, center, width)

        ax_b, ax_c, ax_d = axes[row_idx]

        ax_b.imshow(base_w, cmap='gray', vmin=0, vmax=1)
        ax_b.set_title(f'#{img_idx} oryginał', fontsize=7)
        ax_b.axis('off')

        ax_c.imshow(corr_w, cmap='gray', vmin=0, vmax=1)
        ax_c.set_title(f'#{img_idx} po korekcji', fontsize=7)
        ax_c.axis('off')

        # diff: symetryczna skala, kolormap RdBu
        d_abs = np.abs(diff).max()
        d_abs = d_abs if d_abs > 0 else 1.0
        im_d  = ax_d.imshow(diff, cmap='RdBu', vmin=-d_abs, vmax=d_abs)
        ax_d.set_title(f'#{img_idx} różnica', fontsize=7)
        ax_d.axis('off')

    fig.suptitle(
        f'Sanity check MAR — okno: {args.window} '
        f'(C={center} W={width} HU)',
        fontsize=9, y=1.002,
    )

    out_png = output_dir / f'sanitycheck_{args.window}.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Zapisano: {out_png}')

    # Osobne PNG po jednej parze (łatwiej przeglądać)
    singles_dir = output_dir / 'singles'
    singles_dir.mkdir(exist_ok=True)
    for img_idx in chosen:
        base_hu = load_raw(base_map[img_idx])
        corr_hu = load_raw(corr_map[img_idx])
        diff    = corr_hu - base_hu

        base_w  = apply_window(base_hu, center, width)
        corr_w  = apply_window(corr_hu, center, width)

        fig2, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12, 4.2))
        a1.imshow(base_w, cmap='gray', vmin=0, vmax=1)
        a1.set_title('Oryginał (z artefaktem)', fontsize=10)
        a1.axis('off')

        a2.imshow(corr_w, cmap='gray', vmin=0, vmax=1)
        a2.set_title('Po korekcji', fontsize=10)
        a2.axis('off')

        d_abs = max(np.abs(diff).max(), 1.0)
        im3   = a3.imshow(diff, cmap='RdBu', vmin=-d_abs, vmax=d_abs)
        a3.set_title(f'Różnica (max |ΔHU|={np.abs(diff).max():.0f})', fontsize=10)
        a3.axis('off')
        fig2.colorbar(im3, ax=a3, fraction=0.046, pad=0.04)

        fig2.suptitle(
            f'Obraz #{img_idx} — okno {args.window} (C={center} W={width} HU)',
            fontsize=10,
        )
        fig2.tight_layout()
        fig2.savefig(singles_dir / f'img{img_idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)

    print(f'Pojedyncze PNG: {singles_dir}')
    print('Gotowe.')


if __name__ == '__main__':
    main()
