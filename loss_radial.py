"""
loss_radial.py — Wariant ScalarCorrectionLoss z radialnym cutoffem

Różnica względem loss.py:
  Składowe L1 i MSE są liczone tylko dla pikseli w promieniu r_train [px]
  od centrum metalu. Dalej artefakty są zanikające, a gradient z nich
  wnosi szum do uczenia.

  Cutoff jest przeliczany per próbka w batchu, bo r_max różni się
  w zależności od pozycji centrum metalu w obrazie.

Użycie:
  from loss_radial import ScalarCorrectionLossRadial
  loss_fn = ScalarCorrectionLossRadial(r_train=180.0, ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import HU_MIN, HU_MAX, HU_BIO_LO, HU_BIO_HI

NORM_BIO_LO = (HU_BIO_LO - HU_MIN) / (HU_MAX - HU_MIN)
NORM_BIO_HI = (HU_BIO_HI - HU_MIN) / (HU_MAX - HU_MIN)


class ScalarCorrectionLossRadial(nn.Module):
    """
    Identyczna struktura co ScalarCorrectionLoss, ale L1 i MSE
    są ważone radialną maską binarną:

        r_weight[b, r] = 1  jeśli r_idx < r_train / r_max[b] * N_r
                       = 0  w przeciwnym razie

    Argumenty:
        r_train  -- promień cutoffu w pikselach Kartezjańskich [px].
                    Domyślnie 180 — powyżej tego promienia artefakty
                    metaliczne są zanikające.
                    Ustaw None lub 0 aby wyłączyć (zachowanie jak loss.py).
    """

    def __init__(
        self,
        l1_w:    float = 0.60,
        mse_w:   float = 0.15,
        zero_w:  float = 0.20,
        range_w: float = 0.05,
        r_train: float = 180.0,
    ):
        super().__init__()
        self.l1_w    = l1_w
        self.mse_w   = mse_w
        self.zero_w  = zero_w
        self.range_w = range_w
        self.r_train = r_train if (r_train and r_train > 0) else None

    def _radial_weight(self, B: int, N_r: int, r_max: torch.Tensor) -> torch.Tensor:
        """
        Zwraca wagę [B, 1, N_r] — 1 dla r < cutoff, 0 poza cutoffem.
        Jeśli r_train jest None, zwraca same jedynki (brak cutoffu).
        """
        if self.r_train is None:
            return torch.ones(B, 1, N_r, device=r_max.device)

        # r_cut_idx[b] = min(r_train / r_max[b] * N_r, N_r)
        r_cut_idx = (self.r_train / r_max.float() * N_r).clamp(max=N_r)  # [B]
        r_idx = torch.arange(N_r, device=r_max.device).float()            # [N_r]
        # r_weight[b, r] = r_idx[r] < r_cut_idx[b]
        w = (r_idx.unsqueeze(0) < r_cut_idx.unsqueeze(1)).float()         # [B, N_r]
        return w.unsqueeze(1)                                              # [B, 1, N_r]

    def forward(
        self,
        pred_c:     torch.Tensor,  # [B, N_a, N_r]
        c_target:   torch.Tensor,  # [B, N_a, N_r]
        angle_mask: torch.Tensor,  # [B, N_a]        1 = kąt ze smugą artefaktu
        art_polar:  torch.Tensor,  # [B, N_a, N_r]
        r_max:      torch.Tensor,  # [B]              promień transformacji per próbka
        polar_mask: torch.Tensor,  # [B, N_a, N_r]   1 = piksel metalu w biegunowej
    ):
        B, N_a, N_r = pred_c.shape
        rw = self._radial_weight(B, N_r, r_max)          # [B, 1, N_r]

        diff = pred_c - c_target                          # [B, N_a, N_r]

        # 1. Rekonstrukcja — tylko piksele w cutoffie
        w_sum = rw.sum() * N_a + 1e-8
        l1  = (torch.abs(diff) * rw).sum() / w_sum
        mse = ((diff ** 2)     * rw).sum() / w_sum

        # 2. Regularyzacja zera — kąty bez smugi, bez cutoffu radialnego
        non_metal_ang = (1.0 - angle_mask).unsqueeze(-1)  # [B, N_a, 1]
        zero_reg = (pred_c.abs() * non_metal_ang).mean()

        # 3. Regularyzacja zakresu HU — kąty ze smugą, POZA pikselami metalu.
        # NAPRAWKA: dodano (1 - polar_mask) — wyklucza piksele metalu,
        # które generowały sprzeczny gradient (HU metalu >> NORM_BIO_HI).
        angle_mask_3d = angle_mask.unsqueeze(-1)           # [B, N_a, 1]
        non_metal_3d  = 1.0 - polar_mask                  # [B, N_a, N_r]
        corrected  = art_polar - pred_c
        below      = F.relu(NORM_BIO_LO - corrected)
        above      = F.relu(corrected - NORM_BIO_HI)
        range_mask = angle_mask_3d * non_metal_3d         # [B, N_a, N_r]
        range_reg  = ((below + above) * range_mask).mean()

        total = (self.l1_w    * l1
               + self.mse_w   * mse
               + self.zero_w  * zero_reg
               + self.range_w * range_reg)

        return total, {
            'l1':    l1.item(),
            'mse':   mse.item(),
            'zero':  zero_reg.item(),
            'range': range_reg.item(),
        }
