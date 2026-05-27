# neurosim_utils.py
#
# Refactor pass aligned with MLP_NeuroSim V3.0 (Shimeng Yu, Georgia Tech / ASU).
#
# v3 features (this file):
#   1. NonlinearWeight LTP/LTD with closed-form B (V3.0 boundary)
#   2. Pmax_LTP / Pmax_LTD clamp
#   3. σ_c2c per-pulse Gaussian noise (cycle-to-cycle)
#   4. σ_d2d per-cell A/B sampling   (NEW — device-to-device variation)
#   5. Differential-pair (G⁺ − G⁻) mode  (NEW)
#   6. Pulse-energy / read-energy / area accounting  (NEW)
#   7. Conductance discretisation (numWeightBit)
#
# All new features default to OFF, so existing call sites are bit-for-bit
# compatible with the v2 release.

from __future__ import annotations

import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# NL <-> A heuristic (kept for analysis_runners)
# ---------------------------------------------------------------------------
def estimate_nl_from_norm_a(norm_a_val: float) -> float:
    if norm_a_val is None:
        return 10.0
    try:
        x = float(norm_a_val)
    except Exception:
        return 10.0
    x = max(0.0, x)
    bins = [(0.001, 10.0), (0.002, 9.0), (0.005, 8.0), (0.01, 7.0), (0.02, 6.0),
            (0.05, 5.0), (0.1, 4.0), (0.2, 3.0), (0.5, 2.0)]
    for thr, val in bins:
        if x <= thr:
            return val
    return 1.0


# ---------------------------------------------------------------------------
# Default per-pulse energetics (overrideable from config)
# ---------------------------------------------------------------------------
DEFAULT_ENERGY = {
    "E_pulse_LTP_J": 1.0e-12,   # 1 pJ per write pulse (typical RRAM-class)
    "E_pulse_LTD_J": 1.0e-12,
    "E_read_J":      1.0e-15,   # 1 fJ per read (one cell, one MAC contribution)
    "cell_area_m2":  1.0e-12,   # 1 µm² per cell (single)
}


# ---------------------------------------------------------------------------
# Fitter
# ---------------------------------------------------------------------------
class NeuroSimFitter:
    """V3.0-style NonlinearWeight fitter.

    Parameters
    ----------
    ltp_df, ltd_df : pandas.DataFrame
        Must have columns "PulseNum" and "Conductance". LTD's PulseNum is
        automatically remapped so its effective range starts at 0.
    target_range : (float, float)
        Software weight range, e.g. (-1, 1).
    sigma_c2c : float
        Cycle-to-cycle noise (std on normalised conductance per pulse). If
        None, estimated from data residuals (diff-based).
    sigma_d2d : float
        Device-to-device variation, *relative*. A_LTP[i,j] is sampled as
        A_LTP_mean * exp(sigma_d2d * N(0,1)). 0 → all cells identical.
    num_conductance_states : int or None
        If set, G is discretised into that many levels (V3.0 numWeightBit).
    """

    # ---- construction ----
    def __init__(
        self,
        ltp_df,
        ltd_df,
        target_range=(-1.0, 1.0),
        ltp_fit_ratio: float = 1.0,
        ltd_fit_ratio: float = 1.0,
        use_p_start_offset: bool = True,        # kept for API
        p_start_force_zero: bool = True,        # kept for API
        sigma_c2c: float | None = None,
        sigma_d2d: float = 0.0,
        num_conductance_states: int | None = None,
        verbose: bool = True,
    ):
        self.verbose = bool(verbose)
        self.target_min, self.target_max = float(target_range[0]), float(target_range[1])
        self.ltp_fit_ratio = float(ltp_fit_ratio)
        self.ltd_fit_ratio = float(ltd_fit_ratio)
        self.sigma_d2d = float(max(sigma_d2d, 0.0))

        # API compat (no-ops in this refactor)
        self.use_p_start_offset = True
        self.p_start_force_zero = True

        ltp_p_raw = ltp_df["PulseNum"].values.astype(np.float64)
        ltd_p_raw = ltd_df["PulseNum"].values.astype(np.float64)
        ltp_g_real = ltp_df["Conductance"].values.astype(np.float64)
        ltd_g_real = ltd_df["Conductance"].values.astype(np.float64)

        self.ltp_p_np = ltp_p_raw - ltp_p_raw.min()
        self.ltd_p_np = ltd_p_raw - ltd_p_raw.min()
        self.Pmax_LTP = float(self.ltp_p_np.max())
        self.Pmax_LTD = float(self.ltd_p_np.max())

        self.g_min_real = float(min(ltp_g_real.min(), ltd_g_real.min()))
        self.g_max_real = float(max(ltp_g_real.max(), ltd_g_real.max()))

        self.ltp_g_scaled = self._scale(ltp_g_real)
        self.ltd_g_scaled = self._scale(ltd_g_real)
        self.g_min_scaled = float(self.target_min)
        self.g_max_scaled = float(self.target_max)

        self.A_LTP = self._fit_curve_for_A(
            self.ltp_p_np, self.ltp_g_scaled, start_scaled=self.g_min_scaled,
            ratio=self.ltp_fit_ratio, ltp=True, pmax=self.Pmax_LTP)
        self.A_LTD = self._fit_curve_for_A(
            self.ltd_p_np, self.ltd_g_scaled, start_scaled=self.g_max_scaled,
            ratio=self.ltd_fit_ratio, ltp=False, pmax=self.Pmax_LTD)

        self.B_LTP = self._B_from_A_scalar(self.A_LTP, self.Pmax_LTP)
        self.B_LTD = self._B_from_A_scalar(self.A_LTD, self.Pmax_LTD)

        if sigma_c2c is None:
            self.sigma_c2c = self._estimate_sigma_c2c()
        else:
            self.sigma_c2c = float(sigma_c2c)

        self.num_conductance_states = num_conductance_states
        if self.num_conductance_states is not None:
            self.num_conductance_states = int(self.num_conductance_states)

        # public aliases (back-compat)
        self.g_min_fit_scaled = self.g_min_scaled
        self.g_max_fit_scaled = self.g_max_scaled
        self.A_LTP_Norm = self.A_LTP
        self.A_LTD_Norm = self.A_LTD
        self.A_LTP_Raw = self.A_LTP
        self.A_LTD_Raw = self.A_LTD

        if self.verbose:
            print("\n[INFO] NeuroSim V3.0 fitter")
            print(f"  G_real  = [{self.g_min_real:.3e}, {self.g_max_real:.3e}]")
            print(f"  W_target= [{self.target_min:+.2f}, {self.target_max:+.2f}]")
            print(f"  Pmax    = LTP {self.Pmax_LTP:.0f},  LTD {self.Pmax_LTD:.0f}")
            print(f"  A_LTP={self.A_LTP:.4f}  B_LTP={self.B_LTP:.4f}")
            print(f"  A_LTD={self.A_LTD:.4f}  B_LTD={self.B_LTD:.4f}")
            print(f"  σ_c2c = {self.sigma_c2c:.4f} (scaled-W units)")
            if self.sigma_d2d > 0:
                print(f"  σ_d2d = {self.sigma_d2d:.4f} (relative, log-normal on A)")
            if self.num_conductance_states:
                print(f"  numConductanceStates = {self.num_conductance_states}")

    # ---------- scaling ----------
    def _scale(self, g):
        denom = (self.g_max_real - self.g_min_real)
        if denom == 0:
            return np.full_like(g, self.target_min, dtype=np.float64)
        return ((g - self.g_min_real) / denom) * (self.target_max - self.target_min) + self.target_min

    def scale(self, g): return self._scale(g)

    def _unscale(self, g):
        denom = (self.target_max - self.target_min)
        if denom == 0:
            return np.full_like(g, self.g_min_real, dtype=np.float64)
        return ((g - self.target_min) / denom) * (self.g_max_real - self.g_min_real) + self.g_min_real

    def unscale(self, g): return self._unscale(g)

    # ---------- curve fitting ----------
    @staticmethod
    def _B_from_A_scalar(A, Pmax):
        denom = 1.0 - float(np.exp(-Pmax / max(A, 1e-9)))
        if abs(denom) < 1e-12:
            return 0.0
        return 2.0 / denom

    @staticmethod
    def _B_from_A_tensor(A: torch.Tensor, Pmax: float) -> torch.Tensor:
        denom = 1.0 - torch.exp(-Pmax / A.clamp(min=1e-9))
        return 2.0 / denom.clamp(min=1e-12)

    def _fit_curve_for_A(self, P, G_scaled, start_scaled, ratio, ltp, pmax):
        n = len(P)
        keep = max(2, int(np.ceil(n * float(ratio))))
        P_use = P[:keep]; G_use = G_scaled[:keep]

        if ltp:
            def model(P_, A):
                B = self._B_from_A_scalar(A, pmax)
                return B * (1.0 - np.exp(-P_ / max(A, 1e-9))) + start_scaled
        else:
            def model(P_, A):
                B = self._B_from_A_scalar(A, pmax)
                return -B * (1.0 - np.exp(-P_ / max(A, 1e-9))) + start_scaled

        try:
            popt, _ = curve_fit(model, P_use, G_use, p0=[max(pmax / 2.0, 1.0)],
                                maxfev=20000, bounds=(1e-2, 1e6))
            return float(popt[0])
        except Exception as e:
            if self.verbose:
                print(f"[WARN] curve_fit failed ({'LTP' if ltp else 'LTD'}): {e}")
            try:
                def free_model(P_, A, B):
                    if ltp:
                        return B * (1.0 - np.exp(-P_ / max(A, 1e-9))) + start_scaled
                    return -B * (1.0 - np.exp(-P_ / max(A, 1e-9))) + start_scaled
                popt, _ = curve_fit(free_model, P_use, G_use,
                                    p0=[max(pmax / 2.0, 1.0), 2.0], maxfev=20000)
                return float(popt[0])
            except Exception as e2:
                print(f"[ERR] LTP/LTD fit failed: {e2}")
                return float(max(pmax / 2.0, 1.0))

    def _estimate_sigma_c2c(self):
        def per_dir(P, G_scaled, ltp):
            if len(P) < 2: return 0.0
            P_t = torch.from_numpy(P)
            with torch.no_grad():
                if ltp: G_pred = self.g_of_p_ltp(P_t).numpy()
                else:   G_pred = self.g_of_p_ltd(P_t).numpy()
            r = G_scaled - G_pred
            dr = np.diff(r)
            return float(np.std(dr) / np.sqrt(2.0))
        s_ltp = per_dir(self.ltp_p_np, self.ltp_g_scaled, ltp=True)
        s_ltd = per_dir(self.ltd_p_np, self.ltd_g_scaled, ltp=False)
        return max(0.5 * (s_ltp + s_ltd), 0.0)

    # ---------- scalar forward maps (kept for plotting / external use) ----------
    def _to_t(self, x):
        if not isinstance(x, torch.Tensor):
            return torch.as_tensor(x, dtype=torch.float32)
        return x

    def g_of_p_ltp(self, P):
        P = self._to_t(P).clamp(min=0.0, max=self.Pmax_LTP)
        return self.B_LTP * (1.0 - torch.exp(-P / (self.A_LTP + 1e-9))) + self.g_min_scaled

    def g_of_p_ltd(self, P):
        P = self._to_t(P).clamp(min=0.0, max=self.Pmax_LTD)
        return -self.B_LTD * (1.0 - torch.exp(-P / (self.A_LTD + 1e-9))) + self.g_max_scaled

    def p_of_g_ltp(self, G):
        eps = 1e-9
        arg = 1.0 - (G - self.g_min_scaled) / (self.B_LTP + eps)
        P = -self.A_LTP * torch.log(arg.clamp(min=eps, max=1.0))
        return P.clamp(min=0.0, max=self.Pmax_LTP)

    def p_of_g_ltd(self, G):
        eps = 1e-9
        arg = 1.0 - (self.g_max_scaled - G) / (self.B_LTD + eps)
        P = -self.A_LTD * torch.log(arg.clamp(min=eps, max=1.0))
        return P.clamp(min=0.0, max=self.Pmax_LTD)

    # ---------- per-cell maps (used by σ_d2d path) ----------
    def g_of_p_ltp_cell(self, P, A, B):
        P = P.clamp(min=0.0, max=self.Pmax_LTP)
        return B * (1.0 - torch.exp(-P / (A + 1e-9))) + self.g_min_scaled

    def g_of_p_ltd_cell(self, P, A, B):
        P = P.clamp(min=0.0, max=self.Pmax_LTD)
        return -B * (1.0 - torch.exp(-P / (A + 1e-9))) + self.g_max_scaled

    def p_of_g_ltp_cell(self, G, A, B):
        eps = 1e-9
        arg = 1.0 - (G - self.g_min_scaled) / (B + eps)
        return (-A * torch.log(arg.clamp(min=eps, max=1.0))).clamp(min=0.0, max=self.Pmax_LTP)

    def p_of_g_ltd_cell(self, G, A, B):
        eps = 1e-9
        arg = 1.0 - (self.g_max_scaled - G) / (B + eps)
        return (-A * torch.log(arg.clamp(min=eps, max=1.0))).clamp(min=0.0, max=self.Pmax_LTD)

    # ---------- σ_d2d sampler ----------
    def sample_per_cell_AB(self, shape, generator: torch.Generator | None = None):
        """Sample per-cell A_LTP, A_LTD (log-normal) and matching B."""
        if self.sigma_d2d <= 0.0:
            return None
        if generator is None:
            generator = torch.Generator()
        # log-normal keeps A > 0
        A_LTP = self.A_LTP * torch.exp(self.sigma_d2d *
                torch.randn(shape, generator=generator))
        A_LTD = self.A_LTD * torch.exp(self.sigma_d2d *
                torch.randn(shape, generator=generator))
        A_LTP = A_LTP.clamp(min=1e-2)
        A_LTD = A_LTD.clamp(min=1e-2)
        B_LTP = self._B_from_A_tensor(A_LTP, self.Pmax_LTP)
        B_LTD = self._B_from_A_tensor(A_LTD, self.Pmax_LTD)
        return {"A_LTP": A_LTP, "B_LTP": B_LTP,
                "A_LTD": A_LTD, "B_LTD": B_LTD}

    # ---------- conductance discretisation ----------
    def discretise(self, G):
        if self.num_conductance_states is None:
            return G
        levels = self.num_conductance_states
        span = (self.target_max - self.target_min)
        step = span / max(levels - 1, 1)
        q = torch.round((G - self.target_min) / step) * step + self.target_min
        return q.clamp(self.target_min, self.target_max)

    # ---------- plotting helpers (kept for main.py) ----------
    def _normalize_0_1_real(self, g_real):
        denom = (self.g_max_real - self.g_min_real)
        if denom == 0:
            return np.zeros_like(g_real, dtype=np.float32)
        return ((g_real - self.g_min_real) / denom).astype(np.float32)

    def plot_fit_normalized_0_1(self, ltp_df, ltd_df, title_suffix=""):
        ltp_p = ltp_df["PulseNum"].values.astype(np.float32)
        ltd_p = ltd_df["PulseNum"].values.astype(np.float32)
        ltp_g_real = ltp_df["Conductance"].values.astype(np.float32)
        ltd_g_real = ltd_df["Conductance"].values.astype(np.float32)
        ltp_p_e = ltp_p - ltp_p.min()
        ltd_p_e = ltd_p - ltd_p.min()
        ltp_g_norm = self._normalize_0_1_real(ltp_g_real)
        ltd_g_norm = self._normalize_0_1_real(ltd_g_real)
        with torch.no_grad():
            ltp_fit_s = self.g_of_p_ltp(torch.from_numpy(ltp_p_e)).numpy()
            ltd_fit_s = self.g_of_p_ltd(torch.from_numpy(ltd_p_e)).numpy()
        ltp_fit_real = self._unscale(ltp_fit_s)
        ltd_fit_real = self._unscale(ltd_fit_s)
        ltp_fit_norm = self._normalize_0_1_real(ltp_fit_real)
        ltd_fit_norm = self._normalize_0_1_real(ltd_fit_real)
        plt.figure(figsize=(10, 6))
        plt.scatter(ltp_p_e, ltp_g_norm, s=18, alpha=0.7, label="LTP data")
        plt.scatter(ltd_p_e, ltd_g_norm, s=18, alpha=0.7, label="LTD data")
        ord_ltp = np.argsort(ltp_p_e); ord_ltd = np.argsort(ltd_p_e)
        plt.plot(ltp_p_e[ord_ltp], ltp_fit_norm[ord_ltp], lw=2, label="LTP fit")
        plt.plot(ltd_p_e[ord_ltd], ltd_fit_norm[ord_ltd], lw=2, label="LTD fit")
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Pulse # (rezeroed)"); plt.ylabel("Normalised G (0~1)")
        plt.title(f"NeuroSim V3 fit {title_suffix}")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        try:
            plt.show()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Optimizer (handles single-cell + differential-pair + σ_d2d + energy)
# ---------------------------------------------------------------------------
class NeuroSimOptimizer(optim.Optimizer):
    """V3.0 pulse-domain optimizer with optional advanced modes.

    Modes (all default OFF for backward compatibility):
      pair_mode=True       → differential pair (G⁺ − G⁻) per weight
      sigma_d2d via fitter → per-cell A, B sampled at init
      energy accounting    → always-on counters; report via energy_report()

    Backwards-compat kwargs:
      degraded_fitter, masks  — per-cell device-degradation (degradation_ui.py)
    """

    def __init__(self, params, lr=1e-3, fitter: NeuroSimFitter = None,
                 pulse_scaling_factor: float = 1.0,
                 degraded_fitter: NeuroSimFitter | None = None,
                 masks: dict | None = None,
                 use_c2c_noise: bool = True,
                 use_discretisation: bool = True,
                 pair_mode: bool = False,
                 pair_strategy: str = "mixed",
                 energy_params: dict | None = None,
                 d2d_seed: int = 0):
        """
        pair_strategy : str
            Only matters when pair_mode=True.
            - "mixed"    : V3.0 canonical. Uses BOTH LTP and LTD on either cell;
                           per-element selects the cell with more headroom in the
                           required direction. Cells stay balanced naturally.
            - "ltp_only" : Simpler variant (Burr 2015, Boybat 2018). Always uses
                           LTP, picks G_pos for W↑ and G_neg for W↓. Periodic
                           refresh when both saturate.
        """
        if fitter is None:
            raise ValueError("fitter is required")
        self.fitter = fitter
        self.degraded_fitter = degraded_fitter
        self.masks = masks if masks is not None else {}
        self.use_c2c_noise = bool(use_c2c_noise)
        self.use_discretisation = bool(use_discretisation)
        self.pair_mode = bool(pair_mode)
        if pair_strategy not in ("mixed", "ltp_only"):
            raise ValueError(f"pair_strategy must be 'mixed' or 'ltp_only', got {pair_strategy!r}")
        self.pair_strategy = str(pair_strategy)

        defaults = dict(lr=lr, pulse_scaling_factor=pulse_scaling_factor)
        super().__init__(params, defaults)

        # σ_d2d storage (per-parameter dict of {A_LTP, B_LTP, A_LTD, B_LTD} tensors)
        self.per_cell_AB: dict[int, dict] = {}
        self.per_cell_AB_deg: dict[int, dict] = {}
        # Pair-mode storage
        self.G_pos: dict[int, torch.Tensor] = {}
        self.G_neg: dict[int, torch.Tensor] = {}

        # Energy
        self.energy_params = dict(DEFAULT_ENERGY)
        if energy_params:
            self.energy_params.update(energy_params)
        self.total_pulses_ltp: float = 0.0
        self.total_pulses_ltd: float = 0.0
        self.total_reads: int = 0
        self.total_weights: int = 0

        rng = torch.Generator()
        rng.manual_seed(int(d2d_seed))

        for group in self.param_groups:
            for p in group["params"]:
                pid = id(p)
                self.total_weights += int(p.numel())

                if self.fitter.sigma_d2d > 0.0:
                    self.per_cell_AB[pid] = self.fitter.sample_per_cell_AB(
                        tuple(p.shape), generator=rng)
                if degraded_fitter is not None and degraded_fitter.sigma_d2d > 0.0:
                    self.per_cell_AB_deg[pid] = degraded_fitter.sample_per_cell_AB(
                        tuple(p.shape), generator=rng)

                if self.pair_mode:
                    self._init_pair(p, pid)

    # ---------- internal helpers ----------
    @staticmethod
    def _stochastic_round(dP: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(dP)
        mag = torch.abs(dP)
        noise = torch.rand_like(mag)
        return sign * torch.floor(mag + noise)

    def _get_AB(self, pid, ltp: bool, fitter: NeuroSimFitter, deg: bool):
        store = self.per_cell_AB_deg if deg else self.per_cell_AB
        if pid in store:
            d = store[pid]
            return (d["A_LTP"], d["B_LTP"]) if ltp else (d["A_LTD"], d["B_LTD"])
        return (fitter.A_LTP, fitter.B_LTP) if ltp else (fitter.A_LTD, fitter.B_LTD)

    def _forward_AB(self, P, A, B, ltp, fitter):
        if ltp:
            return fitter.g_of_p_ltp_cell(P, A, B) if isinstance(A, torch.Tensor) \
                   else fitter.g_of_p_ltp(P)
        return fitter.g_of_p_ltd_cell(P, A, B) if isinstance(A, torch.Tensor) \
               else fitter.g_of_p_ltd(P)

    def _inverse_AB(self, G, A, B, ltp, fitter):
        if ltp:
            return fitter.p_of_g_ltp_cell(G, A, B) if isinstance(A, torch.Tensor) \
                   else fitter.p_of_g_ltp(G)
        return fitter.p_of_g_ltd_cell(G, A, B) if isinstance(A, torch.Tensor) \
               else fitter.p_of_g_ltd(G)

    def _pulse_branch(self, sel_mask, pid, fitter, G_cur, G_tgt, G_new,
                      psf, use_ltp, deg=False):
        if not sel_mask.any():
            return
        A, B = self._get_AB(pid, use_ltp, fitter, deg)
        # gather per-cell A,B if they're tensors
        if isinstance(A, torch.Tensor):
            A_sel = A[sel_mask]; B_sel = B[sel_mask]
        else:
            A_sel = A; B_sel = B

        G_cur_sel = G_cur[sel_mask]
        G_tgt_sel = G_tgt[sel_mask]

        P_cur = self._inverse_AB(G_cur_sel, A_sel, B_sel, use_ltp, fitter)
        P_tgt = self._inverse_AB(G_tgt_sel, A_sel, B_sel, use_ltp, fitter)
        Pmax = fitter.Pmax_LTP if use_ltp else fitter.Pmax_LTD

        dP = ((P_tgt - P_cur) * psf).nan_to_num(0.0)
        dP_q = self._stochastic_round(dP)
        P_next = (P_cur + dP_q).clamp(0.0, Pmax)
        G_branch = self._forward_AB(P_next, A_sel, B_sel, use_ltp, fitter)

        if self.use_c2c_noise and fitter.sigma_c2c > 0.0:
            actually_pulsed = (dP_q.abs() > 0.0)
            if actually_pulsed.any():
                noise = torch.randn_like(G_branch) * fitter.sigma_c2c
                G_branch = torch.where(actually_pulsed, G_branch + noise, G_branch)

        G_branch = G_branch.clamp(fitter.target_min, fitter.target_max)
        G_new[sel_mask] = G_branch

        # energy accounting
        n = float(dP_q.abs().sum().item())
        if use_ltp:
            self.total_pulses_ltp += n
        else:
            self.total_pulses_ltd += n

    # ---------- pair-mode init / step ----------
    def _init_pair(self, p, pid):
        """Initialise G_pos, G_neg so that W = G_pos − G_neg."""
        W = p.data.clamp(self.fitter.target_min, self.fitter.target_max).clone()
        # split: positive part of W goes to G_pos, negative part to G_neg
        Gp = W.clamp(min=0.0, max=self.fitter.target_max)
        Gn = (-W).clamp(min=0.0, max=self.fitter.target_max)
        self.G_pos[pid] = Gp
        self.G_neg[pid] = Gn
        # write back so model sees consistent W
        p.data.copy_((Gp - Gn).clamp(self.fitter.target_min, self.fitter.target_max))

    # ---------- shared helper: per-cell normal/degraded split mask -------------
    def _split_normal_degraded(self, pid, G_cur):
        """Return (normal_mask, degraded_mask) tensors matching G_cur shape.

        Honours `masks[pid]` (set by degradation_ui) and only flags cells
        as degraded when a `degraded_fitter` has been supplied. Otherwise
        every cell is normal.
        """
        cell_mask = self.masks.get(pid)
        if cell_mask is None or self.degraded_fitter is None:
            normal = torch.ones_like(G_cur, dtype=torch.bool)
            degraded = torch.zeros_like(G_cur, dtype=torch.bool)
        else:
            cm = cell_mask.to(device=G_cur.device).bool()
            normal = cm
            degraded = ~cm
        return normal, degraded

    # ---------- pair-mode step: LTP-only with refresh (Burr / Boybat) ----------
    def _step_pair_ltp_only(self, p, pid, grad, lr, psf):
        """Differential pair using LTP on both cells + periodic refresh.

        - W↑: LTP on G_pos
        - W↓: LTP on G_neg
        - When both cells saturate, reset keeping (G_pos − G_neg) constant.

        Honours `masks` + `degraded_fitter` so degradation_ui keeps working:
        cells flagged in `masks` use the degraded_fitter for their pulses.
        """
        f_n = self.fitter
        f_d = self.degraded_fitter
        Gp = self.G_pos[pid]
        Gn = self.G_neg[pid]
        delta_W = -lr * grad
        tmax = f_n.target_max

        normal, degraded = self._split_normal_degraded(pid, Gp)

        ltp_pos_mask = grad < 0
        ltp_neg_mask = grad > 0

        Gp_target = (Gp + delta_W.clamp(min=0.0)).clamp(min=0.0, max=tmax)
        Gn_target = (Gn + (-delta_W).clamp(min=0.0)).clamp(min=0.0, max=tmax)

        Gp_new = Gp.clone()
        Gn_new = Gn.clone()

        # normal cells go through nominal fitter
        self._pulse_branch(ltp_pos_mask & normal, pid, f_n,
                           Gp, Gp_target, Gp_new, psf, use_ltp=True)
        self._pulse_branch(ltp_neg_mask & normal, pid, f_n,
                           Gn, Gn_target, Gn_new, psf, use_ltp=True)
        # degraded cells go through degraded fitter (if any)
        if f_d is not None:
            self._pulse_branch(ltp_pos_mask & degraded, pid, f_d,
                               Gp, Gp_target, Gp_new, psf, use_ltp=True, deg=True)
            self._pulse_branch(ltp_neg_mask & degraded, pid, f_d,
                               Gn, Gn_target, Gn_new, psf, use_ltp=True, deg=True)

        if self.use_discretisation:
            Gp_new = f_n.discretise(Gp_new)
            Gn_new = f_n.discretise(Gn_new)

        # Refresh: when *both* cells are near max, reset keeping (Gp − Gn)
        thr = 0.85 * tmax
        sat = (Gp_new > thr) & (Gn_new > thr)
        if sat.any():
            delta = Gp_new[sat] - Gn_new[sat]
            Gp_new[sat] = delta.clamp(min=0.0, max=tmax)
            Gn_new[sat] = (-delta).clamp(min=0.0, max=tmax)

        Gp_new = Gp_new.clamp(min=0.0, max=tmax)
        Gn_new = Gn_new.clamp(min=0.0, max=tmax)
        self.G_pos[pid] = Gp_new
        self.G_neg[pid] = Gn_new
        p.data.copy_((Gp_new - Gn_new).clamp(f_n.target_min, f_n.target_max))

    # ---------- pair-mode step: mixed LTP+LTD (V3.0 canonical) ----------
    def _step_pair_mixed(self, p, pid, grad, lr, psf):
        """V3.0-canonical differential pair using BOTH LTP and LTD.

        Per-element cell selection by headroom:
          grad < 0 (W↑):   LTP on G_pos  if (target_max − G_pos) ≥ G_neg
                           LTD on G_neg  otherwise
          grad > 0 (W↓):   LTD on G_pos  if G_pos ≥ (target_max − G_neg)
                           LTP on G_neg  otherwise

        Naturally keeps cells balanced — no refresh required.
        """
        f_n = self.fitter
        f_d = self.degraded_fitter
        Gp = self.G_pos[pid]
        Gn = self.G_neg[pid]
        delta_W = -lr * grad
        tmax = f_n.target_max

        # Headroom-based cell selection per element
        up_mask = grad < 0      # W needs to increase
        dn_mask = grad > 0      # W needs to decrease

        prefer_ltp_pos = (tmax - Gp) >= Gn          # bigger headroom on G_pos LTP
        prefer_ltd_pos = Gp >= (tmax - Gn)          # bigger headroom on G_pos LTD

        ltp_pos_mask = up_mask & prefer_ltp_pos
        ltd_neg_mask = up_mask & (~prefer_ltp_pos)
        ltd_pos_mask = dn_mask & prefer_ltd_pos
        ltp_neg_mask = dn_mask & (~prefer_ltd_pos)

        # Per-cell targets
        Gp_target_up   = (Gp + delta_W.clamp(min=0.0)).clamp(0.0, tmax)   # LTP G_pos
        Gn_target_up   = (Gn - delta_W.clamp(min=0.0)).clamp(0.0, tmax)   # LTD G_neg
        Gp_target_down = (Gp + delta_W.clamp(max=0.0)).clamp(0.0, tmax)   # LTD G_pos
        Gn_target_down = (Gn - delta_W.clamp(max=0.0)).clamp(0.0, tmax)   # LTP G_neg

        Gp_new = Gp.clone()
        Gn_new = Gn.clone()

        normal, degraded = self._split_normal_degraded(pid, Gp)

        # 4-way branching for NORMAL cells (nominal fitter)
        self._pulse_branch(ltp_pos_mask & normal, pid, f_n,
                           Gp, Gp_target_up,   Gp_new, psf, use_ltp=True)
        self._pulse_branch(ltd_pos_mask & normal, pid, f_n,
                           Gp, Gp_target_down, Gp_new, psf, use_ltp=False)
        self._pulse_branch(ltp_neg_mask & normal, pid, f_n,
                           Gn, Gn_target_down, Gn_new, psf, use_ltp=True)
        self._pulse_branch(ltd_neg_mask & normal, pid, f_n,
                           Gn, Gn_target_up,   Gn_new, psf, use_ltp=False)
        # 4-way branching for DEGRADED cells (degraded fitter)
        if f_d is not None:
            self._pulse_branch(ltp_pos_mask & degraded, pid, f_d,
                               Gp, Gp_target_up,   Gp_new, psf, use_ltp=True, deg=True)
            self._pulse_branch(ltd_pos_mask & degraded, pid, f_d,
                               Gp, Gp_target_down, Gp_new, psf, use_ltp=False, deg=True)
            self._pulse_branch(ltp_neg_mask & degraded, pid, f_d,
                               Gn, Gn_target_down, Gn_new, psf, use_ltp=True, deg=True)
            self._pulse_branch(ltd_neg_mask & degraded, pid, f_d,
                               Gn, Gn_target_up,   Gn_new, psf, use_ltp=False, deg=True)

        if self.use_discretisation:
            Gp_new = f_n.discretise(Gp_new)
            Gn_new = f_n.discretise(Gn_new)

        Gp_new = Gp_new.clamp(0.0, tmax)
        Gn_new = Gn_new.clamp(0.0, tmax)
        self.G_pos[pid] = Gp_new
        self.G_neg[pid] = Gn_new
        p.data.copy_((Gp_new - Gn_new).clamp(f_n.target_min, f_n.target_max))

    def _step_pair(self, p, pid, grad, lr, psf):
        """Dispatch to the selected pair strategy."""
        if self.pair_strategy == "mixed":
            return self._step_pair_mixed(p, pid, grad, lr, psf)
        return self._step_pair_ltp_only(p, pid, grad, lr, psf)

    # ---------- single-cell step (default, unchanged from v2 logic) ----------
    def _step_single(self, p, pid, grad, lr, psf):
        f_n = self.fitter
        f_d = self.degraded_fitter
        G_cur = p.data
        delta_W = -lr * grad
        G_tgt = (G_cur + delta_W).clamp(f_n.target_min, f_n.target_max)

        ltp_mask = grad < 0
        ltd_mask = grad > 0
        G_new = G_cur.clone()

        normal, degraded = self._split_normal_degraded(pid, G_cur)

        self._pulse_branch(ltp_mask & normal,  pid, f_n, G_cur, G_tgt, G_new, psf, use_ltp=True)
        self._pulse_branch(ltd_mask & normal,  pid, f_n, G_cur, G_tgt, G_new, psf, use_ltp=False)
        if f_d is not None:
            self._pulse_branch(ltp_mask & degraded, pid, f_d, G_cur, G_tgt, G_new, psf, use_ltp=True, deg=True)
            self._pulse_branch(ltd_mask & degraded, pid, f_d, G_cur, G_tgt, G_new, psf, use_ltp=False, deg=True)

        if self.use_discretisation:
            G_new = f_n.discretise(G_new)
        p.data.copy_(G_new.clamp(f_n.target_min, f_n.target_max))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            psf = group["pulse_scaling_factor"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                pid = id(p)
                grad = p.grad.data
                if self.pair_mode:
                    self._step_pair(p, pid, grad, lr, psf)
                else:
                    self._step_single(p, pid, grad, lr, psf)

    # ---------- energy reporting ----------
    def record_reads(self, n: int):
        """Add n read events (e.g. forward pass once over the model)."""
        self.total_reads += int(n)

    def reset_energy_counters(self):
        self.total_pulses_ltp = 0.0
        self.total_pulses_ltd = 0.0
        self.total_reads = 0

    def energy_report(self) -> dict:
        ep = self.energy_params
        e_write = (self.total_pulses_ltp * ep["E_pulse_LTP_J"] +
                   self.total_pulses_ltd * ep["E_pulse_LTD_J"])
        e_read = self.total_reads * ep["E_read_J"]
        return {
            "total_pulses_ltp": float(self.total_pulses_ltp),
            "total_pulses_ltd": float(self.total_pulses_ltd),
            "total_pulses":     float(self.total_pulses_ltp + self.total_pulses_ltd),
            "total_reads":      int(self.total_reads),
            "write_energy_J":   float(e_write),
            "read_energy_J":    float(e_read),
            "total_energy_J":   float(e_write + e_read),
            "array_area_m2":    float(self.total_weights * ep["cell_area_m2"] *
                                      (2 if self.pair_mode else 1)),
            "pair_mode":        bool(self.pair_mode),
            "num_weights":      int(self.total_weights),
        }
