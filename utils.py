import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
import torch
import torchmetrics

class EER(torchmetrics.Metric):
    """
    This class calculates the Equal Error Rate (EER) for binary classification tasks using PyTorch and TorchMetrics.
    
    Args:
        positive_label (int): The label for the positive class.
        plot_path (str): The path to save the ROC curve plot. If None, the plot will not be saved.
        dist_sync_on_step (bool): Synchronize metric state across processes at each forward() before returning the value at the step.
    """
    def __init__(self, positive_label=1, plot_path=None, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("preds", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("targets", default=torch.tensor([]), dist_reduce_fx="cat")
        self.positive_label = positive_label
        self.plot_path = plot_path
        self.fpr = None
        self.tpr = None
        self.eer = None

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds = torch.cat((self.preds, preds), dim=0)
        self.targets = torch.cat((self.targets, targets), dim=0)

    def compute(self):
        preds = self.preds
        targets = self.targets
        all_pred_softmax = torch.nn.functional.softmax(preds, dim=-1)
        all_pred_pos_prob = all_pred_softmax[:, 1].cpu().numpy()
        targets = targets.cpu().numpy()
        # Filter any residual NaN/Inf scores that slipped through (last-resort guard)
        valid = np.isfinite(all_pred_pos_prob)
        if not valid.all():
            all_pred_pos_prob = all_pred_pos_prob[valid]
            targets = targets[valid]
        if len(all_pred_pos_prob) == 0:
            return torch.tensor(1.0)   # worst-case EER when no valid predictions
        fpr, tpr, _ = sklearn.metrics.roc_curve(targets, all_pred_pos_prob, pos_label=self.positive_label)
        fnr = 1 - tpr

        eer_1 = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer_2 = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = (eer_1 + eer_2) / 2

        if self.plot_path is not None:
            roc_auc = sklearn.metrics.auc(fpr, tpr)
            plt.figure(figsize=(10,10))
            lw = 2
            plt.rcParams["font.family"] = "CMU Serif"
            plt.rcParams['font.size'] = 18
            plt.rcParams['text.usetex'] = True
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
            self.fpr = fpr
            self.tpr = tpr
            self.eer = eer
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic ' + self.plot_path + ' EER: {:.1f} \%'.format(eer * 100))
            plt.legend(loc="lower right")
            try:
                plt.savefig(self.plot_path+'_roc.png')
            except Exception as e:
                print(f"Error saving the plot: {e}")
            plt.close()
        return torch.tensor(eer)
    


class AcousticMapExtractor(torch.nn.Module):  # unused — kept for reference only
    """
    GPU-ready, vectorized acoustic map extractor supporting SRP-PHAT, DS, and MVDR.

    Differences vs. previous build:
      - Axis permutation (default (2,0,1) = [z,x,y]).
      - torch.stft(center=False) to match SciPy boundary=None.
      - Vectorized power maps with chunking.
      - Enforce Hermitian CSM (0.5*(R + R^H)) for stability.
      - Safer frequency selection (warn & fallback).
      - No in-model normalization by default; plotting auto-scales by percentiles.
    """

    def __init__(
        self,
        mic_positions,
        fs,
        n_fft=2048,
        hop=1024,
        az_grid=None,
        el_grid=None,
        bands=None,
        normalize_in_model=True, 
        device='cpu',
        axis_permutation=(2, 0, 1), # MSFT convention: positions[:, [2,0,1]]
        center_stft=False           # Match scipy.signal.stft(..., boundary=None)
    ):
        super().__init__()
        self.fs = fs
        self.n_fft = n_fft
        self.hop = hop
        self.normalize_in_model = normalize_in_model
        self.center = center_stft

        # --- Buffers: move with .to(device) ---
        mic_positions_t = torch.as_tensor(mic_positions, dtype=torch.float32, device=device)  # (M, 3)
        if axis_permutation:
            mic_positions_perm = mic_positions_t[:, list(axis_permutation)]
        else:
            mic_positions_perm = mic_positions_t
        self.register_buffer('mic_positions', mic_positions_perm)
        self.n_mics = mic_positions_perm.shape[0]

        az_t = torch.linspace(-90.0, 90.0, 91, device=device, dtype=torch.float32) if az_grid is None \
               else torch.as_tensor(az_grid, dtype=torch.float32, device=device)
        el_t = torch.linspace(-90.0, 90.0, 41, device=device, dtype=torch.float32) if el_grid is None \
               else torch.as_tensor(el_grid, dtype=torch.float32, device=device)
        self.register_buffer('az_grid', az_t)
        self.register_buffer('el_grid', el_t)

        freqs_t = torch.linspace(0.0, fs / 2.0, n_fft // 2 + 1, device=device, dtype=torch.float32)
        self.register_buffer('freqs', freqs_t)

        window = torch.hann_window(self.n_fft, periodic=True, device=device, dtype=torch.float32)
        self.register_buffer('window', window)

        self.bands = bands or {'Low (100-500 Hz)': (100, 500),
                               'Mid (500-3000 Hz)': (500, 3000),
                               'High (3000-8000 Hz)': (3000, 8000),
                               'Super-High (8000-22050 Hz)': (8000, 22050)}

        # Precompute flattened direction grid
        elm, azm = torch.meshgrid(self.el_grid, self.az_grid, indexing='ij')
        self.register_buffer('grid_el_flat', elm.reshape(-1))  # (D,)
        self.register_buffer('grid_az_flat', azm.reshape(-1))  # (D,)

    # ---------------------- STFT / CSM ----------------------

    def stft(self, x):
        """
        x: (B, n_mics, n_samples) on the same device as the model
        Returns: (B, n_mics, n_freq, n_frames), complex64 (if x is float32)
        """
        B, n_mics, _ = x.shape
        x_2d = x.reshape(B * n_mics, -1)
        X = torch.stft(
            x_2d,
            n_fft=self.n_fft,
            hop_length=self.hop,
            return_complex=True,
            window=self.window.to(dtype=x.dtype, device=x.device),
            center=self.center  # False to match SciPy boundary=None
        )
        X = X.view(B, n_mics, X.shape[-2], X.shape[-1])
        return X

    def compute_csm(self, X, freq_idx):
        """
        Compute cross-spectral matrices (CSM) in double precision with Hermitian enforcement.

        X: (B, M, F, T) complex
        freq_idx: (n_freq_sel,)
        
        Returns:
            R: (B, n_freq_sel, M, M) complex128
        """
        # Select frequencies
        Xf = X[:, :, freq_idx, :]  # (B, M, Fsel, T)

        # Convert to double precision for stability
        Xf = Xf.to(torch.complex128)

        T = Xf.shape[-1]
        if T == 0:
            return torch.zeros((Xf.shape[0], len(freq_idx), Xf.shape[1], Xf.shape[1]), dtype=torch.complex128, device=X.device)

        # Compute CSM: R[b,f] = (X^H X) / T
        R = torch.einsum('bmft,bnft->bfmn', Xf, Xf.conj()) / float(T)

        # Enforce Hermitian symmetry
        R = 0.5 * (R + R.conj().transpose(-1, -2))

        return R

    # ---------------------- Steering (vectorized) ----------------------

    def _taus_all_dirs(self):
        """Mic delays for all directions: taus_all (M, D)."""
        az = torch.deg2rad(self.grid_az_flat)  # (D,)
        el = torch.deg2rad(self.grid_el_flat)  # (D,)
        ux = torch.cos(el) * torch.cos(az)
        uy = torch.cos(el) * torch.sin(az)
        uz = torch.sin(el)
        U = torch.stack([ux, uy, uz], dim=1)   # (D, 3)

        # taus(m, d) = <pos(m), u(d)> / c, then relative to mic 0
        taus = (self.mic_positions @ U.T) / 343.0  # (M, D)
        taus = taus - taus[0:1, :]
        return taus  # (M, D)

    # ---------------------- Vectorized Power Maps ----------------------


    @torch.no_grad()
    def power_map(self, R, freqs, bf_type='ds', R_phat=None, reg=1e-2):
        """
        Vectorized power maps over all directions.

        Inputs:
            R:      (B, Fsel, M, M) complex
            freqs:  (Fsel,) float (Hz)
            bf_type: 'srp' | 'ds' | 'mvdr'
            R_phat: (B, Fsel, M, M) complex [for 'srp'] or None
            reg:    float, diagonal loading for MVDR

        Returns:
            map_out: (B, n_el, n_az) float32
        """
        device = R.device
        dtype_c = R.dtype
        dtype_r = torch.float64 if dtype_c == torch.complex128 else torch.float32
        B, Fsel, M, _ = R.shape
        E, A = self.el_grid.numel(), self.az_grid.numel()
        D = E * A

        eps_val = torch.finfo(dtype_r).eps
        freqs = freqs.to(device=device, dtype=dtype_r)
        taus_all = self._taus_all_dirs().to(device=device, dtype=dtype_r)  # (M, D)

        map_out = torch.zeros((B, D), dtype=torch.float32, device=device)
        if Fsel == 0 or D == 0:
            return map_out.view(B, E, A)

        if bf_type == 'srp' and R_phat is None:
            R_abs = R.abs().clamp_min(eps_val)
            R_phat = R / R_abs

        I = torch.eye(M, dtype=dtype_c, device=device)[None, :, :]  # (1, M, M)

        # Steering matrix for all directions (Fsel, M, D)
        A_fmd = torch.exp((-2j) * torch.pi * freqs[:, None, None] * taus_all[None, :, :]).to(dtype=dtype_c)

        if bf_type == 'srp':
            score_bd = torch.einsum('bfmn,fmd,fnd->bd', R_phat, A_fmd.conj(), A_fmd).real
            map_out = score_bd.to(torch.float32)

        elif bf_type == 'ds':
            num_bfd = torch.einsum('bfmn,fmd,fnd->bfd', R, A_fmd.conj(), A_fmd).real
            den_fd  = torch.einsum('fmd,fmd->fd', A_fmd.conj(), A_fmd).real + eps_val
            pf_bd   = (num_bfd / den_fd).sum(dim=1)
            map_out = pf_bd.to(torch.float32)

        elif bf_type == 'mvdr':
            R = R.to(torch.complex64)
            freqs = freqs.to(torch.float32)
            trace_R = R.diagonal(dim1=-2, dim2=-1).sum(-1) / M
            R_reg = R + reg * trace_R[:, :, None, None] * I
            R_inv = torch.linalg.pinv(R_reg)
            A_bfmd = A_fmd[None, :, :, :].expand(B, -1, -1, -1)
            x = torch.einsum('bfmn,bfnd->bfmd', R_inv, A_bfmd)
            denom_bfd = torch.einsum('bfmd,bfmd->bfd', A_bfmd.conj(), x).real.clamp_min(eps_val)
            pf_bd = (1.0 / denom_bfd).sum(dim=1)
            map_out = pf_bd.to(torch.float32)

        if getattr(self, 'normalize_in_model', False):
            minv = map_out.amin(dim=1, keepdim=True)
            maxv = map_out.amax(dim=1, keepdim=True)
            map_out = (map_out - minv) / (maxv + eps_val)

        return map_out.view(B, E, A)

    # ---------------------- Forward / Plot ----------------------

    def _select_freqs(self, fmin, fmax):
        # match NumPy: exclude DC explicitly; warn if none found
        idx = torch.where((self.freqs >= fmin) & (self.freqs <= fmax) & (self.freqs > 0))[0]
        if idx.numel() == 0:
            # fallback: include closest valid bins (avoid empty selection)
            close = torch.where(self.freqs > 0)[0]
            if close.numel() == 0:
                print(f"[WARN] No positive frequencies at all (fs={self.fs}).")
                return idx  # empty
            # pick up to 3 closest bins around fmin..fmax range
            idx = close[: min(3, close.numel())]
            print(f"[WARN] Empty band [{fmin},{fmax}] Hz at fs={self.fs}; falling back to {self.freqs[idx].cpu().numpy()}.")
        return idx

    def forward(self, x, verbose=True):
        X = self.stft(x)
        if X.shape[-1] == 0 and verbose:
            print("[WARN] STFT produced zero frames: increase signal length or enable centering.")

        outputs, names = [], []
        for band_name, (fmin, fmax) in self.bands.items():
            freq_idx = self._select_freqs(fmin, fmax)
            if freq_idx.numel() == 0:
                if verbose:
                    print(f"[WARN] Skipping band {band_name} (no freqs).")
                continue

            if verbose:
                fsel = self.freqs[freq_idx]
                print(f"[INFO] {band_name}: {freq_idx.numel()} bins from {float(fsel.min()):.1f} to {float(fsel.max()):.1f} Hz")

            R = self.compute_csm(X, freq_idx)
            map_out = self.power_map(R, self.freqs[freq_idx], bf_type='ds')
            outputs.append(map_out)
            names.append(f'Delay-and-Sum - {band_name}')

        maps = torch.stack(outputs, dim=1) if outputs else torch.zeros((x.shape[0], 0, self.el_grid.numel(), self.az_grid.numel()), device=x.device)
        return maps, {'map_names': names, 'az_grid': self.az_grid, 'el_grid': self.el_grid}

    def show_maps(
    self,
    maps,
    meta,
    suptitle=None,
    figsize=(15, 12),
    cmap='viridis',
    normalize=True,
):
        """
        Render a 3x3 grid like the old NumPy script:
        - Rows = bands in self.bands order
        - Cols = ['SRP-PHAT', 'Delay-and-Sum', 'MVDR']
        - Per-map min-max normalization to [0,1] (normalize=True)
        - 'viridis' colormap, individual colorbars per subplot
        - Titles 'BF - Band' and optional suptitle
        - extent/origin/aspect identical to the NumPy version

        Args:
            maps: Tensor (B, n_maps, n_el, n_az)
            meta: {'map_names': list[str], 'az_grid': tensor, 'el_grid': tensor}
            suptitle: Optional string for figure suptitle (e.g., filename)
            figsize: default (15,12)
            cmap: default 'viridis'
            normalize: if True, per-map min-max normalization to [0,1]
        """

        if maps is None or maps.numel() == 0:
            print("[WARN] No maps to display.")
            return

        B, n_maps, n_el, n_az = maps.shape

        # Preserve NumPy-band order (dict insertion order)
        bands_order = list(self.bands.keys())
        bf_order = ['SRP-PHAT', 'Delay-and-Sum', 'MVDR']

        # Map from "BF - Band" -> tensor index in maps' 2nd dimension
        idx_by_key = {}
        for i, name in enumerate(meta.get('map_names', [])):
            if ' - ' in name:
                bf, band = name.split(' - ', 1)
                idx_by_key[(bf.strip(), band.strip())] = i

        az_min, az_max = float(self.az_grid[0]), float(self.az_grid[-1])
        el_min, el_max = float(self.el_grid[0]), float(self.el_grid[-1])
        extent = [az_min, az_max, el_min, el_max]

        for b in range(B):
            fig, axes = plt.subplots(len(bands_order), len(bf_order), figsize=figsize, squeeze=False)

            for i, band in enumerate(bands_order):
                for j, bf in enumerate(bf_order):
                    ax = axes[i, j]
                    key = (bf, band)

                    if key in idx_by_key:
                        data = maps[b, idx_by_key[key]].flip(dims=(1,0)).detach().cpu().float().numpy()
                    else:
                        # If a particular (bf, band) was not produced, show zeros to keep grid shape
                        data = np.zeros((n_el, n_az), dtype=np.float32)

                    if normalize:
                        dmin = float(data.min())
                        dmax = float(data.max())
                        rng = dmax - dmin
                        if rng <= 1e-12:
                            data = np.zeros_like(data)
                        else:
                            data = (data - dmin) / (rng + 1e-12)

                    im = ax.imshow(
                        data,
                        origin='lower',
                        aspect='auto',
                        extent=extent,
                        cmap=cmap
                    )
                    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(f"{bf} - {band}")
                    ax.set_xlabel("Azimuth (deg)")
                    ax.set_ylabel("Elevation (deg)")

            if suptitle is not None:
                fig.suptitle(str(suptitle))
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            else:
                fig.tight_layout()

            plt.show()



    

