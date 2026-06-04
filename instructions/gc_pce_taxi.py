"""
GC-PCE Experiment — S2P2 on Taxi dataset
Gap-Conditional Probabilistic Calibration Error

Run from examples/ directory:
    python gc_pce_taxi.py

All paths are relative to examples/.
"""

import sys
import os

# Add project root to path so easy_tpp is importable
sys.path.insert(0, os.path.abspath('..'))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — exact paths for your setup
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT  = 'checkpoints/taxi/S2P2/train_2026_06_02_18_44_09/models/saved_model.pt'
CONFIG_YAML = './configs/exp_config_taxi.yaml'
DEVICE      = 'cpu'     # your training used gpu: -1 so stay on CPU
N_MC        = 50        # integration points per inter-event interval
OUTPUT_PKL  = 'gc_pce_s2p2_taxi.pkl'
OUTPUT_PNG  = 'gc_pce_s2p2_taxi.png'

# ─────────────────────────────────────────────────────────────────────────────
# PART A — GC-PCE CORE FUNCTIONS (no model dependency)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pce(pit_values):
    """
    Probabilistic Calibration Error.
    PIT (Probability Integral Transform) values: u_i = F(tau_i | model, history).
    Under perfect calibration u_i ~ Uniform(0,1). PCE = deviation from that.
    Lower = better.
    """
    pit = np.asarray(pit_values, dtype=float)
    valid = pit[(pit > 0) & (pit < 1) & ~np.isnan(pit)]
    if len(valid) < 5:
        return np.nan
    n = len(valid)
    return float(np.mean(np.abs(np.sort(valid) - np.linspace(1.0 / n, 1.0, n))))


def compute_gc_pce(gaps, pit_values, n_q=4):
    """
    Compute PCE separately within each inter-arrival gap quantile bin.

    Returns:
        pce_vals  — PCE per bin, shape [n_q]
        edges     — gap boundaries, shape [n_q+1]
        labels    — x-axis labels for plotting
        counts    — number of events per bin, shape [n_q]
    """
    gaps = np.asarray(gaps, dtype=float)
    pit_values = np.asarray(pit_values, dtype=float)
    assert len(gaps) == len(pit_values), "gaps and pit_values must be same length"

    edges = np.quantile(gaps, np.linspace(0.0, 1.0, n_q + 1))
    pce_vals, counts, labels = [], [], []

    for q in range(n_q):
        lo, hi = edges[q], edges[q + 1]
        mask = (gaps >= lo) & (gaps <= hi) if q == n_q - 1 else (gaps >= lo) & (gaps < hi)
        pce_vals.append(compute_pce(pit_values[mask]))
        counts.append(int(mask.sum()))
        labels.append(f'Q{q+1}\n[{lo:.3f},{hi:.3f}]')

    return np.array(pce_vals), edges, labels, np.array(counts)


def print_results_table(gaps, pits):
    pce_vals, _, _, counts = compute_gc_pce(gaps, pits, 4)
    ratio = pce_vals[3] / pce_vals[0] if pce_vals[0] > 1e-9 else float('nan')

    print(f'\n{"="*60}')
    print('GC-PCE Results — S2P2 on Taxi')
    print(f'{"="*60}')
    print(f'  Q1 (shortest gaps) : PCE = {pce_vals[0]:.4f}   (n={counts[0]})')
    print(f'  Q2                 : PCE = {pce_vals[1]:.4f}   (n={counts[1]})')
    print(f'  Q3                 : PCE = {pce_vals[2]:.4f}   (n={counts[2]})')
    print(f'  Q4 (longest gaps)  : PCE = {pce_vals[3]:.4f}   (n={counts[3]})')
    print(f'  Q4 / Q1 ratio      : {ratio:.2f}')

    if ratio > 1.3:
        verdict = 'SUPPORTS hypothesis: ZOH degrades calibration for long gaps'
    elif ratio > 1.1:
        verdict = 'WEAK signal — slight gap-dependent degradation'
    else:
        verdict = 'FLAT — no gap-dependent degradation detected at this scale'
    print(f'  Verdict            : {verdict}')
    print(f'{"="*60}\n')


def plot_results(gaps, pits, save_path=OUTPUT_PNG):
    pce_vals, _, labels, counts = compute_gc_pce(gaps, pits, 4)
    ratio = pce_vals[3] / pce_vals[0] if pce_vals[0] > 1e-9 else float('nan')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        'Gap-Conditional PCE — S2P2 on Taxi\n'
        fr'Q4/Q1 ratio = {ratio:.2f}   '
        r'(>1.3 supports ZOH degradation hypothesis)',
        fontsize=11
    )

    # Left: bar chart
    ax1 = axes[0]
    bars = ax1.bar(range(4), pce_vals, color='#1D9E75',
                   alpha=0.82, edgecolor='white', linewidth=0.4)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('PCE  (lower = better calibrated)')
    ax1.set_title('PCE by inter-arrival gap quartile')
    ax1.grid(axis='y', alpha=0.25, linewidth=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for j, (v, c) in enumerate(zip(pce_vals, counts)):
        if not np.isnan(v):
            ax1.text(j, v * 1.03, f'n={c}', ha='center', fontsize=8, color='#333')

    # Right: trend line
    ax2 = axes[1]
    ax2.plot(range(1, 5), pce_vals, marker='o', color='#1D9E75',
             linewidth=2.5, markersize=8, zorder=3)
    ax2.fill_between(range(1, 5), pce_vals, alpha=0.12, color='#1D9E75')
    ax2.set_xticks(range(1, 5))
    ax2.set_xticklabels(['Q1\n(short)', 'Q2', 'Q3', 'Q4\n(long)'])
    ax2.set_ylabel('PCE')
    ax2.set_title('PCE trend Q1 → Q4\n(rising = ZOH hypothesis confirmed)')
    ax2.grid(alpha=0.25, linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.savefig(save_path, dpi=180, bbox_inches='tight')
    print(f'Plot saved: {save_path}')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# PART B — SYNTHETIC TEST (run first, no model needed)
# ─────────────────────────────────────────────────────────────────────────────

def run_synthetic_test():
    print('=== Synthetic test: verifying GC-PCE logic ===')
    rng = np.random.default_rng(42)
    n = 5000

    gaps = np.concatenate([
        rng.exponential(0.3, n // 3),
        rng.exponential(1.5, n // 3),
        rng.exponential(7.0, n - 2 * (n // 3)),
    ])
    gaps = np.maximum(gaps, 1e-6)

    # Perfect calibration: uniform PITs, should be flat
    pits_good = rng.uniform(0, 1, len(gaps))

    # ZOH-degraded: PITs skewed toward 0 for large gaps, should rise Q1→Q4
    log_g = np.log1p(gaps)
    scale = (log_g - log_g.min()) / (log_g.max() - log_g.min())
    pits_bad = np.array([rng.beta(1.0, 1.0 + s * 4.0) for s in scale])

    for name, pits in [('Calibrated (should be flat)', pits_good),
                       ('ZOH-degraded (should rise Q1→Q4)', pits_bad)]:
        v, _, _, _ = compute_gc_pce(gaps, pits, 4)
        print(f'  {name}:')
        print(f'    Q1={v[0]:.4f}  Q2={v[1]:.4f}  Q3={v[2]:.4f}  Q4={v[3]:.4f}  '
              f'ratio={v[3]/v[0]:.2f}')

    v_bad, _, _, _ = compute_gc_pce(gaps, pits_bad, 4)
    assert v_bad[3] > v_bad[0] * 1.3, (
        f'FAILED: expected Q4 >> Q1 for ZOH-degraded model, '
        f'got Q1={v_bad[0]:.4f}, Q4={v_bad[3]:.4f}'
    )
    print('PASSED.\n')


# ─────────────────────────────────────────────────────────────────────────────
# PART C — LOAD S2P2 MODEL
# Checkpoint is bare OrderedDict (confirmed). Class is in torch_s2p2.py.
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    from easy_tpp.model.torch_model.torch_s2p2 import S2P2
    from easy_tpp.config_factory import Config

    # Load state dict
    state_dict = torch.load(CHECKPOINT, map_location='cpu')

    # Auto-detect architecture dimensions from checkpoint keys
    # C_tilde_HP has shape [H, P]  →  row dim = H, col dim = P
    H = state_dict['layers.0.C_tilde_HP'].shape[0]
    P = state_dict['layers.0.C_tilde_HP'].shape[1]
    n_layers = sum(1 for k in state_dict if k.endswith('.C_tilde_HP'))

    print(f'  Checkpoint architecture: H={H}, P={P}, n_layers={n_layers}')

    # Build config from the same YAML used for training
    config = Config.build_from_yaml_file(CONFIG_YAML, experiment_id='S2P2_train')

    # Instantiate model with the training config
    model = S2P2(config.model_config)

    # Load weights
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Loaded S2P2: {n_params:,} parameters')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PART D — LOAD TAXI TEST DATA
# Uses EasyTPP's config-based runner, which mirrors what train.py does.
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data():
    from easy_tpp.config_factory import Config
    from easy_tpp.runner import TPPRunner

    config = Config.build_from_yaml_file(CONFIG_YAML, experiment_id='S2P2_train')
    config.base_config.stage = 'test'
    runner = TPPRunner.build_from_config(config)
    print('  Test loader found: runner._data_loader.test_loader')
    return runner._data_loader.test_loader()


# ─────────────────────────────────────────────────────────────────────────────
# PART E — EXTRACT (gap, PIT) PAIRS
# Uses compute_intensities_at_sample_times (confirmed in torch_s2p2.py:288).
# Processes full batches for efficiency.
# ─────────────────────────────────────────────────────────────────────────────

def extract_gaps_and_pits(model, test_loader):
    """
    For every inter-event interval in the test set, compute:
      gap   = t_i - t_{i-1}
      PIT   = 1 - exp( -integral_{t_{i-1}}^{t_i} lambda(s | history) ds )

    Under perfect calibration, PIT ~ Uniform(0,1).
    """
    all_gaps = []
    all_pits = []
    total_sequences = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Extracting PITs')):

            # Print batch keys on first iteration to confirm format
            if batch_idx == 0:
                print(f'\n[DEBUG] Batch keys: {list(batch.keys())}')

            # ── Standard EasyTPP batch keys ──────────────────────────────────
            # If these raise KeyError, print batch.keys() and adjust here.
            time_seqs       = batch['time_seqs'].to(DEVICE)        # [B, L] absolute times
            type_seqs       = batch['type_seqs'].to(DEVICE)        # [B, L] mark types

            # Inter-event time deltas — EasyTPP usually pre-computes these.
            # If 'time_delta_seqs' is not in your batch, we compute it below.
            if 'time_delta_seqs' in batch:
                time_delta_seqs = batch['time_delta_seqs'].to(DEVICE)   # [B, L]
            else:
                time_delta_seqs = torch.zeros_like(time_seqs)
                time_delta_seqs[:, 1:] = time_seqs[:, 1:] - time_seqs[:, :-1]
                # First delta: time from 0 to first event
                time_delta_seqs[:, 0] = time_seqs[:, 0]

            # Sequence lengths (non-padded events)
            if 'batch_non_pad_mask' in batch:
                seq_mask = batch['batch_non_pad_mask'].to(DEVICE)   # [B, L] float/bool
            elif 'seq_mask' in batch:
                seq_mask = batch['seq_mask'].to(DEVICE)
            else:
                # Fallback: any event with time > 0 is real
                seq_mask = (time_seqs > 0).float()

            B, L = time_seqs.shape

            # ── Build sample_dtimes: [B, L, N_MC] ────────────────────────────
            # For each event i, sample N_MC relative times in (0, delta_i).
            # We use evenly spaced fractions (trapezoid rule; avoids boundary issues).
            fracs = torch.linspace(0.02, 0.98, N_MC, device=DEVICE)          # [N_MC]
            sample_dtimes = time_delta_seqs.unsqueeze(-1) * fracs             # [B, L, N_MC]

            # ── Call S2P2 intensity function ──────────────────────────────────
            # Signature (confirmed from torch_s2p2.py:288):
            #   compute_intensities_at_sample_times(
            #       time_seqs, time_delta_seqs, type_seqs, sample_dtimes
            #   )
            # Returns: [B, L, N_MC, K] where K = num_event_types
            try:
                intensities = model.compute_intensities_at_sample_times(
                    time_seqs,
                    time_delta_seqs,
                    type_seqs,
                    sample_dtimes,
                )
            except TypeError as e:
                # Some versions pass **kwargs differently — try without time_seqs
                print(f'[WARN] First call failed ({e}), trying alternative signature...')
                intensities = model.compute_intensities_at_sample_times(
                    time_seqs,
                    time_delta_seqs,
                    type_seqs,
                    sample_dtimes,
                    compute_lambda=True,
                )

            # intensities shape: [B, L, N_MC, K] or [B, L, N_MC]
            if intensities.dim() == 4:
                total_intensity = intensities.sum(dim=-1)    # [B, L, N_MC] sum marks
            else:
                total_intensity = intensities                # [B, L, N_MC] already total

            # ── Numerical integration (trapezoidal) ──────────────────────────
            # integral over [0, delta_i] ≈ trapz(total_intensity[b,i,:], sample_dtimes[b,i,:])
            # Since sample_dtimes[b,i,:] = delta_i * fracs, and fracs is uniform:
            # trapz ≈ mean(total_intensity) * delta_i
            cum_hazard = total_intensity.mean(dim=-1) * time_delta_seqs   # [B, L]
            cum_hazard = cum_hazard.clamp(min=0.0)

            # ── PIT values ────────────────────────────────────────────────────
            pits_batch = 1.0 - torch.exp(-cum_hazard)                    # [B, L]
            pits_batch = pits_batch.clamp(1e-6, 1 - 1e-6)

            # ── Extract valid events (skip index 0, skip padding) ─────────────
            # We skip index 0 because there is no previous event to form an interval.
            for b in range(B):
                T = int(seq_mask[b].sum().item())
                if T < 2:
                    continue
                # Events 1..T-1 have a valid previous event
                for i in range(1, T):
                    all_gaps.append(time_delta_seqs[b, i].item())
                    all_pits.append(pits_batch[b, i].item())

            total_sequences += B

    print(f'\nExtracted {len(all_gaps):,} (gap, PIT) pairs from {total_sequences} sequences.')
    return np.array(all_gaps), np.array(all_pits)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='GC-PCE for S2P2 on Taxi')
    parser.add_argument('--skip-synthetic', action='store_true',
                        help='Skip synthetic test (not recommended on first run)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Reload saved pkl and re-plot without re-running model')
    args = parser.parse_args()

    # ── Phase 1: Verify GC-PCE logic (no model, runs in ~5 seconds) ──────────
    if not args.skip_synthetic:
        run_synthetic_test()

    if args.plot_only:
        print(f'Loading saved results: {OUTPUT_PKL}')
        with open(OUTPUT_PKL, 'rb') as f:
            saved = pickle.load(f)
        gaps, pits = saved['gaps'], saved['pits']
        print(f'Loaded {len(gaps):,} events.')

    else:
        # ── Phase 2: Load model ───────────────────────────────────────────────
        print('=== Loading S2P2 model ===')
        model = load_model()

        # ── Phase 3: Load test data ───────────────────────────────────────────
        print('=== Loading taxi test data ===')
        test_loader = load_test_data()

        # ── Phase 4: Extract PITs ─────────────────────────────────────────────
        print('=== Extracting (gap, PIT) pairs ===')
        gaps, pits = extract_gaps_and_pits(model, test_loader)

        # Save to disk — re-use with --plot-only without re-running model
        with open(OUTPUT_PKL, 'wb') as f:
            pickle.dump({'gaps': gaps, 'pits': pits}, f)
        print(f'Saved raw results: {OUTPUT_PKL}')

    # ── Phase 5: Compute GC-PCE and plot ─────────────────────────────────────
    print('=== Results ===')
    print_results_table(gaps, pits)
    plot_results(gaps, pits)

    print('\nFiles created:')
    print(f'  {OUTPUT_PNG}  ← the figure')
    print(f'  {OUTPUT_PKL}  ← raw data (re-plot anytime with --plot-only)')
