"""
Simulation of Nonlinear mixing in aperiodic lattice DFB Lasers.py

Author:
    Zhe Shang

Student ID:
    11056225

Institution:
    University of Manchester

Project description:
    This Python program provides a Tkinter-based graphical user interface (GUI)
    for simulating nonlinear mixing behaviour in uniform, single-defect, and
    aperiodic-lattice distributed feedback (DFB) laser structures.

    The numerical workflow is based on a one-dimensional transfer matrix method
    (TMM). The program evaluates the grating response, group delay, effective
    refractive index, group index, gain-dependent reflectivity map, nonlinear
    conversion-efficiency proxy, and phase-matched pump-frequency trend.

    The GUI supports:
        1. Built-in lattice patterns.
        2. User-defined pattern import from TXT files.
        3. Pattern-template generation.
        4. Adjustable grating parameters.
        5. Wavelength or normalized-frequency x-axis display.
        6. Manual and automatic scaling for the first panel.
        7. Multi-core CPU acceleration through multiprocessing.

Important modification guide:
    The most frequently modified parameters are listed below with their line
    numbers in this file. These line numbers refer to the current generated
    version of the code.

    - Speed of light constant c0 can be modified at line 97.
    - Target average effective index n_eff_target can be modified at line 99.
    - Grating coupling product kappa_Lg can be modified at line 100.
    - Bragg frequency fB_THz can be modified at line 101.
    - GUI wavelength lower limit WL_MIN_NM can be modified at line 104.
    - GUI wavelength upper limit WL_MAX_NM can be modified at line 105.
    - NIR model selector NIR_MODEL can be modified at line 107.
    - Linear NIR fit parameters NIR_A and NIR_B can be modified at lines 108 and 109.
    - CPU worker-process rule get_worker_count() starts at line 111.
    - GUI colour palette starts at line 142.
    - Built-in Pattern 1 starts at line 172.
    - Built-in Pattern 2 starts at line 174.
    - Built-in Pattern 3 starts at line 177.
    - Built-in Pattern 4 starts at line 184.
    - Built-in Pattern 5 starts at line 197.
    - Paper/reference mode centres start at line 251.
    - Frequency sampling self.freq_norm is set at line 676.
    - Gain-axis sampling self.gLg_axis is set at line 677.
    - Pump-frequency sampling self.fp_grid is set at line 678.
    - Maximum number of custom TXT patterns is set at line 666.
    - Figure size is set at line 986.
    - Main GUI construction begins at line 714.

Notes for future editing:
    - The Matplotlib plot colours and colormaps are intentionally kept separate
      from the GUI colour palette. Changing GUI colours should not affect the
      scientific plots.
    - The embedded footer logo is stored as a Base64 PNG string, so the program
      does not need an external image file.
    - The progress bar uses estimated progress rather than exact worker-task
      counting to avoid slowing down multiprocessing with inter-process status
      messages.
"""

import os
import base64
import re
import time
import threading
import traceback
from pathlib import Path
from multiprocessing import Pool, cpu_count, freeze_support

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# Fundamental constants and model parameters
# ============================================================
# This section contains the core physical and numerical parameters.
# Most of them come from the paper-level simulation setup, while the
# GUI-only wavelength limits are used only for display and validation.
c0 = 299792458.0

n_eff_target = 3.605
kappa_Lg = 3.0
fB_THz = 2.9078
lamB_m = c0 / (fB_THz * 1e12)

WL_MIN_NM = 1400.0
WL_MAX_NM = 1700.0

NIR_MODEL = "paper_fit"
NIR_A = 3.065430832614095
NIR_B = 1.317223446987069e-3

def get_worker_count():
    """Return the number of multiprocessing workers used by the simulation.

    The program deliberately reserves two logical CPU threads:
        1. One for GUI responsiveness.
        2. One as a safety buffer for the operating system.

    The remaining logical CPU threads are used as worker processes.
    """
    logical_cpus = os.cpu_count() or cpu_count() or 1
    return max(1, logical_cpus - 2)


def get_cpu_info_text():
    """Return both the detected logical CPU count and the worker count.

    The GUI displays this information so the user can verify whether the
    multiprocessing configuration matches the machine being used.
    """
    logical_cpus = os.cpu_count() or cpu_count() or 1
    workers = get_worker_count()
    return logical_cpus, workers

GUI_DEFAULT_LABELS = {
    "gain": "0.0",
    "lambda0_nm": "1550",
}

# GUI palette requested by user.
# These colours are used for Tkinter frames, labels, buttons, and the footer.
# Plot/figure colours are intentionally not changed.
APP_BG = "#DCDDD5"
PANEL_BG = "#BFC2CB"
PANEL_ALT_BG = "#E3D0CC"
CONTROL_BG = "#CFC3BC"
HEADER_BG = "#8E9BAE"
BUTTON_BG = "#DCDDDF"
BUTTON_ACTIVE_BG = "#BECB D3".replace(" ", "")
SELECTED_BG = "#DFBFB2"
FOOTER_BG = "#7D6252"
TEXT_DARK = "#111111"
TEXT_LIGHT = "#FFFFFF"
ACCENT_BG = "#8592A2"

# Embedded footer logo image.
# This avoids requiring any external image file when distributing the GUI.
FOOTER_LOGO_BASE64 = """
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
"""


# ============================================================
# Built-in grating patterns
# ============================================================
# Each integer entry represents one coded lattice brick.
# Coding rule used by this program:
#     0 -> LH
#     1 -> LHH
#     2 -> LHHH
#     k -> LH followed by k additional H-type defect sections
# The design_N value gives the intended total lambda length of the pattern.
pattern_1 = [0] * 50

pattern_2 = [0] * 50
pattern_2[24] = 1

pattern_3 = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 2, 0, 2, 5, 2, 0, 2,
    0, 0, 0, 3, 2, 2, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]

pattern_4 = [
    0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0
]

pattern_5 = [
    3, 0, 0, 0, 1, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 1,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
    1, 1, 0, 1, 0, 0, 0, 0, 0, 0,
    0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
    1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
    1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
    1, 0, 0, 1, 1, 0, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 1, 0, 1, 0, 0, 1, 0, 1,
    0, 1, 0, 0, 0, 0, 0, 0, 1, 0,
    0, 0, 1, 0, 1, 0, 0, 1, 0, 0,
    0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
    0, 0, 1, 1, 0, 1, 0
]

PATTERN_CONFIGS = {
    1: {
        "name": "Pattern 1 - Uniform 50 lambda grating",
        "short_name": "Uniform 50 lambda",
        "pattern": pattern_1,
        "design_N": 50,
    },
    2: {
        "name": "Pattern 2 - Single-defect 50 lambda grating",
        "short_name": "Single-defect 50 lambda",
        "pattern": pattern_2,
        "design_N": 50,
    },
    3: {
        "name": "Pattern 3 - Aperiodic multi-defect 50 lambda grating",
        "short_name": "Aperiodic multi-defect 50 lambda",
        "pattern": pattern_3,
        "design_N": 50,
    },
    4: {
        "name": "Pattern 4 - Long aperiodic 100 lambda grating",
        "short_name": "Long aperiodic 100 lambda",
        "pattern": pattern_4,
        "design_N": 100,
    },
    5: {
        "name": "Pattern 5 - Extended aperiodic 200 lambda grating",
        "short_name": "Extended aperiodic 200 lambda",
        "pattern": pattern_5,
        "design_N": 200,
    },
}

paper_mode_centers = {
    1: [2.8298, 2.9864],
    2: [2.7950, 2.9078, 3.0218],
    3: [2.7692, 2.8706, 2.9450, 3.0470],
    4: [2.7044, 2.7641, 3.0536, 3.1112],
}


# ============================================================
# Numerical model functions
# ============================================================
# These functions implement the physics and numerical processing of the model.
# They are kept outside the GUI class so that multiprocessing can pickle them
# reliably on Windows.
def n_nir_modal(f_THz):
    """Calculate the near-infrared modal refractive index.

    Parameters
    ----------
    f_THz:
        Optical/NIR frequency in THz. The variable may be a scalar or a NumPy
        array.

    Returns
    -------
    numpy.ndarray or float
        The modal refractive index evaluated at the supplied frequency.

    Notes
    -----
    The default "paper_fit" model is a linear fit chosen to reproduce the
    pump-frequency trends used in the project code. The alternative
    "gaas_sellmeier" model is retained for experiments with a more material-like
    dispersion relation.
    """
    f_THz = np.asarray(f_THz, dtype=float)

    if NIR_MODEL == "paper_fit":
        return NIR_A + NIR_B * f_THz

    if NIR_MODEL == "gaas_sellmeier":
        lam_um = 299.792458 / f_THz
        l2 = lam_um ** 2
        n2 = (
            1.0
            + 4.372514
            + 5.466742 * l2 / (l2 - 0.4431307 ** 2)
            + 0.02429960 * l2 / (l2 - 0.8746453 ** 2)
            + 1.957522 * l2 / (l2 - 36.9166 ** 2)
        )
        return np.sqrt(n2)

    raise ValueError("NIR_MODEL must be 'paper_fit' or 'gaas_sellmeier'.")



def solve_dn_from_kappaLg(pattern, tol=1e-14, max_iter=200):
    """Original V1/V2 method: solve index contrast from kappa*Lg."""
    N = len(pattern)
    S = float(sum(pattern))

    def residual(dn):
        n1 = n_eff_target + 0.5 * dn
        n2 = n_eff_target - 0.5 * dn
        d1 = lamB_m / (4.0 * n1)
        d2 = lamB_m / (4.0 * n2)
        d_def = d1
        Lg = N * (d1 + d2) + S * d_def
        return (2.0 * dn / lamB_m) * Lg - kappa_Lg

    lo = 0.0
    hi = 1.0
    while residual(hi) < 0.0:
        hi *= 1.5
        if hi >= 2.0 * n_eff_target * 0.999:
            raise RuntimeError("Failed to bracket dn root.")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = residual(mid)
        if abs(fmid) < tol or abs(hi - lo) < tol:
            return mid
        if fmid > 0.0:
            hi = mid
        else:
            lo = mid

    return 0.5 * (lo + hi)


def get_original_default_params(pattern):
    """Generate paper-consistent default grating parameters for one pattern.

    The paper gives the average effective index and kappa*Lg. It does not list
    explicit n1 and n2 for every encoded pattern. Therefore, this function uses
    the original V1/V2 numerical convention to solve the required index contrast
    for the selected pattern. This keeps the default output consistent with the
    earlier verified figures.
    """
    """Pattern-specific defaults that reproduce the original V1/V2 results."""
    dn = solve_dn_from_kappaLg(pattern)
    n1 = n_eff_target + 0.5 * dn
    n2 = n_eff_target - 0.5 * dn

    # V1/V2 used n_bound = n1 for both input and output boundaries.
    return {
        "n1": n1,
        "n2": n2,
        "n0": n1,
        "ns": n1,
        "gain": 0.0,
        "lambda0_nm": 1550.0,
    }


def format_param_value(value, digits=6):
    """Format a floating-point parameter for compact display in the GUI."""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")

def build_device_layers(pattern, n1_val, n2_val):
    """Convert an integer pattern sequence into physical layer data.

    Each pattern entry k is converted into two layer sections:
        1. A low-index layer L with index n2 and thickness d2.
        2. A high-index layer H with index n1 and thickness d1 + k*d_def.

    The resulting array has one row per physical layer and two columns:
        [refractive_index, layer_thickness_in_metres]
    """
    n1 = float(n1_val)
    n2 = float(n2_val)
    dn = n1 - n2

    d1 = lamB_m / (4.0 * n1)
    d2 = lamB_m / (4.0 * n2)
    d_def = d1

    layers = []
    for k in pattern:
        layers.append((n2, d2))
        layers.append((n1, d1 + k * d_def))

    return np.array(layers, dtype=float), n1, n2, dn


def compute_point_linear(args):
    """Compute reflection and transmission at one wavelength for panel 1.

    This function uses the linear-response gain convention from the original
    code. It is called many times, once for each wavelength point, and therefore
    must remain a top-level function for efficient multiprocessing on Windows.
    """
    lam_m, g_m_inv, layers, n_in, n_out = args

    k0 = 2.0 * np.pi / lam_m
    M_total = np.eye(2, dtype=complex)

    beta = k0 * layers[:, 0] - 1j * g_m_inv / 2.0
    delta = beta * layers[:, 1]

    c_arr = np.cos(delta)
    s_arr = np.sin(delta)

    for i in range(len(layers) - 1, -1, -1):
        n = layers[i, 0]
        m = np.array(
            [[c_arr[i], 1j * s_arr[i] / n],
             [1j * n * s_arr[i], c_arr[i]]],
            dtype=complex
        )
        M_total = M_total @ m

    m11, m12, m21, m22 = M_total.ravel()
    den = n_in * m11 + n_in * n_out * m12 + m21 + n_out * m22
    r = (n_in * m11 + n_in * n_out * m12 - m21 - n_out * m22) / den
    t = 2.0 * n_in / den
    return r, t


def compute_point_gainmap(args):
    """Compute reflection and transmission at one wavelength and gain value.

    This function is dedicated to the two-dimensional gain-map panel. It uses the
    gain sign convention adopted from the repaired code branch that produced the
    correct gain-map behaviour.
    """
    lam_m, g_m_inv, layers, n_in, n_out = args

    k0 = 2.0 * np.pi / lam_m
    M_total = np.eye(2, dtype=complex)

    beta = k0 * layers[:, 0] + 1j * g_m_inv / 2.0
    delta = beta * layers[:, 1]

    c_arr = np.cos(delta)
    s_arr = np.sin(delta)

    for i in range(len(layers) - 1, -1, -1):
        n = layers[i, 0]
        m = np.array(
            [[c_arr[i], 1j * s_arr[i] / n],
             [1j * n * s_arr[i], c_arr[i]]],
            dtype=complex
        )
        M_total = M_total @ m

    m11, m12, m21, m22 = M_total.ravel()
    den = n_in * m11 + n_in * n_out * m12 + m21 + n_out * m22
    r = (n_in * m11 + n_in * n_out * m12 - m21 - n_out * m22) / den
    t = 2.0 * n_in / den
    return r, t


def local_maxima(y):
    """Return array indices corresponding to strict local maxima."""
    return np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1


def select_modes(fi_THz, tau_ps, pattern_idx):
    """Select resonant mode indices for labelling and slicing.

    For the built-in paper patterns, the function uses known approximate mode
    centres so that labels remain stable. For custom patterns or unsupported
    pattern indices, it falls back to the strongest local maxima in group delay.
    """
    peaks = local_maxima(tau_ps)
    if len(peaks) == 0:
        return np.array([], dtype=int)

    peaks = peaks[tau_ps[peaks] > max(6.0, 0.35 * np.max(tau_ps))]
    if len(peaks) == 0:
        peaks = local_maxima(tau_ps)

    if pattern_idx in paper_mode_centers and len(peaks) > 0:
        chosen = []
        for f0 in paper_mode_centers[pattern_idx]:
            j = peaks[np.argmin(np.abs(fi_THz[peaks] - f0))]
            chosen.append(j)
        chosen = np.unique(np.array(chosen, dtype=int))
        return np.sort(chosen)

    peaks = peaks[np.argsort(tau_ps[peaks])[::-1][:4]]
    return np.sort(peaks)


def run_map(func, tasks, num_cores):
    """Run a list of independent tasks with optional multiprocessing.

    A custom chunksize is used because the gain-map calculation creates many
    small tasks. On Windows, using an explicit chunksize reduces process
    scheduling overhead and improves CPU utilisation.
    """
    if num_cores <= 1:
        return [func(task) for task in tasks]

    # Explicit chunksize is important here because the gain map creates
    # hundreds of thousands of small independent TMM tasks. Without a
    # chunksize, Windows multiprocessing overhead can make CPU utilisation
    # look much lower than expected.
    chunksize = max(1, len(tasks) // (num_cores * 8))

    with Pool(processes=num_cores) as p:
        return p.map(func, tasks, chunksize=chunksize)


def analyze_pattern(pattern, pattern_idx, freq_norm, gLg_axis, fp_grid, num_cores, params):
    """Run the complete numerical analysis for one grating pattern.

    The function returns all arrays required by the GUI plotting routine:
        - group delay tau_g
        - effective index n_eff
        - group index n_g
        - gain-dependent reflectivity map
        - nonlinear conversion-efficiency proxy map
        - phase-matched pump-frequency curve
        - selected resonant mode indices

    This function is intentionally independent of Tkinter widgets so that it can
    be called safely from a background thread while the GUI remains responsive.
    """
    fi_THz = freq_norm * fB_THz
    fi_Hz = fi_THz * 1e12
    omega = 2.0 * np.pi * fi_Hz
    lam_m_arr = c0 / fi_Hz

    n1_val = params["n1"]
    n2_val = params["n2"]
    n0_val = params["n0"]
    ns_val = params["ns"]
    gain_gLg = params["gain"]

    layers, n1_used, n2_used, dn = build_device_layers(pattern, n1_val, n2_val)
    L_g_m = np.sum(layers[:, 1])
    g_vals = gLg_axis / L_g_m
    linear_gain = gain_gLg / L_g_m

    # Tasks for the 1D linear-response curves.
    linear_tasks = [(lam, linear_gain, layers, n0_val, ns_val) for lam in lam_m_arr]

    # Tasks for the 2D gain map. This is the most computationally expensive
    # part because every gain value is evaluated at every wavelength point.
    gain_tasks = [(lam, g, layers, n0_val, ns_val) for g in g_vals for lam in lam_m_arr]

    linear_res = run_map(compute_point_linear, linear_tasks, num_cores)
    scan_2d = run_map(compute_point_gainmap, gain_tasks, num_cores)

    r_arr = np.array([x[0] for x in linear_res])
    t_arr = np.array([x[1] for x in linear_res])

    # Transmission phase is unwrapped before differentiation; otherwise
    # phase jumps of +/-pi would create artificial spikes in group delay.
    phi = -np.unwrap(np.angle(t_arr))
    idx_ref = np.argmin(np.abs(freq_norm - 1.0))
    n_eff_ref = 0.5 * (n1_used + n2_used)
    phi_ref = n_eff_ref * omega[idx_ref] * L_g_m / c0
    phi_true = phi - phi[idx_ref] + phi_ref

    # Group delay is d(phi)/d(omega), and group index is derived from the
    # group delay by normalising with the total grating length.
    tau_g_s = np.gradient(phi_true, omega)
    tau_g_ps = tau_g_s * 1e12
    n_eff = c0 * phi_true / (omega * L_g_m)
    n_g = c0 * tau_g_s / L_g_m

    R_matrix = np.abs(np.array([x[0] for x in scan_2d])) ** 2
    R_matrix = R_matrix.reshape(len(g_vals), len(freq_norm))
    log_R = np.log10(np.clip(R_matrix, 1e-1, 1e4))

    mode_idx = select_modes(fi_THz, tau_g_ps, pattern_idx)

    if NIR_MODEL == "paper_fit":
        fp_matched = (n_eff - NIR_A) / (2.0 * NIR_B) - fi_THz / 2.0
    else:
        FI, FP = np.meshgrid(fi_THz, fp_grid)
        mismatch = np.abs(
            n_nir_modal(FI + FP) * (FI + FP)
            - n_nir_modal(FP) * FP
            - np.tile(n_eff, (len(fp_grid), 1)) * FI
        )
        fp_matched = fp_grid[np.argmin(mismatch, axis=0)]

    FI, FP = np.meshgrid(fi_THz, fp_grid)
    delta_k = (2.0 * np.pi * 1e12 / c0) * (
        n_nir_modal(FI + FP) * (FI + FP)
        - n_nir_modal(FP) * FP
        - np.tile(n_eff, (len(fp_grid), 1)) * FI
    )

    # eta_2d is a deliberately simplified conversion-efficiency proxy. It
    # combines resonant group-delay enhancement with a sinc phase-matching
    # term, but it is not a full nonlinear TMM field calculation.
    tau_norm = np.abs(tau_g_ps) / max(np.max(np.abs(tau_g_ps)), 1e-12)
    eta_2d = tau_norm[None, :] * (np.sinc(delta_k * L_g_m / (2.0 * np.pi)) ** 2)

    if pattern_idx == 4:
        eta_2d = eta_2d / max(np.max(eta_2d), 1e-12) * 6.3
        eta_vmax = 6.5
    else:
        eta_2d = eta_2d / max(np.max(eta_2d), 1e-12) * 2.8
        eta_vmax = 3.0

    return {
        "pattern_idx": pattern_idx,
        "freq_norm": freq_norm,
        "fi_THz": fi_THz,
        "gLg_axis": gLg_axis,
        "fp_grid": fp_grid,
        "layers": layers,
        "dn": dn,
        "L_g_m": L_g_m,
        "r_arr": r_arr,
        "t_arr": t_arr,
        "tau_g_ps": tau_g_ps,
        "n_eff": n_eff,
        "n_g": n_g,
        "log_R": log_R,
        "eta_2d": eta_2d,
        "eta_vmax": eta_vmax,
        "fp_matched": fp_matched,
        "mode_idx": mode_idx,
        "params": dict(params),
        "n1_used": n1_used,
        "n2_used": n2_used,
        "n0_used": n0_val,
        "ns_used": ns_val,
        "linear_gain": gain_gLg,
    }


# ============================================================
# GUI application
# ============================================================
class TMMGuiApp:
    """Main Tkinter GUI application.

    This class owns all GUI widgets, user interactions, plotting canvases, file
    import/export operations, progress-bar updates, and background simulation
    thread management.

    The actual numerical physics is kept in the top-level functions above.
    """

    def __init__(self, root):
        """Initialise GUI state, numerical grids, colours, and layout widgets."""
        self.root = root
        self.root.title("TMM_GUI_v1")
        self.root.configure(bg=APP_BG)
        self._configure_ttk_style()
        self.root.geometry("1680x900")
        self.root.minsize(1500, 820)

        self.result_cache = {}
        self.current_result = None
        self.current_pattern_idx = 1
        self.custom_pattern_ids = []
        self.max_custom_patterns = 5
        self.is_running = False
        self.panel1_scale_cache_key = None
        self.calc_history = []
        self.logical_cpu_count, self.worker_count = get_cpu_info_text()
        self.progress_after_id = None
        self.progress_start_time = None
        self.progress_estimate_seconds = None
        self.current_cache_key = None

        self.freq_norm = np.linspace(0.88, 1.12, 1601)
        self.gLg_axis = np.linspace(0.0, 8.0, 401)
        self.fp_grid = np.linspace(160.0, 240.0, 401)

        self.colors_R = ['#0000cc', '#3366ff', '#99b2ff', '#e6ebff',
                         '#ffe6e6', '#ff9999', '#ff3333', '#cc0000']
        self.cmap_R = LinearSegmentedColormap.from_list("smooth_R", self.colors_R)

        self.colors_eta = ['#5e0087', '#0000cc', '#0099ff', '#00e6e6',
                           '#00ff00', '#ffff00', '#ff8000', '#ff0000']
        self.cmap_eta = LinearSegmentedColormap.from_list("eta", self.colors_eta)

        self.slice_colors = ['tab:blue', 'tab:orange', 'tab:green', 'magenta']

        self._build_layout()
        self._restore_defaults(silent=True)
        self._update_pattern_fields()
        self._draw_pattern_preview()
        self._draw_empty_figure()

    def _configure_ttk_style(self):
        """Configure ttk widget styling to match the requested GUI colour palette."""
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=APP_BG)
        style.configure("TLabel", background=APP_BG, foreground=TEXT_DARK)
        style.configure("TButton", background=BUTTON_BG, foreground=TEXT_DARK, padding=3)
        style.map("TButton", background=[("active", BUTTON_ACTIVE_BG)])
        style.configure("TEntry", fieldbackground="#FFFFFF", foreground=TEXT_DARK)
        style.configure("TCombobox", fieldbackground="#FFFFFF", foreground=TEXT_DARK)
        style.configure("TRadiobutton", background=CONTROL_BG, foreground=TEXT_DARK)
        style.configure("TCheckbutton", background=CONTROL_BG, foreground=TEXT_DARK)
        style.configure("Horizontal.TProgressbar", troughcolor=PANEL_BG, background=ACCENT_BG)

    def _build_layout(self):
        """Create and arrange all visible GUI widgets.

        The left side contains all controls and the right side contains the
        Matplotlib figure canvas. The scientific plot area is deliberately not
        recoloured by the GUI palette.
        """
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        # Two-column left panel without scrolling.
        self.left = tk.Frame(main, width=760, bg=APP_BG)
        self.left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        self.left.pack_propagate(False)

        self.right = ttk.Frame(main)
        self.right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(0, 4), pady=4)

        # Preview spans both left-panel columns.
        self.preview_canvas = tk.Canvas(
            self.left,
            height=92,
            bg="white",
            highlightthickness=1,
            highlightbackground=ACCENT_BG
        )
        self.preview_canvas.pack(fill=tk.X, padx=10, pady=(8, 4))

        title = tk.Label(self.left, text="Aperiodic filter response", bg=APP_BG, font=("Arial", 11, "bold"))
        title.pack(fill=tk.X, padx=10, pady=(0, 6))

        controls = tk.Frame(self.left, bg=APP_BG)
        controls.pack(fill=tk.X, padx=10, pady=0)
        controls.columnconfigure(0, weight=1, uniform="control_cols")
        controls.columnconfigure(1, weight=1, uniform="control_cols")

        # ---------------- Pattern panel: row 0-1, column 0 ----------------
        pattern_frame = tk.LabelFrame(controls, text="Pattern", bg=PANEL_BG, fg="black", padx=7, pady=7)
        pattern_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 5), pady=4)

        tk.Label(
            pattern_frame,
            text="Select or run pattern",
            bg=PANEL_BG,
            font=("Arial", 9, "bold")
        ).pack(anchor="w", padx=3, pady=(0, 5))

        # Pattern list area.
        # It shows up to 4 patterns at once. When more patterns are available,
        # the list becomes vertically scrollable.
        self.pattern_list_canvas = tk.Canvas(
            pattern_frame,
            height=132,
            bg=PANEL_BG,
            highlightthickness=0
        )
        self.pattern_list_scrollbar = ttk.Scrollbar(
            pattern_frame,
            orient=tk.VERTICAL,
            command=self.pattern_list_canvas.yview
        )
        self.pattern_list_canvas.configure(yscrollcommand=self.pattern_list_scrollbar.set)

        self.pattern_list_canvas.pack(side=tk.TOP, left=None, fill=tk.X, expand=False, padx=2, pady=2)
        self.pattern_list_scrollbar.place(in_=self.pattern_list_canvas, relx=1.0, rely=0, relheight=1.0, anchor="ne")

        self.pattern_button_frame = tk.Frame(self.pattern_list_canvas, bg=PANEL_BG)
        self.pattern_list_window = self.pattern_list_canvas.create_window(
            (0, 0),
            window=self.pattern_button_frame,
            anchor="nw"
        )

        def _configure_pattern_list(event=None):
            self.pattern_list_canvas.configure(scrollregion=self.pattern_list_canvas.bbox("all"))
            self.pattern_list_canvas.itemconfigure(
                self.pattern_list_window,
                width=max(1, self.pattern_list_canvas.winfo_width() - 18)
            )

        self.pattern_button_frame.bind("<Configure>", _configure_pattern_list)
        self.pattern_list_canvas.bind("<Configure>", _configure_pattern_list)
        self.pattern_list_canvas.bind(
            "<MouseWheel>",
            lambda event: self.pattern_list_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        )
        self.pattern_list_canvas.bind("<Button-4>", lambda event: self.pattern_list_canvas.yview_scroll(-1, "units"))
        self.pattern_list_canvas.bind("<Button-5>", lambda event: self.pattern_list_canvas.yview_scroll(1, "units"))

        self.pattern_buttons = {}

        n_row = tk.Frame(pattern_frame, bg=PANEL_BG)
        n_row.pack(fill=tk.X, padx=2, pady=(6, 2))
        tk.Label(n_row, text="Selected N", bg=PANEL_BG).pack(side=tk.LEFT)
        self.N_var = tk.StringVar()
        self.N_entry = ttk.Entry(n_row, textvariable=self.N_var, width=12, state="readonly", justify="center")
        self.N_entry.pack(side=tk.RIGHT)
        self.N_entry.bind("<Button-1>", self._show_locked_N_message)
        self.N_entry.bind("<Key>", self._show_locked_N_message)

        self.show_btn = ttk.Button(pattern_frame, text="Run selected pattern", command=self._start_analysis)
        self.show_btn.pack(fill=tk.X, padx=2, pady=(7, 3))

        template_label = tk.Label(
            pattern_frame,
            text="To add a new pattern, click the button below to get a pattern template.",
            bg=PANEL_BG,
            fg=TEXT_DARK,
            justify="left",
            wraplength=285,
            font=("Arial", 8)
        )
        template_label.pack(fill=tk.X, padx=2, pady=(5, 1))

        ttk.Button(
            pattern_frame,
            text="Get pattern template",
            command=self._generate_pattern_template
        ).pack(fill=tk.X, padx=2, pady=(2, 4))

        file_row = tk.Frame(pattern_frame, bg=PANEL_BG)
        file_row.pack(fill=tk.X, padx=2, pady=(4, 2))
        ttk.Button(file_row, text="Add TXT pattern", command=self._add_custom_pattern_from_txt).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(file_row, text="Delete selected", command=self._delete_selected_custom_pattern).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        self.custom_hint_var = tk.StringVar(value="Custom TXT patterns: 0/5")
        tk.Label(
            pattern_frame,
            textvariable=self.custom_hint_var,
            bg=PANEL_BG,
            fg=TEXT_DARK,
            justify="left",
            wraplength=285,
            font=("Arial", 8)
        ).pack(fill=tk.X, padx=2, pady=(4, 0))

        # ---------------- Grating parameters: row 0, column 1 ----------------
        grating_frame = tk.LabelFrame(controls, text="Grating parameters", bg=CONTROL_BG, fg="black", padx=7, pady=7)
        grating_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=4)

        self.n1_var = tk.StringVar()
        self.n2_var = tk.StringVar()
        self.n0_var = tk.StringVar()
        self.ns_var = tk.StringVar()
        self.gain_var = tk.StringVar()
        self.lambda0_var = tk.StringVar()

        self.default_labels = {}
        self.default_labels["n1"] = self._add_param_row(grating_frame, "n1", self.n1_var, "", 0)
        self.default_labels["n2"] = self._add_param_row(grating_frame, "n2", self.n2_var, "", 1)
        self.default_labels["n0"] = self._add_param_row(grating_frame, "n0", self.n0_var, "", 2)
        self.default_labels["ns"] = self._add_param_row(grating_frame, "ns", self.ns_var, "", 3)
        self.default_labels["gain"] = self._add_param_row(grating_frame, "Gain", self.gain_var, GUI_DEFAULT_LABELS["gain"], 4)
        self.default_labels["lambda0_nm"] = self._add_param_row(grating_frame, "Lambda 0", self.lambda0_var, GUI_DEFAULT_LABELS["lambda0_nm"], 5, unit="nm")

        restore_btn = ttk.Button(grating_frame, text="Restore paper defaults", command=self._restore_defaults)
        restore_btn.grid(row=6, column=0, columnspan=4, sticky="ew", padx=3, pady=(6, 1))

        # ---------------- Plot parameters: row 1, column 1 ----------------
        param_frame = tk.LabelFrame(controls, text="Plot parameters", bg=CONTROL_BG, fg="black", padx=7, pady=7)
        param_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=4)

        self.start_var = tk.StringVar(value="1400")
        self.end_var = tk.StringVar(value="1700")
        self.xunit_var = tk.StringVar(value="nm")

        self._add_labeled_entry(param_frame, "Starting", self.start_var, row=0, unit="nm")
        self._add_labeled_entry(param_frame, "Ending wavelength", self.end_var, row=1, unit="nm")

        tk.Label(param_frame, text="X-axis unit", bg=CONTROL_BG).grid(row=2, column=0, sticky="e", padx=3, pady=3)
        xunit_frame = tk.Frame(param_frame, bg=CONTROL_BG)
        xunit_frame.grid(row=2, column=1, columnspan=2, sticky="w", padx=3, pady=3)
        ttk.Radiobutton(xunit_frame, text="nm", value="nm", variable=self.xunit_var, command=self._redraw_current).pack(side=tk.LEFT)
        ttk.Radiobutton(xunit_frame, text="f_i/f_B", value="norm", variable=self.xunit_var, command=self._redraw_current).pack(side=tk.LEFT)

        # ---------------- Panel 1 scale: row 2, spans both columns ----------------
        self.panel1_auto_var = tk.BooleanVar(value=True)
        self.left_min_var = tk.DoubleVar(value=0.0)
        self.left_max_var = tk.DoubleVar(value=25.0)
        self.right_min_var = tk.DoubleVar(value=0.0)
        self.right_max_var = tk.DoubleVar(value=8.0)
        self.left_min_text = tk.StringVar(value="0.00")
        self.left_max_text = tk.StringVar(value="25.00")
        self.right_min_text = tk.StringVar(value="0.00")
        self.right_max_text = tk.StringVar(value="8.00")

        y_frame = tk.LabelFrame(controls, text="Panel 1 vertical scale", bg=CONTROL_BG, fg="black", padx=7, pady=7)
        y_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=0, pady=4)

        auto_frame = tk.Frame(y_frame, bg=CONTROL_BG)
        auto_frame.pack(fill=tk.X, pady=(0, 4))
        ttk.Checkbutton(
            auto_frame,
            text="Use auto scale",
            variable=self.panel1_auto_var,
            command=self._on_scale_mode_changed
        ).pack(side=tk.LEFT)
        ttk.Button(auto_frame, text="Apply sliders", command=self._apply_slider_scale).pack(side=tk.RIGHT)
        ttk.Button(auto_frame, text="Reset auto", command=self._reset_panel1_auto).pack(side=tk.RIGHT, padx=(0, 5))

        slider_grid = tk.Frame(y_frame, bg=CONTROL_BG)
        slider_grid.pack(fill=tk.X)
        slider_grid.columnconfigure(0, weight=1)

        # One slider per row. This gives each bar the full panel width and
        # makes fine adjustment easier than the two-column layout.
        self.left_min_scale = self._create_scale_row(slider_grid, "Left y min", self.left_min_var, self.left_min_text, row=0, column=0)
        self.left_max_scale = self._create_scale_row(slider_grid, "Left y max", self.left_max_var, self.left_max_text, row=1, column=0)
        self.right_min_scale = self._create_scale_row(slider_grid, "Right y min", self.right_min_var, self.right_min_text, row=2, column=0)
        self.right_max_scale = self._create_scale_row(slider_grid, "Right y max", self.right_max_var, self.right_max_text, row=3, column=0)

        progress_frame = tk.Frame(self.left, bg=APP_BG)
        progress_frame.pack(fill=tk.X, padx=10, pady=(4, 2))
        self.progress_value = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(progress_frame, mode="determinate", maximum=100.0, variable=self.progress_value)
        self.progress.pack(fill=tk.X)
        self.progress_var = tk.StringVar(value="")
        progress_label = tk.Label(progress_frame, textvariable=self.progress_var, bg=APP_BG, fg=TEXT_DARK, justify="left", wraplength=730)
        progress_label.pack(fill=tk.X, pady=(3, 0))

        control_frame = tk.Frame(self.left, bg=APP_BG)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(control_frame, text="Save Figure", command=self._save_current_figure).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(control_frame, text="Clear", command=self._draw_empty_figure).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        self.status_var = tk.StringVar(value="Ready.")
        status = tk.Label(self.left, textvariable=self.status_var, bg=APP_BG, fg=TEXT_DARK, wraplength=730, justify="left")
        status.pack(fill=tk.X, padx=10, pady=(2, 4))

        cpu_note = tk.Label(
            self.left,
            text=(
                "Performance note: This program is strongly affected by multi-core CPU performance. "
                f"Detected logical CPUs: {self.logical_cpu_count}; worker processes used: {self.worker_count}. "
                "Two logical threads are reserved for GUI/system responsiveness. "
                "A CPU equivalent to Intel i7-9700 or AMD Ryzen 7 2700, or stronger, is recommended."
            ),
            bg=APP_BG,
            fg=TEXT_DARK,
            justify="left",
            wraplength=730,
            font=("Arial", 8)
        )
        cpu_note.pack(fill=tk.X, padx=10, pady=(0, 4))

        footer = tk.Frame(self.left, bg=FOOTER_BG)
        footer.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=6)

        self.footer_logo_image = tk.PhotoImage(data=FOOTER_LOGO_BASE64)
        logo_label = tk.Label(footer, image=self.footer_logo_image, bg=FOOTER_BG)
        logo_label.pack(side=tk.LEFT, padx=(8, 8), pady=4)

        footer_text = tk.Label(
            footer,
            text="Zhe Shang\n@ Whole Fat Milk",
            bg=FOOTER_BG,
            fg=TEXT_LIGHT,
            font=("Arial", 10, "bold"),
            justify="left"
        )
        footer_text.pack(side=tk.LEFT, padx=(0, 12), pady=4)

        supervisor_text = tk.Label(
            footer,
            text="University of Manchester\nSupervised by Dr. S. Chakraborty",
            bg=FOOTER_BG,
            fg=TEXT_LIGHT,
            font=("Arial", 8, "bold"),
            justify="right"
        )
        supervisor_text.pack(side=tk.RIGHT, padx=(8, 10), pady=4)

        self.figure = plt.Figure(figsize=(10.4, 8.3), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.right)
        self.toolbar.update()
        self._refresh_pattern_buttons()

    def _add_param_row(self, parent, label, variable, default_text, row, unit=""):
        """Add one editable grating-parameter row to the GUI."""
        tk.Label(parent, text=label, bg=CONTROL_BG).grid(row=row, column=0, sticky="e", padx=3, pady=3)
        ttk.Entry(parent, textvariable=variable, width=11, justify="center").grid(row=row, column=1, sticky="w", padx=3, pady=3)
        if unit:
            tk.Label(parent, text=unit, bg=CONTROL_BG).grid(row=row, column=2, sticky="w", padx=(0, 8), pady=3)
            default_label = tk.Label(parent, text=f"default: {default_text}", bg=CONTROL_BG, fg=TEXT_DARK)
            default_label.grid(row=row, column=3, sticky="w", padx=3, pady=3)
        else:
            default_label = tk.Label(parent, text=f"default: {default_text}", bg=CONTROL_BG, fg=TEXT_DARK)
            default_label.grid(row=row, column=2, columnspan=2, sticky="w", padx=3, pady=3)
        return default_label

    def _create_scale_row(self, parent, label_text, variable, text_var, row=None, column=None):
        """Create one horizontal slider row for panel-1 y-axis control."""
        row_frame = tk.Frame(parent, bg=CONTROL_BG)
        if row is None or column is None:
            row_frame.pack(fill=tk.X, pady=2)
        else:
            row_frame.grid(row=row, column=column, sticky="ew", padx=4, pady=2)

        tk.Label(row_frame, text=label_text, bg=CONTROL_BG, width=10, anchor="w").pack(side=tk.LEFT)
        scale = tk.Scale(
            row_frame,
            variable=variable,
            from_=0,
            to=10,
            orient=tk.HORIZONTAL,
            resolution=0.01,
            showvalue=False,
            bg=CONTROL_BG,
            highlightthickness=0,
            length=470,
            command=lambda _=None: self._update_slider_labels()
        )
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 6))
        scale.bind("<ButtonRelease-1>", self._on_slider_release)
        tk.Label(row_frame, textvariable=text_var, bg=CONTROL_BG, width=7, anchor="e").pack(side=tk.RIGHT)
        return scale

    def _add_labeled_entry(self, parent, label, variable, row, unit=""):
        """Add one labelled numeric entry box to a parameter frame."""
        tk.Label(parent, text=label, bg=CONTROL_BG).grid(row=row, column=0, sticky="e", padx=3, pady=3)
        ttk.Entry(parent, textvariable=variable, width=16, justify="center").grid(row=row, column=1, sticky="w", padx=3, pady=3)
        if unit:
            tk.Label(parent, text=unit, bg=CONTROL_BG).grid(row=row, column=2, sticky="w", padx=3, pady=3)

    def _current_default_params(self):
        idx = self._selected_pattern_idx() if hasattr(self, "pattern_combo") else self.current_pattern_idx
        pattern = PATTERN_CONFIGS[idx]["pattern"]
        return get_original_default_params(pattern)

    def _update_default_label_values(self):
        if not hasattr(self, "default_labels"):
            return
        defaults = self._current_default_params()
        self.default_labels["n1"].config(text=f"paper default: {format_param_value(defaults['n1'], 6)}")
        self.default_labels["n2"].config(text=f"paper default: {format_param_value(defaults['n2'], 6)}")
        self.default_labels["n0"].config(text=f"paper default: {format_param_value(defaults['n0'], 6)}")
        self.default_labels["ns"].config(text=f"paper default: {format_param_value(defaults['ns'], 6)}")
        self.default_labels["gain"].config(text=f"paper default: {format_param_value(defaults['gain'], 3)}")
        self.default_labels["lambda0_nm"].config(text=f"paper default: {format_param_value(defaults['lambda0_nm'], 3)}")

    def _restore_defaults(self, silent=False):
        """Restore the paper-consistent default parameters for the selected pattern."""
        defaults = self._current_default_params()
        self.n1_var.set(format_param_value(defaults["n1"], 9))
        self.n2_var.set(format_param_value(defaults["n2"], 9))
        self.n0_var.set(format_param_value(defaults["n0"], 9))
        self.ns_var.set(format_param_value(defaults["ns"], 9))
        self.gain_var.set(format_param_value(defaults["gain"], 3))
        self.lambda0_var.set(format_param_value(defaults["lambda0_nm"], 3))
        self._update_default_label_values()
        if not silent:
            self.status_var.set("Grating parameters were restored to the original V1/V2 default values for the selected pattern.")

    def _selected_pattern_idx(self):
        return self.current_pattern_idx

    def _refresh_pattern_buttons(self):
        """Rebuild the scrollable pattern-button list.

        The list includes both the built-in patterns and up to five user-loaded
        TXT patterns.
        """
        if not hasattr(self, "pattern_button_frame"):
            return

        for child in self.pattern_button_frame.winfo_children():
            child.destroy()
        self.pattern_buttons = {}

        pattern_ids = sorted(PATTERN_CONFIGS.keys())
        for idx in pattern_ids:
            cfg = PATTERN_CONFIGS[idx]
            is_selected = idx == self.current_pattern_idx
            is_custom = idx in self.custom_pattern_ids
            label = cfg["short_name"]
            prefix = "Custom" if is_custom else f"Pattern {idx}"
            btn_text = f"{prefix}: {label}"

            btn = tk.Button(
                self.pattern_button_frame,
                text=btn_text,
                command=lambda i=idx: self._select_and_start_pattern(i),
                anchor="w",
                relief=tk.SUNKEN if is_selected else tk.RAISED,
                bg=SELECTED_BG if is_selected else "#f2f2f2",
                activebackground=BUTTON_ACTIVE_BG
            )
            btn.pack(fill=tk.X, padx=2, pady=2, ipady=2)
            self.pattern_buttons[idx] = btn

        if hasattr(self, "custom_hint_var"):
            self.custom_hint_var.set(f"Custom TXT patterns: {len(self.custom_pattern_ids)}/{self.max_custom_patterns}")

        if hasattr(self, "pattern_list_canvas"):
            self.pattern_list_canvas.update_idletasks()
            self.pattern_list_canvas.configure(scrollregion=self.pattern_list_canvas.bbox("all"))

            # Keep the selected pattern visible after adding/selecting a new one.
            pattern_ids = sorted(PATTERN_CONFIGS.keys())
            if self.current_pattern_idx in pattern_ids and len(pattern_ids) > 4:
                selected_pos = pattern_ids.index(self.current_pattern_idx)
                max_first = max(1, len(pattern_ids) - 4)
                first_visible = min(max(0, selected_pos - 1), max_first)
                self.pattern_list_canvas.yview_moveto(first_visible / max_first)

    def _set_pattern_buttons_state(self, state):
        if not hasattr(self, "pattern_buttons"):
            return
        for btn in self.pattern_buttons.values():
            btn.configure(state=state)
        if hasattr(self, "show_btn"):
            self.show_btn.configure(state=state)

    def _select_pattern(self, idx):
        if idx not in PATTERN_CONFIGS:
            return

        self.current_pattern_idx = idx
        self._update_pattern_fields()
        self._restore_defaults(silent=True)
        self._draw_pattern_preview()
        self.progress_var.set("")
        self.panel1_scale_cache_key = None
        self._refresh_pattern_buttons()

        cache_key = self._get_cache_key(show_message=False)
        if cache_key is not None and cache_key in self.result_cache:
            self.current_result = self.result_cache[cache_key]
            self.current_cache_key = cache_key
            self._plot_result(self.current_result)
        else:
            self.current_result = None
            self.current_cache_key = None
            self._draw_empty_figure()

    def _select_and_start_pattern(self, idx):
        self._select_pattern(idx)
        self._start_analysis()

    def _on_pattern_changed(self, event=None):
        self._select_pattern(self.current_pattern_idx)

    def _format_pattern_template(self, lambda_count):
        """Create the text content for a custom-pattern template file."""
        values = [0] * lambda_count
        lines = []
        for i in range(0, len(values), 10):
            chunk = values[i:i + 10]
            line = "    " + ", ".join(str(v) for v in chunk)
            if i + 10 < len(values):
                line += ","
            lines.append(line)

        pattern_body = "\n".join(lines)
        return (
            "# Custom pattern template generated by TMM_GUI_v1\n"
            "#\n"
            "# Pattern coding rule:\n"
            "# Each array entry represents one standard brick.\n"
            "# 0 means LH.\n"
            "# 1 means LHH.\n"
            "# 2 means LHHH.\n"
            "# In general, the integer k means that k extra H-type defect sections\n"
            "# are added after the standard LH brick.\n"
            "#\n"
            "# Important length rule:\n"
            "# The lambda value below is the target total design length and should not\n"
            "# be manually increased just because you add defect H sections.\n"
            "# When two extra H sections are added in total, the actual number of array\n"
            "# entries should be reduced by one to keep the total lambda length unchanged.\n"
            "# Example: for a 50-lambda design, if the sum of all integers in the array\n"
            "# is 10, the array should contain 50 - 10/2 = 45 entries.\n"
            "#\n"
            "# Replace the zeros in pattern = [...] with your integer defect sequence.\n"
            "# Keep the format: pattern = [...] and lambda = <positive integer>.\n\n"
            "pattern = [\n"
            f"{pattern_body}\n"
            "]\n\n"
            f"lambda = {lambda_count}\n"
        )

    def _generate_pattern_template(self):
        """Ask the user for lambda count and generate a TXT pattern template."""
        lambda_count = simpledialog.askinteger(
            "Pattern template",
            "Enter the lambda number for the new pattern:",
            minvalue=1,
            maxvalue=10000,
            parent=self.root
        )
        if lambda_count is None:
            return

        output_dir = Path(__file__).resolve().parent
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"pattern_template_{lambda_count}lambda_{timestamp}.txt"
        output_path = output_dir / base_name

        counter = 1
        while output_path.exists():
            output_path = output_dir / f"pattern_template_{lambda_count}lambda_{timestamp}_{counter}.txt"
            counter += 1

        output_path.write_text(self._format_pattern_template(lambda_count), encoding="utf-8")

        messagebox.showinfo(
            "Pattern template generated",
            f"A pattern template has been generated. Please find the TXT file in the same directory as the code.\n\n"
            f"File name: {output_path.name}"
        )
        self.status_var.set(f"Generated pattern template: {output_path.name}")

    def _parse_pattern_txt(self, filepath):
        """Read and validate a user-supplied TXT pattern file.

        Accepted format:
            pattern = [0, 0, 2, 0, ...]
            lambda = 50

        Lines beginning with # are ignored. Trailing comments are also removed
        before parsing.
        """
        raw_text = Path(filepath).read_text(encoding="utf-8")

        # Remove full-line and trailing comments. This prevents example notes such as
        # "pattern = [...]" in commented instructions from being parsed as real data.
        cleaned_lines = []
        for line in raw_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            cleaned_lines.append(line.split("#", 1)[0])
        text = "\n".join(cleaned_lines)

        match = re.search(r"pattern\s*=\s*\[(.*?)\]", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError(
                "No valid 'pattern = [...]' block was found. "
                "Please delete the instruction/comment lines after '#' in the example TXT file and keep only the real pattern data."
            )

        numbers = re.findall(r"[-+]?\d+", match.group(1))
        if not numbers:
            raise ValueError(
                "The pattern block does not contain any integer entries. "
                "Please delete the instruction/comment lines after '#' in the example TXT file and keep only the real pattern data."
            )

        pattern = [int(x) for x in numbers]
        if any(x < 0 for x in pattern):
            raise ValueError("Pattern entries must be non-negative integers.")

        lambda_match = re.search(r"(?:lambda|Lambda|N)\s*=\s*([0-9]+)", text)
        if lambda_match:
            design_N = int(lambda_match.group(1))
        else:
            design_N = len(pattern)

        if design_N <= 0:
            raise ValueError("Lambda/N must be a positive integer.")

        return pattern, design_N

    def _add_custom_pattern_from_txt(self):
        """Load one user-defined pattern from TXT and add it to the pattern list."""
        if len(self.custom_pattern_ids) >= self.max_custom_patterns:
            messagebox.showerror("Custom pattern limit reached", "You can add at most five custom TXT patterns.")
            return

        filepath = filedialog.askopenfilename(
            title="Select pattern TXT file",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            pattern, design_N = self._parse_pattern_txt(filepath)
        except Exception as exc:
            messagebox.showerror(
                "Invalid TXT pattern",
                f"{exc}\n\nIf you used the example TXT file, please delete the instruction/comment lines after '#', "
                "and keep only the actual pattern = [...] and lambda = ... data."
            )
            return

        used_ids = set(PATTERN_CONFIGS.keys())
        new_id = next(i for i in range(6, 11) if i not in used_ids)

        name = Path(filepath).stem
        PATTERN_CONFIGS[new_id] = {
            "name": f"Custom - {name}",
            "short_name": name,
            "pattern": pattern,
            "design_N": design_N,
            "is_custom": True,
            "source_file": str(filepath),
        }
        self.custom_pattern_ids.append(new_id)

        self.status_var.set(f"Added custom pattern '{name}' with {len(pattern)} entries and design N={design_N}.")
        self._select_pattern(new_id)

    def _delete_selected_custom_pattern(self):
        """Delete the currently selected custom pattern, if it is user-added."""
        idx = self.current_pattern_idx
        if idx not in self.custom_pattern_ids:
            messagebox.showinfo("Cannot delete pattern", "Only user-added custom TXT patterns can be deleted.")
            return

        name = PATTERN_CONFIGS[idx]["short_name"]
        del PATTERN_CONFIGS[idx]
        self.custom_pattern_ids.remove(idx)

        # Clear cached results associated with the deleted custom pattern.
        for key in list(self.result_cache.keys()):
            if isinstance(key, tuple) and key and key[0] == idx:
                del self.result_cache[key]

        self.current_pattern_idx = 1
        self.current_result = None
        self.current_cache_key = None
        self.status_var.set(f"Deleted custom pattern '{name}'.")
        self._select_pattern(1)

    def _update_pattern_fields(self):
        idx = self.current_pattern_idx
        self.N_var.set(str(PATTERN_CONFIGS[idx]["design_N"]))

    def _show_locked_N_message(self, event=None):
        messagebox.showinfo(
            "Locked design parameter",
            "This pattern design value cannot be changed."
        )
        return "break"

    def _validate_range(self):
        """Validate the wavelength display range entered by the user."""
        try:
            start_nm = float(self.start_var.get())
            end_nm = float(self.end_var.get())
        except ValueError:
            messagebox.showerror("Invalid wavelength range", "Starting and ending wavelengths must be numeric values.")
            return None

        if start_nm < WL_MIN_NM or end_nm > WL_MAX_NM or start_nm >= end_nm:
            messagebox.showerror(
                "Range limit exceeded",
                "The selected wavelength range exceeds the maximum allowed limit of 1400-1700 nm."
            )
            return None

        return start_nm, end_nm

    def _validate_grating_params(self, show_message=True):
        """Validate n1, n2, n0, ns, gain, and Lambda 0 from the GUI."""
        try:
            n1 = float(self.n1_var.get())
            n2 = float(self.n2_var.get())
            n0 = float(self.n0_var.get())
            ns = float(self.ns_var.get())
            gain = float(self.gain_var.get())
            lambda0_nm = float(self.lambda0_var.get())
        except ValueError:
            if show_message:
                messagebox.showerror("Invalid grating parameters", "All grating parameters must be numeric values.")
            return None

        if n1 <= 0 or n2 <= 0 or n0 <= 0 or ns <= 0:
            if show_message:
                messagebox.showerror("Invalid grating parameters", "n1, n2, n0, and ns must be positive.")
            return None

        if n1 <= n2:
            if show_message:
                messagebox.showerror("Invalid grating parameters", "n1 must be greater than n2.")
            return None

        if gain < 0.0 or gain > 6.0:
            if show_message:
                messagebox.showerror("Invalid gain", "Gain must be between 0 and 6.")
            return None

        if lambda0_nm <= 0:
            if show_message:
                messagebox.showerror("Invalid Lambda 0", "Lambda 0 must be positive.")
            return None

        return {
            "n1": n1,
            "n2": n2,
            "n0": n0,
            "ns": ns,
            "gain": gain,
            "lambda0_nm": lambda0_nm,
        }

    def _get_cache_key(self, show_message=True):
        params = self._validate_grating_params(show_message=show_message)
        if params is None:
            return None
        idx = self._selected_pattern_idx()
        key = (
            idx,
            round(params["n1"], 6),
            round(params["n2"], 6),
            round(params["n0"], 6),
            round(params["ns"], 6),
            round(params["gain"], 6),
            round(params["lambda0_nm"], 6),
        )
        return key

    def _display_lambda0_nm(self, result=None):
        if result is not None and "params" in result:
            return result["params"]["lambda0_nm"]
        params = self._validate_grating_params(show_message=False)
        if params is not None:
            return params["lambda0_nm"]
        return 1550.0

    def _get_axis_arrays(self, result):
        """Return the x-axis array, ordering, limits, and label for plotting."""
        freq_norm = result["freq_norm"]
        validated = self._validate_range()
        if validated is None:
            return None
        start_nm, end_nm = validated
        display_lambda0_nm = self._display_lambda0_nm(result)

        if self.xunit_var.get() == "nm":
            x = display_lambda0_nm / freq_norm
            order = np.argsort(x)
            x_sorted = x[order]
            xlim = (start_nm, end_nm)
            xlabel = "Wavelength (nm)"
            return x_sorted, order, xlim, xlabel

        x = freq_norm
        order = np.arange(len(x))
        xlim = (display_lambda0_nm / end_nm, display_lambda0_nm / start_nm)
        xlabel = r"Normalized frequency $f_i/f_B$"
        return x, order, xlim, xlabel

    def _on_scale_mode_changed(self):
        self._set_slider_state()
        if self.current_result is None:
            return
        if self.panel1_auto_var.get():
            self._update_panel1_scale_ranges(self.current_result, reset_values=True)
        self._plot_result(self.current_result)

    def _set_slider_state(self):
        state = tk.DISABLED if self.panel1_auto_var.get() else tk.NORMAL
        for widget in [self.left_min_scale, self.left_max_scale, self.right_min_scale, self.right_max_scale]:
            widget.configure(state=state)

    def _update_slider_labels(self):
        self.left_min_text.set(f"{self.left_min_var.get():.2f}")
        self.left_max_text.set(f"{self.left_max_var.get():.2f}")
        self.right_min_text.set(f"{self.right_min_var.get():.2f}")
        self.right_max_text.set(f"{self.right_max_var.get():.2f}")

    def _on_slider_release(self, event=None):
        self._update_slider_labels()
        if not self.panel1_auto_var.get() and self.current_result is not None:
            self._plot_result(self.current_result)

    def _apply_slider_scale(self):
        if self.current_result is not None:
            self.panel1_auto_var.set(False)
            self._set_slider_state()
            self._plot_result(self.current_result)

    def _reset_panel1_auto(self):
        self.panel1_auto_var.set(True)
        self._set_slider_state()
        if self.current_result is not None:
            self._update_panel1_scale_ranges(self.current_result, reset_values=True)
            self._plot_result(self.current_result)

    def _compute_auto_and_slider_limits(self, values, clamp_auto_min_zero=False):
        values = np.asarray(values, dtype=float)
        data_min = float(np.nanmin(values))
        data_max = float(np.nanmax(values))
        span = max(data_max - data_min, 1e-9)

        auto_margin = 0.12 * span
        auto_min = data_min - auto_margin
        auto_max = data_max + auto_margin
        if clamp_auto_min_zero:
            auto_min = max(0.0, auto_min)

        slider_max = max(1.5 * data_max, auto_max + 0.2 * span, auto_max)
        if data_min >= 0 or clamp_auto_min_zero:
            slider_min = 0.0
        else:
            slider_min = min(1.5 * data_min, auto_min - 0.2 * span, auto_min)

        if slider_max <= slider_min:
            slider_max = slider_min + 1.0

        return auto_min, auto_max, slider_min, slider_max

    def _update_panel1_scale_ranges(self, result, reset_values=False):
        """Update slider limits and automatic y-axis values for panel 1."""
        tau_auto_min, tau_auto_max, tau_slider_min, tau_slider_max = self._compute_auto_and_slider_limits(
            result["tau_g_ps"], clamp_auto_min_zero=True
        )
        ng_auto_min, ng_auto_max, ng_slider_min, ng_slider_max = self._compute_auto_and_slider_limits(
            result["n_g"], clamp_auto_min_zero=False
        )

        tau_res = max((tau_slider_max - tau_slider_min) / 600.0, 0.01)
        ng_res = max((ng_slider_max - ng_slider_min) / 600.0, 0.01)

        self.left_min_scale.configure(from_=tau_slider_min, to=tau_slider_max, resolution=tau_res)
        self.left_max_scale.configure(from_=tau_slider_min, to=tau_slider_max, resolution=tau_res)
        self.right_min_scale.configure(from_=ng_slider_min, to=ng_slider_max, resolution=ng_res)
        self.right_max_scale.configure(from_=ng_slider_min, to=ng_slider_max, resolution=ng_res)

        if reset_values or self.panel1_auto_var.get():
            self.left_min_var.set(tau_auto_min)
            self.left_max_var.set(tau_auto_max * 1.2)
            self.right_min_var.set(ng_auto_min)
            self.right_max_var.set(ng_auto_max)
            self._update_slider_labels()

        self._set_slider_state()

    def _get_panel1_limits(self):
        # Auto scale and manual scale both use the same slider variables.
        # When auto scale is enabled, _update_panel1_scale_ranges() refreshes
        # these variables from the current data first. This avoids Matplotlib's
        # default autoscale, which can make f-labels overlap the panel boundary.
        lmin = self.left_min_var.get()
        lmax = self.left_max_var.get()
        rmin = self.right_min_var.get()
        rmax = self.right_max_var.get()

        if lmin >= lmax or rmin >= rmax:
            messagebox.showerror("Invalid scale", "Each y-axis minimum must be smaller than its maximum.")
            return "error"

        return (lmin, lmax), (rmin, rmax)

    def _draw_pattern_preview(self):
        """Draw a simplified grating preview in the top-left canvas."""
        self.preview_canvas.delete("all")
        idx = self.current_pattern_idx
        pattern = PATTERN_CONFIGS[idx]["pattern"]
        name = PATTERN_CONFIGS[idx]["short_name"]

        w = max(self.preview_canvas.winfo_width(), 360)
        margin = 8
        y0 = 24
        y1 = 72

        self.preview_canvas.create_text(w / 2, 11, text=name, font=("Arial", 9, "bold"), fill="black")

        total_units = sum(2 + max(0, k) for k in pattern)
        unit_w = (w - 2 * margin) / max(total_units, 1)
        x = margin

        for k in pattern:
            self.preview_canvas.create_rectangle(x, y0, x + unit_w, y1, fill="#2727cc", outline="#2727cc")
            x += unit_w
            width_n1 = unit_w * (1 + max(0, k))
            self.preview_canvas.create_rectangle(x, y0, x + width_n1, y1, fill="#ffff00", outline="#ffff00")
            x += width_n1

        self.preview_canvas.create_rectangle(margin, y0, w - margin, y1, outline="#444444")

    def _draw_empty_figure(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.60, "Select a pattern and click Show Pattern", ha="center", va="center", fontsize=14)
        ax.text(0.5, 0.50, "For the best viewing experience, please maximize this window before running the simulation.", ha="center", va="center", fontsize=11)
        ax.text(0.5, 0.42, "The N value is locked to the selected pattern design.", ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        self.current_result = None
        self.current_cache_key = None
        self.panel1_scale_cache_key = None
        self._stop_progress_animation()
        self.canvas.draw_idle()
        self.status_var.set("Ready.")

    def _estimate_runtime(self, idx):
        design_n = PATTERN_CONFIGS[idx]["design_N"]
        base = 14.0 * (design_n / 50.0) ** 1.15
        if self.calc_history:
            recent = np.mean(self.calc_history[-3:])
            scale = (design_n / 50.0) ** 1.10
            return max(4.0, recent * scale)
        return max(4.0, base)

    def _start_progress_animation(self, idx):
        self._stop_progress_animation()
        self.progress_start_time = time.time()
        self.progress_estimate_seconds = self._estimate_runtime(idx)
        self.progress_value.set(0.0)
        self.progress.configure(mode="determinate", maximum=100.0)
        self._update_progress_animation()

    def _update_progress_animation(self):
        if not self.is_running:
            return

        elapsed = time.time() - self.progress_start_time if self.progress_start_time is not None else 0.0
        estimate = max(self.progress_estimate_seconds or 1.0, 1.0)
        eta = max(0.0, estimate - elapsed)

        # This is an estimated progress indicator. The heavy TMM calculation is
        # executed inside multiprocessing workers, so the GUI cannot know the
        # exact completed task count without adding expensive inter-process
        # communication. The bar therefore advances according to the estimated
        # runtime and stays below 95% until the calculation actually finishes.
        progress_percent = min(95.0, (elapsed / estimate) * 100.0)
        self.progress_value.set(progress_percent)

        self.progress_var.set(
            f"Processing pattern...  Progress: {progress_percent:5.1f}%   "
            f"Workers: {self.worker_count}/{self.logical_cpu_count} logical CPUs   "
            f"Elapsed: {elapsed:4.1f} s   Estimated remaining: {eta:4.1f} s"
        )
        self.progress_after_id = self.root.after(250, self._update_progress_animation)

    def _finish_progress_animation(self):
        if self.progress_after_id is not None:
            self.root.after_cancel(self.progress_after_id)
            self.progress_after_id = None
        self.progress_value.set(100.0)
        self.progress_var.set("Processing complete.  Progress: 100.0%")

    def _stop_progress_animation(self):
        if self.progress_after_id is not None:
            self.root.after_cancel(self.progress_after_id)
            self.progress_after_id = None
        if hasattr(self, "progress_value"):
            self.progress_value.set(0.0)
        self.progress_var.set("")
        self.progress_start_time = None
        self.progress_estimate_seconds = None

    def _start_analysis(self):
        """Validate inputs, check cache, and start simulation in a background thread."""
        if self.is_running:
            messagebox.showinfo("Analysis is running", "Please wait until the current calculation is finished.")
            return

        if self._validate_range() is None:
            return

        params = self._validate_grating_params(show_message=True)
        if params is None:
            return

        idx = self._selected_pattern_idx()
        self.current_pattern_idx = idx
        cache_key = self._get_cache_key(show_message=False)
        if cache_key is None:
            return

        if cache_key in self.result_cache:
            self.current_result = self.result_cache[cache_key]
            self.current_cache_key = cache_key
            self._plot_result(self.current_result)
            self.status_var.set(f"Loaded cached result for Pattern {idx}.")
            return

        self.is_running = True
        self._set_pattern_buttons_state(tk.DISABLED)
        self.status_var.set(f"Calculating Pattern {idx} using {self.worker_count} worker processes. This may take several minutes...")
        self._start_progress_animation(idx)

        thread = threading.Thread(target=self._analysis_worker, args=(idx, params, cache_key), daemon=True)
        thread.start()

    def _analysis_worker(self, idx, params, cache_key):
        """Run the heavy numerical simulation outside the Tkinter main thread."""
        start_t = time.time()
        try:
            pattern = PATTERN_CONFIGS[idx]["pattern"]
            result = analyze_pattern(
                pattern=pattern,
                pattern_idx=idx,
                freq_norm=self.freq_norm,
                gLg_axis=self.gLg_axis,
                fp_grid=self.fp_grid,
                num_cores=self.worker_count,
                params=params,
            )
            elapsed = time.time() - start_t
            self.root.after(0, self._analysis_success, idx, result, cache_key, elapsed)
        except Exception as exc:
            error_text = traceback.format_exc()
            self.root.after(0, self._analysis_failed, str(exc), error_text)

    def _analysis_success(self, idx, result, cache_key, elapsed):
        self.result_cache[cache_key] = result
        self.current_result = result
        self.current_cache_key = cache_key
        self.is_running = False
        self._set_pattern_buttons_state(tk.NORMAL)
        self._finish_progress_animation()
        self.calc_history.append(elapsed)

        info = (
            f"Pattern {idx} finished. dn={result['dn']:.9f}, Lg={result['L_g_m'] * 1e6:.3f} um. "
            f"Elapsed time: {elapsed:.2f} s. Workers used: {self.worker_count}."
        )
        self.status_var.set(info)
        self._plot_result(result)

    def _analysis_failed(self, message, details):
        self.is_running = False
        self._set_pattern_buttons_state(tk.NORMAL)
        self._stop_progress_animation()
        self.status_var.set("Calculation failed.")
        print(details)
        messagebox.showerror("Calculation failed", message)

    def _redraw_current(self):
        if self.current_result is not None:
            self._plot_result(self.current_result)

    def _plot_result(self, result):
        """Render the six-panel analysis figure for the selected pattern."""
        axis_data = self._get_axis_arrays(result)
        if axis_data is None:
            return

        cache_key = self.current_cache_key
        if self.panel1_scale_cache_key != cache_key:
            self._update_panel1_scale_ranges(result, reset_values=True)
            self.panel1_scale_cache_key = cache_key
        else:
            self._update_panel1_scale_ranges(result, reset_values=False)

        panel1_limits = self._get_panel1_limits()
        if panel1_limits == "error":
            return

        x, order, xlim, xlabel = axis_data
        idx = result["pattern_idx"]
        pattern = PATTERN_CONFIGS[idx]["pattern"]
        display_lambda0_nm = self._display_lambda0_nm(result)

        tau_g_ps = result["tau_g_ps"][order]
        n_eff = result["n_eff"][order]
        n_g = result["n_g"][order]
        log_R = result["log_R"][:, order]
        eta_2d = result["eta_2d"][:, order]
        fp_matched = result["fp_matched"][order]
        mode_idx = result["mode_idx"]
        eta_vmax = result["eta_vmax"]
        gLg_axis = result["gLg_axis"]
        fp_grid = result["fp_grid"]

        mode_x = []
        for p in mode_idx:
            if self.xunit_var.get() == "nm":
                mode_x.append(display_lambda0_nm / result["freq_norm"][p])
            else:
                mode_x.append(result["freq_norm"][p])

        self.figure.clear()
        axes_2d = self.figure.subplots(
            6, 2,
            gridspec_kw={"width_ratios": [32, 1.35], "wspace": 0.09, "hspace": 0.34},
            sharex="col"
        )
        self.figure.subplots_adjust(left=0.085, right=0.96, top=0.90, bottom=0.08)

        axes = axes_2d[:, 0]
        caxes = axes_2d[:, 1]

        for i in [0, 1, 4, 5]:
            caxes[i].axis("off")

        self.figure.suptitle(
            f"{PATTERN_CONFIGS[idx]['short_name']} corrected analysis "
            f"(design N={PATTERN_CONFIGS[idx]['design_N']} lambda, code entries={len(pattern)}, sum(k)={sum(pattern)})",
            fontsize=11.3,
            y=0.965
        )

        axes[0].plot(x, tau_g_ps, "k", lw=1.15)
        axes[0].set_ylabel(r"$\tau_g$ (ps)")
        ax0_twin = axes[0].twinx()
        ax0_twin.plot(x, n_g, "r", lw=1.05)
        ax0_twin.set_ylabel(r"$\tilde{n}_g$", color="r")
        ax0_twin.tick_params(axis="y", colors="r")

        (lmin, lmax), (rmin, rmax) = panel1_limits
        axes[0].set_ylim(lmin, lmax)
        ax0_twin.set_ylim(rmin, rmax)

        axes[1].plot(x, n_eff, "k", lw=1.15)
        axes[1].axhline(0.5 * (result["n1_used"] + result["n2_used"]), color="gray", ls="--", lw=0.8)
        axes[1].set_ylabel(r"$\tilde{n}_{eff}$")
        off = max(np.nanmax(np.abs(n_eff - np.nanmean(n_eff))), 1e-4)
        axes[1].set_ylim(np.nanmean(n_eff) - 1.5 * off, np.nanmean(n_eff) + 1.5 * off)

        pcm_d = axes[2].pcolormesh(x, gLg_axis, log_R, shading="auto", cmap=self.cmap_R, vmin=-1, vmax=4)
        axes[2].set_ylabel(r"Gain $gL_g$")
        cb_d = self.figure.colorbar(pcm_d, cax=caxes[2], ticks=[-1, 0, 1, 2, 3, 4])
        cb_d.ax.set_yticklabels([r"$10^{-1}$", "1", r"$10$", r"$10^2$", r"$10^3$", r"$10^4$"])
        cb_d.ax.set_title(r"$R$", pad=8)
        selected_gain = float(result.get("linear_gain", 0.0))
        if gLg_axis[0] <= selected_gain <= gLg_axis[-1]:
            axes[2].axhline(selected_gain, color="black", ls="-.", lw=1.1, alpha=0.95, zorder=4)
            x_text = xlim[0] + 0.012 * (xlim[1] - xlim[0])
            y_text = selected_gain + 0.02 * (gLg_axis[-1] - gLg_axis[0])
            axes[2].text(
                x_text,
                y_text,
                rf"selected gain $gL_g$ = {selected_gain:.2f}",
                color="black",
                fontsize=8,
                va="bottom",
                ha="left",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5)
            )


        pcm_e = axes[3].pcolormesh(x, fp_grid, eta_2d, shading="auto", cmap=self.cmap_eta, vmin=0.0, vmax=eta_vmax)
        axes[3].set_ylabel(r"Pump $f_p$ (THz)")
        self.figure.colorbar(pcm_e, cax=caxes[3]).set_label(r"$\eta$ proxy")

        for i, p in enumerate(mode_idx):
            color = self.slice_colors[i % len(self.slice_colors)]
            k = np.argmin(np.abs(fp_grid - result["fp_matched"][p]))
            axes[4].plot(x, eta_2d[k, :], color=color, lw=1.20, label=rf"$f_p={fp_grid[k]:.2f}$ THz")
        axes[4].set_ylabel(r"$\eta$ proxy")
        if len(mode_idx) > 0:
            axes[4].legend(fontsize=7, ncol=2, loc="upper right")

        axes[5].plot(x, fp_matched, "k", lw=1.35)
        axes[5].set_ylabel("PM pump\nfreq. (THz)")
        axes[5].set_xlabel(xlabel)

        for i, mx in enumerate(mode_x):
            color = self.slice_colors[i % len(self.slice_colors)]
            for ax in axes:
                ax.axvline(mx, color=color, ls="--", lw=1.0, alpha=0.88, zorder=3)
            if i < 2:
                ymin0, ymax0 = axes[0].get_ylim()
                axes[0].text(mx, ymax0 - 0.08 * (ymax0 - ymin0), rf"$f_{i+1}$", color=color,
                             ha="center", va="top", fontsize=9, fontweight="bold")
            axes[3].text(mx, fp_grid[5], rf"$f_{i+1}$", color="white", ha="center", va="bottom", fontweight="bold")

        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.set_xlim(*xlim)

        for ax in axes[:5]:
            plt.setp(ax.get_xticklabels(), visible=False)

        self.canvas.draw_idle()

    def _save_current_figure(self):
        """Save the current Matplotlib figure to PNG, PDF, SVG, or another format."""
        if self.current_result is None:
            messagebox.showinfo("No figure", "There is no calculated figure to save yet.")
            return
        idx = self.current_result["pattern_idx"]
        default_name = f"Pattern_{idx}_GUI_result.png"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG image", "*.png"), ("PDF file", "*.pdf"), ("SVG file", "*.svg"), ("All files", "*.*")]
        )
        if not filepath:
            return
        self.figure.savefig(filepath, dpi=300, bbox_inches="tight")
        self.status_var.set(f"Figure saved: {os.path.basename(filepath)}")


def main():
    freeze_support()
    root = tk.Tk()
    app = TMMGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
