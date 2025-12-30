# GitHub Copilot — Project Instructions

These instructions help AI coding agents be productive in this BEM analysis project. Focus on the concrete patterns used here, not generic best practices.

## Overview
- Purpose: Blade Element Momentum (BEM) analysis and visualization for wind turbine blades.
- Core: Numerical solver in `src/BEM_functions.py`; orchestration and plotting in `src/main_BEM.py` and `src/BEM_plots.py`.
- Data: Airfoil polars from `data/DU95W180.cvs` (whitespace-separated columns: `alfa cl cd cm`).

## Architecture
- `src/BEM_functions.py`:
  - `BEM_cycle(...)`: Loops annuli, assembles results dict with spanwise arrays and (optionally) azimuthal dimension.
  - `solve_stream_tube(...)`: Iterative per-annulus solver; applies Glauert and optional Prandtl corrections; normalizes outputs at end.
  - Helpers: `compute_c_t`, `compute_axial_induction`, `prandtl_tip_root_correction`, `load_blade_element`.
- `src/main_BEM.py`:
  - Sets geometry (`chord_distribution`, `twist_distribution`), flow (`u_inf`, TSR, yaw), reads polars, toggles plots.
  - Runs corrected and uncorrected cases; aggregates CT/CP/CQ and prepares data for plots.
- `src/BEM_plots.py`:
  - Plot families: corrections (Glauert/Prandtl), non-yawed spanwise distributions, yawed polar maps, mesh convergence, pressure.
- `src/blade_optimization.py`:
  - Parameter sweep over chord/twist coefficients to maximize `CP` at `CT_ref≈0.75`; returns optimal distributions.
- `src/mesh_convergence.py`:
  - Compares uniform vs cosine radial spacing; computes CT and relative errors.

## Conventions & Patterns
- Units: SI (m, s, rad/s); angles in degrees for inputs/outputs; TSR dimensionless.
- Radial discretization: boundaries `r_over_R` (root→tip); centroids computed as midpoints.
- Results dict keys (spanwise arrays): `a`, `a_line`, `r_over_R`, `normal_force`, `tangential_force`, `gamma`, `alpha`, `inflow_angle`, `c_thrust`, `c_torque`, `c_power`, plus `*_check` variants.
- Yaw handling:
  - If `yaw_angle != 0`: provide `psi_vec`; only TSR=8 is evaluated for yawed plots in `main_BEM.py`.
  - Raising rules: yaw requires `psi_vec`; Prandtl off with yaw is disallowed.
- Normalization in `solve_stream_tube(...)` (final values):
  - `normal_force` and `tangential_force` divided by `(0.5 * u_inf^2 * R)`.
  - `gamma` divided by `(π * u_inf^2 / (B * Ω))`.
- Polars: Read via `pandas.read_csv(..., sep='\s+', names=['alfa','cl','cd','cm'])` from `data/DU95W180.cvs`.

## Developer Workflows
- Environment setup (macOS):
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```
- Run main analysis (plots toggled by flags near top of script):
  ```bash
  python src/main_BEM.py
  ```
- Mesh convergence study:
  ```bash
  python src/mesh_convergence.py
  ```
- Blade optimization (optional): set `perform_blade_optimization = True` in `src/main_BEM.py` or call `blade_optimization(...)` directly.

## Key APIs (examples)
- Non-yawed BEM call:
  ```python
  results = BEM_cycle(u_inf, r_over_R, r_root, r_tip, Omega, R, B,
                      chord_distribution, twist_distribution,
                      yaw_angle=0.0, tip_speed_ratio=8.0,
                      polar_alpha, polar_cl, polar_cd)
  ```
- Yawed BEM call (requires `psi_vec`):
  ```python
  psi_vec = np.arange(0, 2*np.pi, 0.01)
  results = BEM_cycle(u_inf, r_over_R, r_root, r_tip, Omega, R, B,
                      chord_distribution, twist_distribution,
                      yaw_angle=15.0, tip_speed_ratio=8.0,
                      polar_alpha, polar_cl, polar_cd,
                      psi_vec=psi_vec, prandtl_correction=True)
  ```

## Integration Notes
- Data path is built with `os.path.join(os.path.dirname(__file__), '..', 'data', 'DU95W180.cvs')` for portability.
- Plotting functions expect the `results` dict structure from `BEM_cycle`; see `src/BEM_plots.py` for variable names used.
- Changing blade geometry: edit `chord_distribution`/`twist_distribution` in `src/main_BEM.py`; for optimization, use `blade_optimization.py` outputs.

If any of the above is unclear (e.g., yaw constraints, normalization, or data file format), tell us what needs elaboration and we’ll refine these instructions.