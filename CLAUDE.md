# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`dispersionrelations` is a Python library for phenomenological particle physics: constants, kinematics, dispersion integrals, and dynamics (vertices, propagators, coupled-channel models) built on unitarity and analyticity of the S-matrix. Source is in `src/dispersionrelations/`, tests in `tests/`, Sphinx docs in `docs/source/`. Development happens on `dev`; PRs go to `main`.

## Commands

The project is managed with `uv` (CI uses plain `pip install .[dev]` / `.[docs]`).

```bash
uv sync --extra dev                       # install with dev extras (add --extra docs for docs)
uv run pytest                             # full suite; pyproject adds --cov ... --cov-report=html by default
uv run pytest --no-cov                    # faster, no coverage report
uv run pytest tests/test_integrals.py --no-cov                          # one file
uv run pytest tests/test_integrals.py::test_DispersionIntegralRHC_real --no-cov   # one test
cd docs && make html                      # build docs
```

Docs build notes:
- `plotting.py` sets `text.usetex=True` at import, and many docstrings contain `.. plot::` directives that import it, so building docs needs a LaTeX install (CI installs `texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra dvipng cm-super`).
- Citations in docstrings use `:cite:` keys from `docs/source/literature.bib`.

CI: pushes to `dev` run pytest on Python 3.9 only; `main` runs Python 3.9–3.13 plus the docs build.

## Compatibility constraints

- Minimum Python is 3.9. Every module starts with `from __future__ import annotations` so `X | Y` works in annotations, but don't use 3.10+ syntax at runtime (`match`, `X | Y` outside annotations).
- Some code uses numpy-2-only names (`np.atan`, `np.concat`), but `numpy` is not pinned in `pyproject.toml`.

## Conventions

- Natural units with `GeV = 1` (`constants.py`). Masses, widths and branching ratios are module-level constants, with `_STD` siblings for uncertainties.
- **The kinematics and vertex functions take squared masses** (`m1_squared`, `m2_squared`), e.g. `phase_space_twobody(s, m1**2, m2**2)`. Older callers in `dynamics.py` still pass plain masses, which is a known bug.
- Functions are vectorised over numpy arrays and work on complex `s`. Many add `+ 0j` before `np.sqrt` to force the complex branch.
- Modules follow one layout with box-drawn comment headers (Typing / Logging / External imports / Internal imports / section name). Each module defines `logger = logging.getLogger(__name__)`; the package installs a `NullHandler`.
- Docstrings are numpydoc style (`Parameters`, `Returns`, `Notes` with `.. math::`, `Examples` with `.. plot::`).

## Architecture

Module dependency order (lower depends on higher):

```
constants, kinematics, conformal         (leaf functions)
domains                                  (DomainedFunction, DomainedFunctionCollection, Domainify)
complex                                  (branch-cut sqrt/log, schwarz_reflection, Complex2RS)
integrals                                (Gauss–Legendre quadrature, partitions, SimplePoleMonomial, DispersionIntegralRHC)
interpolation                            (RecursiveGrid, ComplexPrecomputatinator, ComplexInterpolator)
dynamics / dynamics_new                  (vertices, taming factors, channels, unstable particles, TwoPotentialModel)
data                                     (independent: chi², covariance helpers, Datablock/Datawriter/Datareader)
plotting                                 (docs-only helpers, excluded from coverage)
```

Key ideas that span several files:

- **Domained functions.** A `DomainedFunction` is a callable with an `in_domain(z)` mask. A `DomainedFunctionCollection` holds an ordered list of them; each point is evaluated by the *first* function whose domain contains it (`exclusive_masks`), and points nobody claims become NaN. This is how precomputed interpolators take priority over the slower exact computation, which acts as an always-valid fallback (`Domainify` wraps a plain callable with an all-true domain).
- **`Complex2RS`** (`complex.py`) represents a function on a two-sheeted Riemann surface. It has three `DomainedFunctionCollection`s: `sheet_1`, `sheet_2`, `discontinuity`. You evaluate it with `f(z, sheet=1|2)`. `integrals.disp_int_to_complex_2RS` fills one from a `DispersionIntegralRHC`: sheet 2 = sheet 1 − discontinuity, and the discontinuity is `2i·integrand`, extended to the lower half-plane by Schwarz reflection.
- **`DispersionIntegralRHC`** computes a subtracted dispersion integral along the right-hand cut. At construction it evaluates the integrand on a fixed Gauss–Legendre grid between `threshold * [1, *integration_split_points, infinity]`. It splits evaluation into real-axis and complex-`s` paths; the real path subtracts `f(s)` and adds the analytic remainder through `SimplePoleMonomial`.
- **Precompute → store → interpolate.** Loops are expensive, so they are evaluated on a `RecursiveGrid` (level `L` covers the square `[-scale·3^L, scale·3^L]²`), written to JSON per level by `ComplexPrecomputatinator.export_channel_level`, and loaded into a `ComplexInterpolator` (`add_level_interpolator_from_file`). Real-axis data uses the special level key `-1`. A loaded interpolator only knows the loop values, not the phase space, vertex or taming factor that produced them.
- **Dynamics is mid-refactor.** `dynamics.py` holds the original, inheritance-based classes (`Channel` extends `DispersionIntegralRHC`; plus `StableTwoBodyChannel`, `UnstableParticle`, `SemiStableTwoBodyCut`/`SemiStableTwoBodyChannel`, interpolated variants, and `TwoPotentialModel`). `dynamics_new.py` temporarily holds the composition-based replacements with the same class names:
  - A channel is a `threshold` plus a `loop: Complex2RS`.
  - `phase_space`, `vertex_function_squared` and `taming_factor_squared` are optional, because loops loaded from precomputed files don't come with them.
  - The loop discontinuity matters most, since it is what feeds the form factor in `TwoPotentialModel`.
  - For semi-stable channels the vertex and taming factor are currently folded into the phase space. The maintainer knows this is fragile and plans to change it.
  - The shared vertex functions, taming factors, `s_pole` and the `VertexFunction`/`TamingFactor` protocols stay in `dynamics.py` and are imported by `dynamics_new.py`.
  - `TwoPotentialModel` has not been ported yet and still expects old-style channels.
- **`TwoPotentialModel`** builds coupled-channel background (`TB`) and resonance (`TR`) amplitudes plus a form factor `F` as numpy arrays of shape `(*s.shape, nC, nC)`. It needs each channel's `loop(s, sheet)`, `vertex_function`, `taming_factor` and `phase_space_function`.
