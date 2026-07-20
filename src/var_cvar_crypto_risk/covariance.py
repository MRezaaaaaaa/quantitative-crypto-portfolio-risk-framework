"""Covariance validation, diagnostics, and deterministic PSD repair."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_covariance_frame(covariance: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(covariance, pd.DataFrame):
        raise ValueError("covariance must be a pandas DataFrame.")
    if covariance.empty:
        raise ValueError("covariance is empty.")
    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"covariance must be square, got {covariance.shape}.")
    if not covariance.index.is_unique or not covariance.columns.is_unique:
        raise ValueError("covariance row and column labels must be unique.")
    if list(covariance.index) != list(covariance.columns):
        raise ValueError(
            "covariance row and column labels must match in the same order."
        )
    values = covariance.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("covariance must contain only finite values.")
    return covariance.astype(float)


def covariance_diagnostics(
    covariance: pd.DataFrame,
    tolerance: float = 1e-10,
) -> dict:
    """Return symmetry, eigenvalue, conditioning, and PSD diagnostics."""
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}.")
    frame = _validate_covariance_frame(covariance)
    values = frame.to_numpy(dtype=float)
    symmetric = 0.5 * (values + values.T)
    diagonal = np.diag(symmetric)
    scale = max(float(np.max(np.abs(diagonal))), np.finfo(float).tiny)
    absolute_tolerance = float(tolerance) * scale
    eigenvalues = np.linalg.eigvalsh(symmetric)
    min_eigenvalue = float(eigenvalues.min())
    max_eigenvalue = float(eigenvalues.max())
    max_asymmetry = float(np.max(np.abs(values - values.T)))
    is_psd = bool(min_eigenvalue >= -absolute_tolerance)
    is_positive_definite = bool(min_eigenvalue > absolute_tolerance)
    condition_number = (
        float(max_eigenvalue / min_eigenvalue) if is_positive_definite else float("inf")
    )
    return {
        "n_assets": int(frame.shape[0]),
        "is_symmetric": bool(max_asymmetry <= absolute_tolerance),
        "is_psd": is_psd,
        "is_positive_definite": is_positive_definite,
        "max_asymmetry": max_asymmetry,
        "min_variance": float(diagonal.min()),
        "min_eigenvalue": min_eigenvalue,
        "max_eigenvalue": max_eigenvalue,
        "condition_number": condition_number,
        "absolute_tolerance": absolute_tolerance,
    }


def prepare_covariance_matrix(
    covariance: pd.DataFrame,
    policy: str = "repair",
    tolerance: float = 1e-10,
    eigenvalue_floor: float = 1e-8,
) -> tuple[pd.DataFrame, dict]:
    """Validate covariance and either repair it or fail under strict policy.

    The repair works in correlation space: it symmetrizes the matrix, clips
    correlation eigenvalues to a positive floor, restores a unit diagonal,
    and then reconstructs covariance with the original variances. This
    preserves marginal variances while producing a PSD matrix. With strictly
    positive variances the repaired matrix is positive definite.

    Parameters
    ----------
    covariance : pandas.DataFrame
        Square covariance matrix with matching row/column labels.
    policy : {"repair", "strict"}
        ``repair`` applies the deterministic transformation when needed.
        ``strict`` rejects matrices that are not symmetric positive definite.
    tolerance : float
        Relative numerical tolerance scaled by the largest variance.
    eigenvalue_floor : float
        Positive eigenvalue floor used in correlation space.
    """
    if policy not in {"repair", "strict"}:
        raise ValueError("policy must be 'repair' or 'strict'.")
    if eigenvalue_floor <= 0:
        raise ValueError(f"eigenvalue_floor must be > 0, got {eigenvalue_floor}.")

    frame = _validate_covariance_frame(covariance)
    before = covariance_diagnostics(frame, tolerance=tolerance)
    values = frame.to_numpy(dtype=float)
    symmetric = 0.5 * (values + values.T)
    diagonal = np.diag(symmetric).copy()
    absolute_tolerance = float(before["absolute_tolerance"])

    if np.any(diagonal < -absolute_tolerance):
        raise ValueError("covariance contains a materially negative variance.")

    needs_repair = not (before["is_symmetric"] and before["is_positive_definite"])
    if policy == "strict" and needs_repair:
        raise ValueError(
            "covariance must be symmetric positive definite under strict policy."
        )

    if not needs_repair:
        report = {
            "policy": policy,
            "repaired": False,
            "reasons": [],
            "before": before,
            "after": before.copy(),
            "frobenius_adjustment": 0.0,
            "relative_frobenius_adjustment": 0.0,
        }
        return frame.copy(), report

    diagonal = np.maximum(diagonal, 0.0)
    scale = max(float(np.max(diagonal)), np.finfo(float).tiny)
    positive = diagonal > float(tolerance) * scale
    repaired_values = np.zeros_like(symmetric)

    if np.any(positive):
        positive_idx = np.flatnonzero(positive)
        std = np.sqrt(diagonal[positive_idx])
        sub_covariance = symmetric[np.ix_(positive_idx, positive_idx)]
        correlation = sub_covariance / np.outer(std, std)
        correlation = 0.5 * (correlation + correlation.T)
        np.fill_diagonal(correlation, 1.0)

        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        clipped = np.maximum(eigenvalues, float(eigenvalue_floor))
        repaired_correlation = (eigenvectors * clipped) @ eigenvectors.T
        repaired_correlation = 0.5 * (repaired_correlation + repaired_correlation.T)
        normalization = np.sqrt(np.diag(repaired_correlation))
        repaired_correlation = repaired_correlation / np.outer(
            normalization, normalization
        )
        np.fill_diagonal(repaired_correlation, 1.0)
        repaired_sub = repaired_correlation * np.outer(std, std)
        repaired_values[np.ix_(positive_idx, positive_idx)] = repaired_sub

    np.fill_diagonal(repaired_values, diagonal)
    repaired_values = 0.5 * (repaired_values + repaired_values.T)
    repaired = pd.DataFrame(
        repaired_values,
        index=frame.index,
        columns=frame.columns,
    )
    after = covariance_diagnostics(repaired, tolerance=tolerance)
    adjustment = float(np.linalg.norm(repaired_values - values, ord="fro"))
    baseline = max(float(np.linalg.norm(values, ord="fro")), np.finfo(float).tiny)

    reasons: list[str] = []
    if not before["is_symmetric"]:
        reasons.append("asymmetry")
    if not before["is_psd"]:
        reasons.append("negative_eigenvalue")
    elif not before["is_positive_definite"]:
        reasons.append("singular_or_near_singular")

    report = {
        "policy": policy,
        "repaired": True,
        "reasons": reasons,
        "before": before,
        "after": after,
        "frobenius_adjustment": adjustment,
        "relative_frobenius_adjustment": adjustment / baseline,
    }
    return repaired, report


__all__ = ["covariance_diagnostics", "prepare_covariance_matrix"]
