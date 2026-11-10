"""Regression tests for the truth structure functions.

Pins the 2026-10-08 fix to StructureFunctions.calc_all, where `W1 = xf2`
and `W5 = xf2` aliased one tensor and each received an in-place
`*= q2fac`, so both carried q2fac**2 while W2/W3 carried it once. That
silently redefined the toy cross section for every campaign run between
commit ee1505b and the fix.
"""
import numpy as np
import pytest
import torch

from ml4nu.structure_functions import PDFInterpolator, StructureFunctions

PDF = "data/pdf/pdf_x2048_q2048_z18_a40_q2max10000_linlog.h5"
pytestmark = pytest.mark.skipif(not __import__("os").path.exists(PDF),
                                reason="PDF grid not present")


def _grid():
    x = torch.tensor([0.05, 0.1, 0.2, 0.4])
    q2 = torch.tensor([0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
    return torch.meshgrid(x, q2, indexing="ij")


def test_q2fac_applied_once():
    X, Q = _grid()
    pdf = PDFInterpolator(PDF)
    xf2 = pdf.xf2(X, Q).clone()          # pristine, before calc_all runs
    xf3 = pdf.xf3(X, Q).clone()
    W1, W2_, W3, W4, W5 = StructureFunctions(pdf).calc_all(X, Q)
    q2fac = Q / (Q + 1)
    assert torch.allclose(W1, xf2 * q2fac), "W1 must carry q2fac exactly once"
    assert torch.allclose(W5, xf2 * q2fac), "W5 must carry q2fac exactly once"
    assert torch.allclose(W2_, 4 * X / Q * xf2 * q2fac)
    assert torch.allclose(W3, 2 * xf3 * 0.1 * q2fac)
    assert torch.all(W4 == 0)


def test_w1_w5_are_independent_objects():
    """The aliasing itself: W1 and W5 must not share storage."""
    X, Q = _grid()
    W1, _, _, _, W5 = StructureFunctions(PDFInterpolator(PDF)).calc_all(X, Q)
    assert W1 is not W5
    before = W5.clone()
    W1 += 1.0                       # mutating one must not touch the other
    assert torch.allclose(W5, before)


def test_calc_all_does_not_mutate_the_pdf_cache():
    """calc_all must not write through to the interpolator's output."""
    X, Q = _grid()
    pdf = PDFInterpolator(PDF)
    first = pdf.xf2(X, Q).clone()
    StructureFunctions(pdf).calc_all(X, Q)
    assert torch.allclose(pdf.xf2(X, Q), first), \
        "calc_all mutated xf2 in place; repeated calls would compound q2fac"
