import numpy as np
import pytest
import torch

from ml4nu.single_channel import (
    bootstrap_minima, bootstrap_plot_limits, in_lepton_region,
    likelihood_grid, shape_log_likelihood,
)


def test_invalid_event_is_not_silently_dropped():
    with pytest.raises(ValueError, match="event density"):
        shape_log_likelihood(torch.tensor([[-1., -1.]]), torch.ones(2), torch.ones(2), 1.)


def test_event_generation_region_matches_lepton_integration_box():
    # v2=sqrt(E)*sin(theta/2); the backward event lies outside v2<=0.65.
    mask = in_lepton_region(torch.tensor([1., 1., 0.01]), torch.tensor([1., -1., 1.]),
                            (0.25, 3.25), (0., 0.65))
    assert mask.tolist() == [True, False, False]


def test_batched_scan_matches_scalar_normalized_likelihood():
    torch.set_default_dtype(torch.float64)
    enu = torch.linspace(0.3, 4., 7)
    ds = torch.arange(1., 22.).reshape(3, 7)
    integral = torch.linspace(1., 2., 7)
    phi = torch.linspace(0.2, 1., 7)
    s23, d31 = torch.meshgrid(torch.tensor([0.65, 0.8]), torch.tensor([0.0024, 0.0026]), indexing="ij")

    def probability(energy, s23, d31):
        p = torch.zeros(3, 3, *torch.broadcast_shapes(energy.shape, s23.shape, d31.shape))
        p[1, 1] = 0.1 + s23 * energy / (1 + energy) + d31 * 10
        return p

    actual = likelihood_grid(ds, integral, enu, phi, s23, d31, probability, chunk_size=2)
    expected = torch.stack([
        shape_log_likelihood(ds, integral, phi * probability(enu, s, d)[1, 1], enu[1]-enu[0])
        for s, d in zip(s23.flatten(), d31.flatten())], dim=1).reshape(3, 2, 2)
    assert torch.allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_chunked_bootstraps_match_direct_maximization():
    torch.set_default_dtype(torch.float64)
    likelihood = torch.tensor([[[1., 4.], [2., 3.]], [[5., 0.], [2., 1.]]])
    weights = torch.tensor([[1., 0.], [0., 1.], [0.5, 0.5]])
    x, y = torch.meshgrid(torch.tensor([0.8, 1.]), torch.tensor([0.0023, 0.0027]), indexing="ij")
    bx, by = bootstrap_minima(likelihood, weights, x, y, batch_size=1)
    indices = (weights @ likelihood.flatten(1)).argmax(1)
    np.testing.assert_array_equal(bx, x.flatten()[indices].numpy())
    np.testing.assert_array_equal(by, y.flatten()[indices].numpy())


def test_scan_with_actual_nufast_matches_unbatched_probabilities():
    from ml4nu.nufast import Probability_Matter_LBL

    torch.set_default_dtype(torch.float64)
    enu = torch.linspace(0.3, 5., 11)
    dsigma = torch.arange(1., 34.).reshape(3, 11)
    integral, phi = torch.ones(11), torch.linspace(0.1, 1., 11)
    s23, d31 = torch.meshgrid(torch.tensor([0.7, 0.75]), torch.tensor([0.0024, 0.0026]), indexing="ij")

    def probability(energy, s23, d31):
        return Probability_Matter_LBL(0.307, 0.02195, s23**2, 3.08923,
                                      7.49e-5, d31, 1300, energy)

    actual = likelihood_grid(dsigma, integral, enu, phi, s23, d31, probability, chunk_size=3)
    expected = torch.stack([
        shape_log_likelihood(dsigma, integral, phi * probability(enu, s, d)[1, 1], enu[1]-enu[0])
        for s, d in zip(s23.flatten(), d31.flatten())], dim=1).reshape(3, 2, 2)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_plot_limits_include_estimates_outside_old_paper_window():
    samples = np.array([[0.85, 0.0021], [0.99, 0.0029], [1., 0.0026]])
    xlim, ylim = bootstrap_plot_limits(samples, np.array([0.985, 0.002534]))
    assert xlim[0] < samples[:, 0].min() and xlim[1] == 1.
    assert ylim[0] < samples[:, 1].min() and ylim[1] > samples[:, 1].max()
