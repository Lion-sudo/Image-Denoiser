#pragma once

#include <denoiser/ImageDenoiser.hpp>

namespace denoiser {

/// \brief PDE-based anisotropic diffusion denoiser (Perona & Malik, IEEE TPAMI 1990).
///
/// The image is treated as a physical intensity field evolving under
///
///     dI/dt = div( c(|grad I|) * grad I )
///
/// discretized on the pixel grid with an explicit (Jacobi) finite-difference
/// scheme. Every time step computes the four cardinal differences
/// N, S, E, W of each pixel, gates them by the Perona-Malik conductance
///
///     c(g) = exp( -(g / k)^2 ),   k = conduction_coefficient
///
/// and advances the field by
///
///     I <- I + dt * (cN*dN + cS*dS + cE*dE + cW*dW),   dt = time_step.
///
/// Flat regions (small gradients) conduct freely and average noise away, while
/// strong edges (gradients >> k) see a vanishing conductance and are preserved.
///
/// Numerics and boundaries:
///   - time stepping is double-buffered (read `state`, write `next`, swap), so
///     every pixel of one update sees the same time step;
///   - `dt <= 0.25` keeps the explicit scheme stable on a 2D grid (enforced by
///     DenoiseParams::validateDiffusion);
///   - the working buffer is padded once with cv::BORDER_REPLICATE and the
///     replicated ring is never updated: a fixed, zero-flux boundary condition
///     without per-iteration padding or clamped-index checks in the hot loop.
class AnisotropicDiffusionFilter final : public ImageDenoiser {
public:
    AnisotropicDiffusionFilter() = default;

    void denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params) const override;
    void denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params,
                 const ProgressFn& onProgress) const override;

    [[nodiscard]] std::string name() const override { return "anisotropic"; }

    [[nodiscard]] std::string description() const override
    {
        return "PDE-based edge-preserving diffusion (Perona-Malik): iterates a "
               "conductance-gated heat equation. Physics-inspired smoothing.";
    }

private:
    /// \brief Validates PDE-specific parameters. \throws InvalidParameterError.
    static void validateParams(const DenoiseParams& params);

    /// \brief One explicit Jacobi step over the padded state:
    ///        next = state + dt * sum_dir c(|d_dir|) * d_dir.
    ///
    /// Both \p state and \p next carry a one-pixel replicated ring; only the
    /// interior is updated, reading every neighbor from \p state.
    static void diffusionStep(const cv::Mat& state, cv::Mat& next, double timeStep,
                              double conductionCoefficient);

    /// \brief Runs num_iterations explicit steps over the [0, 1] float working buffer.
    ///        \p stepProgress is invoked with the completed time-step fraction.
    [[nodiscard]] cv::Mat applyPass(const cv::Mat& working, const DenoiseParams& params,
                                    const ProgressFn& stepProgress) const;
};

} // namespace denoiser
