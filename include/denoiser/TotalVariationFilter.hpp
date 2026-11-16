#pragma once

#include <denoiser/ImageDenoiser.hpp>

namespace denoiser {

/// \brief Optimization-based denoiser: Total Variation (Rudin-Osher-Fateri)
///        minimized with Chambolle's dual projection algorithm.
///
/// Solves the ROF energy
///
///     min_u  1/2 ||u - f||^2  +  lambda * TV(u)
///
/// through its dual formulation. Chambolle's projection iterates a dual vector
/// field p = (px, py), kept in the unit ball, from which the primal estimate is
/// recovered as
///
///     u = f - lambda * div(p).
///
/// Each of the `num_iterations` steps computes
///
///     g  = div(p) - f/lambda
///     p+ = (p + tau * grad(g)) / (1 + tau * |grad(g)|),   tau = time_step
///
/// and the final image is u = f - lambda * div(p) after the last update.
/// Larger `lambda` values trade data fidelity for smoothness, producing
/// piecewise-flat reconstructions with sharp, unblurred edges.
///
/// Discretization (adjoint-consistent, so the fixed point solves the ROF
/// energy above):
///   - grad uses forward differences with a zero difference at the right and
///     bottom border;
///   - div uses backward differences with zero extension at the left and top
///     border (exactly the adjoint of the forward gradient);
///   - channels are processed independently on CV_32F planes.
///
/// \note Chambolle's convergence proof guarantees tau <= 1/8 on a 2D grid; the
///       shared parameter bound (0, 0.25] also works well in practice and is
///       kept for consistency with the explicit PDE scheme.
///
/// \note The shared `num_iterations` default (30) is tuned for the PDE filter;
///       TV benefits from longer runs - about 50-100 projection steps reach
///       full convergence (e.g. --num-iterations 100).
class TotalVariationFilter final : public ImageDenoiser {
public:
    TotalVariationFilter() = default;

    void denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params) const override;
    void denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params,
                 const ProgressFn& onProgress) const override;

    [[nodiscard]] std::string name() const override { return "tv"; }

    [[nodiscard]] std::string description() const override
    {
        return "Optimization-based ROF total variation minimization (Chambolle's dual "
               "projection): piecewise-flat result with sharp edges. Fast iterative solver.";
    }

private:
    /// \brief Validates TV-specific parameters. \throws InvalidParameterError.
    static void validateParams(const DenoiseParams& params);

    /// \brief divergence = backward-difference div of (px, py) with zero
    ///        extension: div(y,x) = (px(y,x) - px(y,x-1)) + (py(y,x) - py(y-1,x)).
    ///
    /// All matrices are single-channel CV_32F planes of identical size.
    static void backwardDivergence(const cv::Mat& px, const cv::Mat& py, cv::Mat& divergence);

    /// \brief Forward differences of \p g: gx(y,x) = g(y,x+1) - g(y,x) (zero at
    ///        the last column), gy(y,x) = g(y+1,x) - g(y,x) (zero at the last row).
    static void forwardGradient(const cv::Mat& g, cv::Mat& gx, cv::Mat& gy);

    /// \brief Chambolle dual update in place:
    ///        p+ = (p + tau * (gx, gy)) / (1 + tau * |(gx, gy)|).
    static void projectDual(cv::Mat& px, cv::Mat& py, const cv::Mat& gx, const cv::Mat& gy,
                            double tau);

    /// \brief Runs the dual iteration on one single-channel plane and writes
    ///        u = f - lambda * div(p) into \p u. \p stepProgress is invoked
    ///        with the completed projection-step fraction.
    static void solveChannel(const cv::Mat& f, cv::Mat& u, double lambda, double tau,
                             int numIterations, const ProgressFn& stepProgress);

    /// \brief Runs the full solve channel-by-channel over the [0, 1] float working buffer.
    ///        \p channelProgress is invoked with the completed channel fraction.
    [[nodiscard]] cv::Mat applyPass(const cv::Mat& working, const DenoiseParams& params,
                                    const ProgressFn& channelProgress) const;
};

} // namespace denoiser
