#include <denoiser/AnisotropicDiffusionFilter.hpp>

#include "detail/ImageUtils.hpp"

#include <denoiser/Error.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

namespace denoiser {

void AnisotropicDiffusionFilter::validateParams(const DenoiseParams& params)
{
    params.validateDiffusion();
}

void AnisotropicDiffusionFilter::diffusionStep(const cv::Mat& state, cv::Mat& next,
                                               double timeStep, double conductionCoefficient)
{
    // The one-pixel replicated ring is never updated; iterating over the
    // interior means all four neighbors are guaranteed to be in bounds, so the
    // hot loop needs no boundary checks at all.
    const int rows = state.rows - 2;
    const int cols = state.cols - 2;
    const int channels = state.channels();
    const double invKSquared = 1.0 / (conductionCoefficient * conductionCoefficient);

#if defined(DENOISER_WITH_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 1; y <= rows; ++y) {
        // Direct row pointers: the inner loop walks contiguous float data.
        const float* north = state.ptr<float>(y - 1);
        const float* current = state.ptr<float>(y);
        const float* south = state.ptr<float>(y + 1);
        float* updated = next.ptr<float>(y);

        for (int x = 1; x <= cols; ++x) {
            for (int c = 0; c < channels; ++c) {
                const std::size_t center = static_cast<std::size_t>(x) * channels + c;
                const double value = current[center];

                // Cardinal gradient estimates from the previous time step.
                const double dNorth = north[center] - value;
                const double dSouth = south[center] - value;
                const double dEast = current[center + channels] - value;
                const double dWest = current[center - channels] - value;

                // Perona-Malik c1 conductance, one per direction: gradients far
                // above k (edges) yield ~0, gradients below k yield ~1.
                const double cNorth = std::exp(-dNorth * dNorth * invKSquared);
                const double cSouth = std::exp(-dSouth * dSouth * invKSquared);
                const double cEast = std::exp(-dEast * dEast * invKSquared);
                const double cWest = std::exp(-dWest * dWest * invKSquared);

                updated[center] = static_cast<float>(
                    value + timeStep * (cNorth * dNorth + cSouth * dSouth
                                        + cEast * dEast + cWest * dWest));
            }
        }
    }
}

cv::Mat AnisotropicDiffusionFilter::applyPass(const cv::Mat& working, const DenoiseParams& params,
                                              const ProgressFn& stepProgress) const
{
    // Pad once with a replicated ring; the ring is never written afterwards,
    // which realizes a fixed zero-flux boundary condition for all iterations.
    cv::Mat state;
    cv::copyMakeBorder(working, state, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    // Double buffer for the Jacobi scheme; the clone gives `next` the same
    // (invariant) replicated border.
    cv::Mat next = state.clone();

    for (int step = 0; step < params.num_iterations; ++step) {
        diffusionStep(state, next, params.time_step, params.conduction_coefficient);
        std::swap(state, next); // O(1): cv::Mat swap moves reference counts only
        if (stepProgress) {
            stepProgress(static_cast<double>(step + 1) / params.num_iterations);
        }
    }

    // After the last swap, `state` holds the newest field; cut the ring off.
    return state(cv::Rect(1, 1, working.cols, working.rows)).clone();
}

void AnisotropicDiffusionFilter::denoise(const cv::Mat& input, cv::Mat& output,
                                         const DenoiseParams& params) const
{
    denoise(input, output, params, {});
}

void AnisotropicDiffusionFilter::denoise(const cv::Mat& input, cv::Mat& output,
                                         const DenoiseParams& params,
                                         const ProgressFn& onProgress) const
{
    validateCommon(input, params);
    validateParams(params);

    // Copy into a private [0, 1] float buffer up front, which makes in-place
    // calls (output aliasing input) safe by construction.
    cv::Mat current = detail::toWorkingFloat(input);
    const int totalPasses = std::max(1, params.iterations);
    for (int iteration = 0; iteration < params.iterations; ++iteration) {
        current = applyPass(current, params, [&](double fractionWithinPass) {
            if (onProgress) {
                onProgress((iteration + fractionWithinPass) / totalPasses);
            }
        });
    }

    if (onProgress) {
        onProgress(1.0);
    }

    detail::fromWorkingFloat(current, output, input.depth());
}

} // namespace denoiser
