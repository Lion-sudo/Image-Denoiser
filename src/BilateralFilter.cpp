#include <denoiser/BilateralFilter.hpp>

#include "detail/ImageUtils.hpp"

#include <denoiser/Error.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <string>

namespace denoiser {

void BilateralFilter::validateParams(const DenoiseParams& params)
{
    params.validate();
    // The kernel radius grows linearly with sigma; bound it so a typo cannot
    // turn into an unbounded buffer allocation.
    if (params.spatialSigma > 100.0) {
        throw InvalidParameterError("spatialSigma is unreasonably large (got "
                                    + std::to_string(params.spatialSigma) + ", expected <= 100)");
    }
}

BilateralFilter::SpatialKernel BilateralFilter::buildSpatialKernel(double spatialSigma) const
{
    SpatialKernel kernel;
    // Truncate the Gaussian at 2 sigma: captures ~98% of the energy while
    // keeping the per-pixel cost moderate. At least 1 so degenerate sigmas
    // still average a 3x3 neighborhood.
    kernel.radius = std::max(1, static_cast<int>(std::ceil(2.0 * spatialSigma)));

    const int diameter = 2 * kernel.radius + 1;
    kernel.weights.assign(static_cast<std::size_t>(diameter) * diameter, 0.0F);

    const double invTwoSigmaSquared = 1.0 / (2.0 * spatialSigma * spatialSigma);
    for (int dy = -kernel.radius; dy <= kernel.radius; ++dy) {
        for (int dx = -kernel.radius; dx <= kernel.radius; ++dx) {
            const double squaredDistance = static_cast<double>(dx * dx + dy * dy);
            kernel.weights[static_cast<std::size_t>(dy + kernel.radius) * diameter
                           + (dx + kernel.radius)] =
                static_cast<float>(std::exp(-squaredDistance * invTwoSigmaSquared));
        }
    }
    return kernel;
}

cv::Mat BilateralFilter::applyPass(const cv::Mat& working, const SpatialKernel& kernel,
                                   double intensitySigma, const ProgressFn& rowProgress) const
{
    const int channels = working.channels();
    const int radius = kernel.radius;
    const int diameter = 2 * radius + 1;
    const double invTwoSigmaISquared = 1.0 / (2.0 * intensitySigma * intensitySigma);
    const double rowCount = static_cast<double>(working.rows);

    // Completed-row counter for progress reporting; row order is unspecified
    // under OpenMP, so the fraction is based on the count alone.
    std::atomic<int> completedRows{0};

    // Padded copy: every neighborhood access stays inside the buffer.
    const cv::Mat padded = detail::makePadded(working, radius);
    cv::Mat result(working.size(), CV_MAKETYPE(CV_32F, channels));

#if defined(DENOISER_WITH_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < working.rows; ++y) {
        float* resultRow = result.ptr<float>(y);
        for (int x = 0; x < working.cols; ++x) {
            // Center pixel in padded coordinates.
            const float* center =
                padded.ptr<float>(y + radius) + static_cast<std::size_t>(x + radius) * channels;

            for (int c = 0; c < channels; ++c) {
                const double centerValue = center[c];

                double weightSum = 0.0;
                double valueSum = 0.0;

                for (int dy = -radius; dy <= radius; ++dy) {
                    // Base the row pointer at padded column x so every indexed
                    // offset (dx + radius) * channels stays non-negative.
                    const float* neighbourRow =
                        padded.ptr<float>(y + radius + dy) + static_cast<std::size_t>(x) * channels;
                    const float* spatialRow =
                        kernel.weights.data() + static_cast<std::size_t>(dy + radius) * diameter;

                    for (int dx = -radius; dx <= radius; ++dx) {
                        const double difference =
                            static_cast<double>(neighbourRow[(dx + radius) * channels + c]) - centerValue;
                        const double weight =
                            static_cast<double>(spatialRow[dx + radius])
                            * std::exp(-difference * difference * invTwoSigmaISquared);

                        weightSum += weight;
                        valueSum += weight * neighbourRow[(dx + radius) * channels + c];
                    }
                }

                resultRow[static_cast<std::size_t>(x) * channels + c] =
                    static_cast<float>(valueSum / weightSum);
            }
        }

        // Report once per completed row, outside the per-pixel loop: the
        // callback is a std::function call plus an atomic read-modify-write,
        // which has no place in the innermost loop, and per-pixel reporting
        // would jump the fraction past 1.0 within the first row.
        if (rowProgress) {
            rowProgress(
                static_cast<double>(completedRows.fetch_add(1, std::memory_order_relaxed) + 1)
                / rowCount);
        }
    }
    return result;
}

void BilateralFilter::denoise(const cv::Mat& input, cv::Mat& output,
                              const DenoiseParams& params) const
{
    denoise(input, output, params, {});
}

void BilateralFilter::denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params,
                              const ProgressFn& onProgress) const
{
    validateCommon(input, params);
    validateParams(params);

    // Copy into a private [0, 1] float buffer up front, which makes in-place
    // calls (output aliasing input) safe by construction.
    cv::Mat current = detail::toWorkingFloat(input);

    const SpatialKernel kernel = buildSpatialKernel(params.spatialSigma);
    const int totalPasses = std::max(1, params.iterations);
    for (int iteration = 0; iteration < params.iterations; ++iteration) {
        current = applyPass(
            current, kernel, params.intensitySigma, [&](double fractionWithinPass) {
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
