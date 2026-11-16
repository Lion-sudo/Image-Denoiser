#include <denoiser/NonLocalMeansFilter.hpp>

#include "detail/ImageUtils.hpp"

#include <opencv2/core.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>

namespace denoiser {

void NonLocalMeansFilter::validateParams(const DenoiseParams& params)
{
    // Window and patch sizes must be positive odd integers; sigmas are checked
    // by the shared validation as well.
    params.validatePatchBased();
}

double NonLocalMeansFilter::patchSquaredDistance(const cv::Mat& padded, int x1, int y1,
                                                 int x2, int y2, int halfPatch, int channels)
{
    double distance = 0.0;

    for (int dy = -halfPatch; dy <= halfPatch; ++dy) {
        // Both row pointers are based at the patch's leftmost column, so every
        // indexed offset stays non-negative.
        const float* patchRow =
            padded.ptr<float>(y1 + dy) + static_cast<std::size_t>(x1 - halfPatch) * channels;
        const float* referenceRow =
            padded.ptr<float>(y2 + dy) + static_cast<std::size_t>(x2 - halfPatch) * channels;

        for (int dx = -halfPatch; dx <= halfPatch; ++dx) {
            const std::size_t offset = static_cast<std::size_t>(dx + halfPatch) * channels;
            for (int c = 0; c < channels; ++c) {
                const double difference =
                    static_cast<double>(patchRow[offset + c]) - referenceRow[offset + c];
                distance += difference * difference;
            }
        }
    }

    // Normalize over patch area and channels: h keeps the same meaning for any
    // patchSize / channel configuration.
    const double samples = static_cast<double>(2 * halfPatch + 1) * (2 * halfPatch + 1) * channels;
    return distance / samples;
}

double NonLocalMeansFilter::weightFromDistance(double distance, double intensitySigma)
{
    return std::exp(-distance / (intensitySigma * intensitySigma));
}

cv::Mat NonLocalMeansFilter::applyPass(const cv::Mat& working, const DenoiseParams& params,
                                       const ProgressFn& rowProgress) const
{
    const int channels = working.channels();
    const int halfPatch = params.patchSize / 2;
    const int halfSearch = params.searchWindow / 2;
    // Padding must cover the farthest patch corner: half search + half patch.
    const int radius = halfPatch + halfSearch;
    const double intensitySigma = params.intensitySigma;
    const double rowCount = static_cast<double>(working.rows);

    // Completed-row counter for progress reporting; row order is unspecified
    // under OpenMP, so the fraction is based on the count alone.
    std::atomic<int> completedRows{0};

    const cv::Mat padded = detail::makePadded(working, radius);
    cv::Mat result(working.size(), CV_MAKETYPE(CV_32F, channels));

#if defined(DENOISER_WITH_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < working.rows; ++y) {
        float* resultRow = result.ptr<float>(y);

        for (int x = 0; x < working.cols; ++x) {
            const int centerX = x + radius;
            const int centerY = y + radius;

            double weightSum = 0.0;
            double accumulation[detail::kMaxSupportedChannels] = {0.0, 0.0, 0.0, 0.0};

            for (int offsetY = -halfSearch; offsetY <= halfSearch; ++offsetY) {
                for (int offsetX = -halfSearch; offsetX <= halfSearch; ++offsetX) {
                    const int neighbourX = centerX + offsetX;
                    const int neighbourY = centerY + offsetY;

                    const double distance = patchSquaredDistance(
                        padded, centerX, centerY, neighbourX, neighbourY, halfPatch, channels);
                    const double weight = weightFromDistance(distance, intensitySigma);

                    weightSum += weight;

                    const float* neighbour =
                        padded.ptr<float>(neighbourY) + static_cast<std::size_t>(neighbourX) * channels;
                    for (int c = 0; c < channels; ++c) {
                        accumulation[c] += weight * neighbour[c];
                    }
                }
            }

            for (int c = 0; c < channels; ++c) {
                resultRow[static_cast<std::size_t>(x) * channels + c] =
                    static_cast<float>(accumulation[c] / weightSum);
            }
        }

        if (rowProgress) {
            rowProgress(
                static_cast<double>(completedRows.fetch_add(1, std::memory_order_relaxed) + 1)
                / rowCount);
        }
    }
    return result;
}

void NonLocalMeansFilter::denoise(const cv::Mat& input, cv::Mat& output,
                                  const DenoiseParams& params) const
{
    denoise(input, output, params, {});
}

void NonLocalMeansFilter::denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params,
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
