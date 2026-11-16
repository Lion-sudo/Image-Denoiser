#include <denoiser/DctFilter.hpp>

#include "detail/ImageUtils.hpp"

#include <denoiser/Error.hpp>

#include <opencv2/core.hpp>

#include <algorithm>
#include <atomic>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace denoiser {

void DctFilter::validateParams(const DenoiseParams& params)
{
    params.validateTransform();
}

int DctFilter::effectiveBlockSize(const cv::Mat& working, const DenoiseParams& params)
{
    // Two constraints, in this order:
    //   1. cv::dct() only implements *even* block sizes - an odd block raises
    //      "Odd-size DCT's are not implemented" from inside the parallel
    //      region, where the exception would escape into std::terminate;
    //   2. the block must fit the image, so clamp to the smaller dimension.
    // Rounding down to even after clamping, then flooring at 2, satisfies both
    // (images below 2 pixels are rejected before this point).
    const int limit = std::min(working.rows, working.cols);
    int block = std::min(params.blockSize, limit);
    block -= block % 2;
    return std::max(2, block);
}

/// \brief Records the first exception raised inside a parallel region.
///
/// An exception must never cross an OpenMP region boundary: the runtime calls
/// std::terminate instead of unwinding, so one failing block would abort the
/// whole process. Workers stash the exception here and the calling thread
/// rethrows it once the region has been left.
void recordParallelFailure(std::exception_ptr& failure, std::exception_ptr error)
{
#if defined(DENOISER_WITH_OPENMP)
#pragma omp critical(denoiser_dct_failure)
#endif
    {
        if (!failure) {
            failure = std::move(error);
        }
    }
}

void DctFilter::softThresholdInPlace(cv::Mat& coefficients, double threshold)
{
    float* data = coefficients.ptr<float>();
    const std::size_t count = coefficients.total();

    // Element 0 is the DC coefficient: shrinking it would bias brightness, so
    // only the AC coefficients are soft-thresholded.
    for (std::size_t i = 1; i < count; ++i) {
        const double value = data[i];
        data[i] = value > threshold  ? static_cast<float>(value - threshold)
                  : value < -threshold ? static_cast<float>(value + threshold)
                                       : 0.0F;
    }
}

cv::Mat DctFilter::applyPass(const cv::Mat& working, const DenoiseParams& params,
                             const ProgressFn& rowProgress) const
{
    const int rows = working.rows;
    const int cols = working.cols;
    const int channels = working.channels();
    const int blockSize = effectiveBlockSize(working, params);
    const double threshold = params.threshold * params.intensitySigma;
    // 50% overlap: the classic compromise between artifact suppression and cost.
    const int step = std::max(1, blockSize / 2);

    std::vector<cv::Mat> planes;
    cv::split(working, planes);

    // Every pixel accumulates the reconstructions of all blocks covering it and
    // is finally normalized by the per-pixel overlap count.
    std::vector<cv::Mat> accumulated(channels);
    cv::Mat weights = cv::Mat::zeros(working.size(), CV_32F);
    for (cv::Mat& plane : accumulated) {
        plane = cv::Mat::zeros(working.size(), CV_32F);
    }

    // Block grid: origins advance by `step` and the last origin is clamped to
    // keep blocks inside the image; the clamped block still covers the border.
    const int lastY = std::max(0, rows - blockSize);
    const int lastX = std::max(0, cols - blockSize);
    const int blockRows = lastY / step + 1;
    const int blockCols = lastX / step + 1;
    const double blockRowCount = static_cast<double>(blockRows);

    // Completed block-row counter for progress reporting; processing order is
    // unspecified under OpenMP, so the fraction is based on the count alone.
    std::atomic<int> completedRows{0};

    // First failure from any worker; rethrown after the parallel region.
    std::exception_ptr failure;

#if defined(DENOISER_WITH_OPENMP)
#pragma omp parallel
#endif
    {
        // Overlapping blocks would race on shared accumulators, so each thread
        // accumulates privately and merges once at the end of the region.
        std::vector<cv::Mat> accumulatedLocal(channels);
        cv::Mat weightsLocal = cv::Mat::zeros(working.size(), CV_32F);
        for (cv::Mat& plane : accumulatedLocal) {
            plane = cv::Mat::zeros(working.size(), CV_32F);
        }

        cv::Mat block(blockSize, blockSize, CV_32F);
        cv::Mat coefficients(blockSize, blockSize, CV_32F);
        cv::Mat reconstructed(blockSize, blockSize, CV_32F);

#if defined(DENOISER_WITH_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int yIndex = 0; yIndex < blockRows; ++yIndex) {
            // The block work and the progress callback are both user-visible
            // failure points (cv::dct throws on unsupported sizes, an OpenCV
            // allocation can fail, the callback may throw), so neither may
            // unwind out of this region.
            try {
                const int by = std::min(yIndex * step, lastY);
                for (int xIndex = 0; xIndex < blockCols; ++xIndex) {
                    const int bx = std::min(xIndex * step, lastX);
                    const cv::Rect patch(bx, by, blockSize, blockSize);

                    for (int c = 0; c < channels; ++c) {
                        planes[c](patch).copyTo(block);
                        cv::dct(block, coefficients);
                        softThresholdInPlace(coefficients, threshold);
                        cv::idct(coefficients, reconstructed);
                        accumulatedLocal[c](patch) += reconstructed;
                    }
                    weightsLocal(patch) += 1.0;
                }

                if (rowProgress) {
                    rowProgress(
                        static_cast<double>(completedRows.fetch_add(1, std::memory_order_relaxed)
                                            + 1)
                        / blockRowCount);
                }
            } catch (...) {
                recordParallelFailure(failure, std::current_exception());
            }
        }

#if defined(DENOISER_WITH_OPENMP)
#pragma omp critical
#endif
        {
            for (int c = 0; c < channels; ++c) {
                accumulated[c] += accumulatedLocal[c];
            }
            weights += weightsLocal;
        }
    }

    // Rethrow on the calling thread: the region has been left, so this is an
    // ordinary exception the CLI can report instead of a std::terminate.
    if (failure) {
        std::rethrow_exception(failure);
    }

    // Normalize by the overlap count; the block grid guarantees every pixel is
    // covered, so the weights are strictly positive everywhere.
    for (int c = 0; c < channels; ++c) {
        cv::divide(accumulated[c], weights, planes[c]);
    }

    cv::Mat result;
    cv::merge(planes, result);
    return result;
}

void DctFilter::denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params) const
{
    denoise(input, output, params, {});
}

void DctFilter::denoise(const cv::Mat& input, cv::Mat& output, const DenoiseParams& params,
                        const ProgressFn& onProgress) const
{
    validateCommon(input, params);
    validateParams(params);

    // cv::dct() needs an even block that fits the image, so a sliver image
    // (1 pixel tall or wide) has no valid block at all. Report that clearly
    // instead of failing deep inside the transform.
    if (input.rows < 2 || input.cols < 2) {
        throw InvalidParameterError("dct requires an image of at least 2x2 pixels (got "
                                    + std::to_string(input.cols) + "x"
                                    + std::to_string(input.rows) + ")");
    }

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
