#include <denoiser/TotalVariationFilter.hpp>

#include "detail/ImageUtils.hpp"

#include <denoiser/Error.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace denoiser {

void TotalVariationFilter::validateParams(const DenoiseParams& params)
{
    params.validateVariational();
}

void TotalVariationFilter::backwardDivergence(const cv::Mat& px, const cv::Mat& py,
                                              cv::Mat& divergence)
{
    const int rows = px.rows;
    const int cols = px.cols;

#if defined(DENOISER_WITH_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < rows; ++y) {
        const float* pxRow = px.ptr<float>(y);
        const float* pyRow = py.ptr<float>(y);
        const float* pyAbove = (y > 0) ? py.ptr<float>(y - 1) : nullptr;
        float* outRow = divergence.ptr<float>(y);

        for (int x = 0; x < cols; ++x) {
            // Backward differences with zero extension: the x-terms vanish at
            // the left border, the y-terms at the top border.
            const float dpx = pxRow[x] - (x > 0 ? pxRow[x - 1] : 0.0F);
            const float dpy = pyRow[x] - (y > 0 ? pyAbove[x] : 0.0F);
            outRow[x] = dpx + dpy;
        }
    }
}

void TotalVariationFilter::forwardGradient(const cv::Mat& g, cv::Mat& gx, cv::Mat& gy)
{
    const int rows = g.rows;
    const int cols = g.cols;

#if defined(DENOISER_WITH_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < rows; ++y) {
        const float* gRow = g.ptr<float>(y);
        const float* gBelow = (y < rows - 1) ? g.ptr<float>(y + 1) : nullptr;
        float* gxRow = gx.ptr<float>(y);
        float* gyRow = gy.ptr<float>(y);

        for (int x = 0; x < cols; ++x) {
            // Forward differences with a zero difference at the right and
            // bottom border (edge-replicated in the continuum picture).
            gxRow[x] = (x < cols - 1) ? gRow[x + 1] - gRow[x] : 0.0F;
            gyRow[x] = (y < rows - 1) ? gBelow[x] - gRow[x] : 0.0F;
        }
    }
}

void TotalVariationFilter::projectDual(cv::Mat& px, cv::Mat& py, const cv::Mat& gx,
                                       const cv::Mat& gy, double tau)
{
    const int rows = px.rows;
    const int cols = px.cols;

#if defined(DENOISER_WITH_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < rows; ++y) {
        float* pxRow = px.ptr<float>(y);
        float* pyRow = py.ptr<float>(y);
        const float* gxRow = gx.ptr<float>(y);
        const float* gyRow = gy.ptr<float>(y);

        for (int x = 0; x < cols; ++x) {
            // Chambolle's explicit scheme:
            // p+ = (p + tau * grad(g)) / (1 + tau * |grad(g)|).
            const double nx = static_cast<double>(pxRow[x]) + tau * gxRow[x];
            const double ny = static_cast<double>(pyRow[x]) + tau * gyRow[x];
            const double gradNorm = std::sqrt(gxRow[x] * gxRow[x] + gyRow[x] * gyRow[x]);
            const double denominator = 1.0 + tau * gradNorm;

            pxRow[x] = static_cast<float>(nx / denominator);
            pyRow[x] = static_cast<float>(ny / denominator);
        }
    }
}

void TotalVariationFilter::solveChannel(const cv::Mat& f, cv::Mat& u, double lambda, double tau,
                                        int numIterations, const ProgressFn& stepProgress)
{
    const int rows = f.rows;
    const int cols = f.cols;

    // f/lambda is invariant across the iteration; precompute it once.
    cv::Mat fOverLambda = f.clone();
    fOverLambda *= 1.0 / lambda;

    // Dual vector field, initialized to zero (u = f is the initial primal).
    cv::Mat px = cv::Mat::zeros(rows, cols, CV_32F);
    cv::Mat py = cv::Mat::zeros(rows, cols, CV_32F);

    // Preallocated scratch: no allocations inside the hot iteration loop.
    cv::Mat divergence(rows, cols, CV_32F);
    cv::Mat g(rows, cols, CV_32F);
    cv::Mat gx(rows, cols, CV_32F);
    cv::Mat gy(rows, cols, CV_32F);

    for (int step = 0; step < numIterations; ++step) {
        backwardDivergence(px, py, divergence);      // g = div(p) - f/lambda
        cv::subtract(divergence, fOverLambda, g);
        forwardGradient(g, gx, gy);
        projectDual(px, py, gx, gy, tau);
        if (stepProgress) {
            stepProgress(static_cast<double>(step + 1) / numIterations);
        }
    }

    // Primal recovery: u = f - lambda * div(p).
    backwardDivergence(px, py, divergence);
    cv::scaleAdd(divergence, -lambda, f, u);
}

cv::Mat TotalVariationFilter::applyPass(const cv::Mat& working, const DenoiseParams& params,
                                        const ProgressFn& channelProgress) const
{
    // Per-channel solve: the dual fields are independent per plane, which keeps
    // every inner loop single-channel and contiguous.
    std::vector<cv::Mat> planes;
    cv::split(working, planes);

    const int channelCount = static_cast<int>(planes.size());
    for (int c = 0; c < channelCount; ++c) {
        cv::Mat solved;
        solveChannel(
            planes[c], solved, params.lambda, params.time_step, params.num_iterations,
            [&](double fractionWithinChannel) {
                if (channelProgress) {
                    channelProgress((c + fractionWithinChannel) / channelCount);
                }
            });
        planes[c] = std::move(solved);
    }

    cv::Mat result;
    cv::merge(planes, result);
    return result;
}void TotalVariationFilter::denoise(const cv::Mat& input, cv::Mat& output,
                                   const DenoiseParams& params) const
{
    denoise(input, output, params, {});
}

void TotalVariationFilter::denoise(const cv::Mat& input, cv::Mat& output,
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
