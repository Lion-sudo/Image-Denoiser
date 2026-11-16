#pragma once

// Internal, header-only helpers shared by the denoiser implementations.
// This header is NOT part of the public API and is never installed.

#include <denoiser/Error.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace denoiser::detail {

/// Maximum channel count accepted by the library (grayscale, BGR, BGRA).
inline constexpr int kMaxSupportedChannels = 4;

/// \brief Converts an input image to a CV_32F working buffer scaled into [0, 1].
///
/// CV_8U is divided by 255, CV_16U by 65535; CV_32F is assumed to already lie
/// in [0, 1] (this is documented as a precondition of the public API).
[[nodiscard]] inline cv::Mat toWorkingFloat(const cv::Mat& input)
{
    cv::Mat working;
    input.convertTo(working, CV_32F); // preserves the channel count

    const double scale = input.depth() == CV_8U   ? 1.0 / 255.0
                         : input.depth() == CV_16U ? 1.0 / 65535.0
                                                   : 1.0;
    if (scale != 1.0) {
        working *= scale;
    }
    return working;
}

/// \brief Converts the [0, 1] float working buffer back to \p depth.
///
/// `cv::Mat::convertTo` performs a saturating cast, so values are clamped into
/// the valid range of the target depth automatically.
inline void fromWorkingFloat(const cv::Mat& working, cv::Mat& output, int depth)
{
    const double scale = depth == CV_8U   ? 255.0
                         : depth == CV_16U ? 65535.0
                                           : 1.0;
    working.convertTo(output, depth, scale);
}

/// \brief Returns a copy of \p src padded on all sides with \p radius pixels.
///
/// BORDER_REFLECT_101 (cv::borderInterpolate's default) keeps edge statistics
/// symmetric, so the filters never need per-pixel boundary special cases.
[[nodiscard]] inline cv::Mat makePadded(const cv::Mat& src, int radius)
{
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, radius, radius, radius, radius, cv::BORDER_REFLECT_101);
    return padded;
}

} // namespace denoiser::detail
