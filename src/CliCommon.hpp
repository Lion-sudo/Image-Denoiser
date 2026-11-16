#pragma once

// Internal, header-only helpers shared by the CLI's batch and interactive
// front-ends. Not part of the public library API.

#include <denoiser/DenoiserFactory.hpp>
#include <denoiser/ImageDenoiser.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace denoiser::cli {

/// True when stdout is an interactive terminal, i.e. safe for \\r line redraws.
/// Piped output (files, CI logs) stays free of progress noise.
inline bool stdoutIsTerminal()
{
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

/// \brief Serializes console writes from progress callbacks.
///
/// Function-local static: initialized once, thread-safely, on first use, and
/// shared by every callback so that two progress reporters can never
/// interleave their partial lines.
inline std::mutex& consoleWriteMutex()
{
    static std::mutex mutex;
    return mutex;
}

/// \brief A ProgressFn rendering a \\r-updated percentage line on a terminal.
///
/// Returns an empty callback when stdout is piped, so batch output stays clean
/// and the filters skip the callback entirely.
///
/// Thread safety: the filters invoke this from OpenMP worker threads. The
/// percentage is deduplicated with an atomic compare-and-set, and the line is
/// formatted into a local buffer first, then handed to the stream under a
/// shared mutex as one unformatted `write()`. Formatting directly into
/// `std::cout` from several workers would interleave `\\r` redraws and produce
/// garbled output even though each insertion is individually atomic.
inline ImageDenoiser::ProgressFn consoleProgress(const std::string& label)
{
    if (!stdoutIsTerminal()) {
        return {};
    }
    return [label, lastPrinted = std::make_shared<std::atomic<int>>(-1)](double fraction) {
        const int percent =
            std::max(0, std::min(100, static_cast<int>(fraction * 100.0 + 0.5)));
        if (lastPrinted->exchange(percent, std::memory_order_relaxed) == percent) {
            return;
        }

        std::ostringstream formatted;
        formatted << '\r' << "  " << label << ' ' << std::setw(3) << percent << '%';
        const std::string line = formatted.str();

        const std::lock_guard<std::mutex> guard(consoleWriteMutex());
        std::cout.write(line.data(), static_cast<std::streamsize>(line.size()));
        std::cout.flush();
    };
}

/// Terminates a consoleProgress line (no-op when stdout is piped).
inline void endProgressLine()
{
    if (stdoutIsTerminal()) {
        const std::lock_guard<std::mutex> guard(consoleWriteMutex());
        std::cout << '\n';
    }
}

/// Adds zero-mean Gaussian noise in the image's native intensity domain.
inline void addGaussianNoise(cv::Mat& image, double sigma)
{
    cv::Mat working;
    image.convertTo(working, CV_32F);

    cv::Mat noise(image.size(), CV_MAKETYPE(CV_32F, image.channels()));
    cv::randn(noise, 0.0, sigma);
    working += noise;

    // Saturating cast clamps the noisy values back into the valid range.
    working.convertTo(image, image.depth());
}

/// Human-readable name for a `cv::Mat` depth, e.g. "512x512, 3 channel(s), 8-bit".
///
/// The raw depth constant (0, 2, 5, ...) is an implementation detail and means
/// nothing to an operator, so every user-facing message goes through here.
inline std::string describeDepth(int depth)
{
    switch (depth) {
    case CV_8U:
        return "8-bit";
    case CV_16U:
        return "16-bit";
    case CV_32F:
        return "32-bit float";
    default:
        return "depth " + std::to_string(depth);
    }
}

/// One-line summary of an image's geometry, e.g. "512x512, 3 channel(s), 8-bit".
inline std::string describeImage(const cv::Mat& image)
{
    std::ostringstream out;
    out << image.cols << "x" << image.rows << ", " << image.channels() << " channel(s), "
        << describeDepth(image.depth());
    return out.str();
}

/// \brief Width of the identifier column used by every listing and menu.
///
/// Derived from the longest registered identifier rather than hard-coded: a
/// fixed width of 10 truncated the column for "anisotropic" (11 characters),
/// which glued the description onto the identifier.
inline int identifierColumnWidth()
{
    std::size_t width = std::string("Identifier").size();
    for (const auto& identifier : DenoiserFactory::available()) {
        width = std::max(width, identifier.size());
    }
    return static_cast<int>(width) + 2;
}

/// Prints every registered denoiser with its description.
inline void printAvailableFilters(std::ostream& out)
{
    out << "Available denoisers:\n\n";
    const int width = identifierColumnWidth();
    for (const auto& identifier : DenoiserFactory::available()) {
        const auto denoiser = DenoiserFactory::create(identifier);
        out << "  " << std::left << std::setw(width) << identifier << denoiser->description()
            << '\n';
    }
}

/// Runs one denoising pass and returns the elapsed wall-clock time in milliseconds.
///
/// A const reference is enough: denoise() is const and writes only into \p
/// output. \p onProgress may be empty; see consoleProgress().
inline double timedDenoise(const ImageDenoiser& denoiser, const cv::Mat& input,
                           cv::Mat& output, const DenoiseParams& params,
                           const ImageDenoiser::ProgressFn& onProgress = {})
{
    const auto startedAt = std::chrono::steady_clock::now();
    if (onProgress) {
        denoiser.denoise(input, output, params, onProgress);
    } else {
        denoiser.denoise(input, output, params);
    }
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - startedAt)
        .count();
}

} // namespace denoiser::cli
