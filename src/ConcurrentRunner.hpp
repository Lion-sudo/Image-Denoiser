#pragma once

// Internal, header-only concurrent runner shared by the CLI's batch and
// interactive front-ends. Not part of the public library API.

#include "CliCommon.hpp"

#include <denoiser/DenoiserFactory.hpp>
#include <denoiser/ImageDenoiser.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace denoiser::cli {

/// Output directory for the multi-method run.
inline constexpr const char* kOutputDirName = "output";

/// \brief One all-methods outcome: status, timing and where the result was saved.
struct RunResult {
    std::string identifier;  ///< Registry identifier, e.g. "bilateral".
    std::string displayName; ///< Human-readable class name used in the table.
    bool success = false;    ///< False if the filter threw or saving failed.
    std::string error;       ///< Diagnostic when success is false.
    std::string outputPath;  ///< Saved result image when success is true.
    double elapsedMs = 0.0;  ///< Wall-clock time of the denoise() call only (not I/O).
};

/// \brief Pretty class name for the summary table; unknown (custom) denoisers
///        fall back to their registry identifier.
inline std::string benchmarkDisplayName(const std::string& identifier)
{
    static const std::map<std::string, std::string> prettyNames = {
        {"anisotropic", "AnisotropicDiffusion"},
        {"bilateral", "BilateralFilter"},
        {"dct", "DctFilter"},
        {"nlm", "NonLocalMeansFilter"},
        {"tv", "TotalVariationFilter"},
    };
    const auto it = prettyNames.find(identifier);
    return it != prettyNames.end() ? it->second : identifier;
}

/// Input "C:/dir/photo.png" -> stem "photo", extension ".png" (defaulted).
inline void splitInputPath(const std::string& inputPath, std::string& stem, std::string& extension)
{
    const std::filesystem::path path(inputPath);
    stem = path.stem().string();
    extension = path.extension().string();
    if (extension.empty()) {
        extension = ".png";
    }
}

/// Per-result destination, e.g. output/photo_bilateral.png.
inline std::filesystem::path outputPathFor(const std::filesystem::path& outputDir,
                                           const std::string& stem,
                                           const std::string& identifier,
                                           const std::string& extension)
{
    return outputDir / (stem + "_" + identifier + extension);
}

/// \brief Compact label for the live status line (full display names go to
///        the summary table instead).
///
/// The status line must never exceed a typical 80-column terminal: a wrapped
/// line cannot be erased with '\\r' (which only returns to the last physical
/// row), and stale fragments of earlier redraws then mix with fresh numbers.
/// With 5-character labels the line stays around 60 columns for any number of
/// registered methods.
inline std::string shortStatusLabel(const std::string& identifier)
{
    static const std::map<std::string, std::string> shortNames = {
        {"anisotropic", "aniso"},
        {"bilateral", "bilat"},
        {"dct", "dct"},
        {"nlm", "nlm"},
        {"tv", "tv"},
    };
    const auto it = shortNames.find(identifier);
    return it != shortNames.end() ? it->second : identifier.substr(0, 5);
}

/// \brief One-line live status, e.g. "  aniso   42%  bilat    7%  dct  100% ...".
///
/// Each method's percentage comes from an atomic slot updated by its worker
/// thread; the main thread renders, so nothing else writes to stdout while the
/// methods run. Percentages are clamped to [0, 100], and trailing padding
/// overwrites the previous, possibly longer line.
inline std::string renderProgressLine(const std::vector<std::string>& labels,
                                      const std::vector<std::atomic<int>>& percents)
{
    std::ostringstream out;
    out << "  ";
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const int percent = std::max(0, std::min(100, percents[i].load(std::memory_order_relaxed)));
        out << std::left << std::setw(5) << labels[i] << std::right << std::setw(4) << percent
            << "%  ";
    }
    out << "            ";
    return out.str();
}

/// \brief Runs every registered denoiser concurrently on \p input, with live
///        progress feedback and no timing.
///
/// One denoiser instance is created per method and moved into its own
/// `std::async(std::launch::async, ...)` task, so the stateless filters never
/// share mutable state. The input matrix is only ever read (each task holds
/// its own refcounted cv::Mat copy) and each task writes into its own output
/// Mat - no data races.
///
/// Progress: every task forwards its filter's ProgressFn into a dedicated
/// atomic slot; the main thread polls the futures and, on a terminal, redraws
/// a one-line status showing each method's completion percentage. On a piped
/// stdout nothing is printed until the summary table.
///
/// Failures are captured per task instead of unwinding the whole suite, so one
/// broken filter never loses the others' results. Images are saved from the
/// worker tasks (one file each, no shared I/O).
inline std::vector<RunResult> runConcurrentSuite(const cv::Mat& input,
                                                 const DenoiseParams& params,
                                                 const std::filesystem::path& outputDir,
                                                 const std::string& stem,
                                                 const std::string& extension)
{
    std::filesystem::create_directories(outputDir);

    const std::vector<std::string> identifiers = DenoiserFactory::available();
    const std::size_t methodCount = identifiers.size();

    std::vector<std::string> displayNames;
    std::vector<std::string> statusLabels;
    displayNames.reserve(methodCount);
    statusLabels.reserve(methodCount);
    for (const std::string& identifier : identifiers) {
        displayNames.push_back(benchmarkDisplayName(identifier));
        statusLabels.push_back(shortStatusLabel(identifier));
    }

    std::cout << "\nRunning " << methodCount << " denoiser(s) concurrently on "
              << describeImage(input) << ", saving results to '" << outputDir.string() << "/'...\n";

    // Progress slots: one atomic percentage per method (value-initialized 0).
    std::vector<std::atomic<int>> percents(methodCount);
    for (std::atomic<int>& percent : percents) {
        percent.store(0, std::memory_order_relaxed);
    }

    std::vector<std::future<RunResult>> futures;
    futures.reserve(methodCount);
    for (std::size_t i = 0; i < methodCount; ++i) {
        const std::string& identifier = identifiers[i];
        auto denoiser = DenoiserFactory::create(identifier);
        std::atomic<int>& percent = percents[i];

        auto task = [denoiser = std::move(denoiser), input, params, identifier,
                     displayName = displayNames[i], outputDir, stem, extension,
                     &percent]() -> RunResult {
            RunResult result;
            result.identifier = identifier;
            result.displayName = displayName;
            try {
                cv::Mat output;
                const ImageDenoiser::ProgressFn onProgress =
                    [&percent](double fraction) {
                        percent.store(std::min(100, static_cast<int>(fraction * 100.0 + 0.5)),
                                      std::memory_order_relaxed);
                    };

                // Time the algorithm only; file writing is I/O, not filtering.
                const auto startedAt = std::chrono::steady_clock::now();
                denoiser->denoise(input, output, params, onProgress);
                result.elapsedMs =
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - startedAt)
                        .count();

                const std::filesystem::path destination =
                    outputPathFor(outputDir, stem, identifier, extension);
                if (cv::imwrite(destination.string(), output)) {
                    result.success = true;
                    result.outputPath = destination.string();
                    percent.store(100, std::memory_order_relaxed);
                } else {
                    result.error = "could not write result image '" + destination.string() + "'";
                }
            } catch (const std::exception& error) {
                result.error = error.what();
            } catch (...) {
                result.error = "unknown error";
            }
            return result;
        };

        futures.push_back(std::async(std::launch::async, std::move(task)));
    }

    // Feedback loop: on a terminal, redraw the status line while the methods
    // run; on a piped stdout, wait quietly. futures stay valid until get().
    const bool interactiveStdout = stdoutIsTerminal();
    std::vector<bool> ready(methodCount, false);
    std::size_t pending = methodCount;
    std::size_t statusLineLength = 0;

    while (pending > 0) {
        std::size_t nextPending = methodCount;
        for (std::size_t i = 0; i < methodCount; ++i) {
            if (ready[i]) {
                continue;
            }
            if (futures[i].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                ready[i] = true;
                percents[i].store(100, std::memory_order_relaxed);
                --pending;
            } else if (nextPending == methodCount) {
                nextPending = i;
            }
        }
        if (pending == 0) {
            break;
        }

        if (interactiveStdout) {
            const std::string line = renderProgressLine(statusLabels, percents);
            statusLineLength = line.size();
            std::cout << '\r' << line << std::flush;
            futures[nextPending].wait_for(std::chrono::milliseconds(150));
        } else {
            futures[nextPending].wait();
        }
    }

    if (interactiveStdout && statusLineLength > 0) {
        std::cout << '\r' << std::string(statusLineLength, ' ') << '\r' << std::flush;
    }

    std::vector<RunResult> results;
    results.reserve(methodCount);
    for (std::future<RunResult>& future : futures) {
        results.push_back(future.get());
    }
    return results;
}

/// \brief Prints the summary table: method, timing and where its result was saved.
///        Failed entries show "failed" and their diagnostics below the table.
///
/// Timings are wall-clock for the denoise() call while *all* methods run at
/// once, so they measure each filter under contention rather than in isolation.
/// The footnote states this, because the numbers are routinely lower when a
/// filter is run alone (see `--filter <name>`).
inline void printResultsTable(const std::vector<RunResult>& results)
{
    std::size_t nameWidth = std::string("Method").size();
    std::size_t outputWidth = std::string("Output").size();
    for (const RunResult& result : results) {
        nameWidth = std::max(nameWidth, result.displayName.size());
        outputWidth = std::max(outputWidth, (result.success ? result.outputPath.size() : 6U));
    }
    const int nameColumn = static_cast<int>(nameWidth);
    const int outputColumn = static_cast<int>(outputWidth);
    const int timeColumn = 11; // fits the "137860.3" of the slowest filter
    const int totalWidth = nameColumn + 3 + timeColumn + 3 + outputColumn;
    const std::string rule(static_cast<std::size_t>(totalWidth), '=');

    std::cout << '\n'
              << rule << '\n'
              << "Image Denoising Results\n"
              << rule << '\n'
              << std::left << std::setw(nameColumn) << "Method" << " | " << std::right
              << std::setw(timeColumn) << "Time (ms)" << " | " << std::left << "Output\n"
              << std::string(static_cast<std::size_t>(totalWidth), '-') << '\n';

    for (const RunResult& result : results) {
        std::cout << std::left << std::setw(nameColumn) << result.displayName << " | "
                  << std::right << std::setw(timeColumn);
        if (result.success) {
            std::cout << std::fixed << std::setprecision(1) << result.elapsedMs;
        } else {
            std::cout << "-";
        }
        std::cout << " | " << std::left << (result.success ? result.outputPath : "failed")
                  << '\n';
    }

    std::cout << std::defaultfloat << rule << '\n'
              << "  Times are measured while all methods run concurrently and compete\n"
              << "  for the same cores; run a single filter for isolated timings.\n";

    for (const RunResult& result : results) {
        if (!result.success) {
            std::cout << "  [" << result.displayName << "] failed: " << result.error << '\n';
        }
    }
}

} // namespace denoiser::cli
