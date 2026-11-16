#include "InteractiveSession.hpp"

#include "CliCommon.hpp"
#include "ConcurrentRunner.hpp"

#include <denoiser/DenoiserFactory.hpp>
#include <denoiser/Error.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace denoiser::cli {
namespace {

constexpr const char* kDefaultFilter = "bilateral";
constexpr const char* kAllMethodsChoice = "all";

/// \brief Line-based terminal input with graceful EOF handling.
///
/// Interactive prompts must never spin forever: when stdin is closed or
/// exhausted (e.g. piped input ran out, Ctrl+Z / Ctrl+D), every read sets the
/// eof flag and callers unwind the session.
class TerminalInput {
public:
    /// Reads one line after printing \p prompt. On EOF, sets the eof flag and
    /// returns an empty string; callers must check eof().
    std::string readLine(const std::string& prompt)
    {
        std::cout << prompt;
        std::string line;
        if (!std::getline(std::cin, line)) {
            eof_ = true;
            std::cout << '\n';
            return {};
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back(); // tolerate CRLF-terminated piped input
        }
        return line;
    }

    [[nodiscard]] bool eof() const { return eof_; }

private:
    bool eof_ = false;
};

std::string promptText(TerminalInput& in, const std::string& label, const std::string& fallback)
{
    const std::string text = in.readLine(label + " [" + fallback + "]: ");
    return (in.eof() || text.empty()) ? fallback : text;
}

double promptDouble(TerminalInput& in, const std::string& label, double fallback)
{
    std::ostringstream defaultText;
    defaultText << fallback;

    while (!in.eof()) {
        const std::string text =
            in.readLine("  " + label + " [" + defaultText.str() + "]: ");
        if (in.eof() || text.empty()) {
            return fallback;
        }

        std::istringstream parser(text);
        double value = 0.0;
        if (parser >> value) {
            parser >> std::ws;
            if (parser.eof()) {
                return value;
            }
        }
        std::cout << "  please enter a number (e.g. 0.1).\n";
    }
    return fallback;
}

int promptInt(TerminalInput& in, const std::string& label, int fallback)
{
    std::ostringstream defaultText;
    defaultText << fallback;

    while (!in.eof()) {
        const std::string text =
            in.readLine("  " + label + " [" + defaultText.str() + "]: ");
        if (in.eof() || text.empty()) {
            return fallback;
        }

        std::istringstream parser(text);
        int value = 0;
        if (parser >> value) {
            parser >> std::ws;
            if (parser.eof()) {
                return value;
            }
        }
        std::cout << "  please enter an integer (e.g. 21).\n";
    }
    return fallback;
}

/// \brief Presents the registry contents and resolves the user's choice.
///
/// Accepts a 1-based menu number, a (case-insensitive) identifier, or 'a' for
/// the concurrent all-methods benchmark. Re-prompts until the choice is valid
/// or input ends.
///
/// The menu is printed once and only the prompt repeats on invalid input:
/// reprinting eight lines of descriptions for a single typo pushes the actual
/// error message off the screen.
/// \returns the chosen identifier, "all" for the benchmark, or nullopt on EOF.
std::optional<std::string> chooseFilterName(TerminalInput& in)
{
    const std::vector<std::string> identifiers = DenoiserFactory::available();

    // Column widths come from the registry, not from hard-coded constants:
    // a fixed width truncated "anisotropic" and glued on its description, and
    // the index column also has to stay aligned past nine filters.
    const int nameWidth = identifierColumnWidth();
    const std::size_t numberWidth = std::to_string(identifiers.size()).size();
    const std::string continuation(4 + numberWidth + 2 + static_cast<std::size_t>(nameWidth), ' ');

    std::cout << "\nAvailable denoisers:\n";
    for (std::size_t i = 0; i < identifiers.size(); ++i) {
        std::cout << "  [" << std::right << std::setw(static_cast<int>(numberWidth)) << (i + 1)
                  << "] " << std::left << std::setw(nameWidth) << identifiers[i]
                  << DenoiserFactory::create(identifiers[i])->description() << '\n';
    }
    std::cout << "  [" << std::right << std::setw(static_cast<int>(numberWidth)) << 'a' << "] "
              << std::left << std::setw(nameWidth) << kAllMethodsChoice
              << "Run every registered denoiser concurrently with live progress\n"
              << continuation << "and save each result to the '" << kOutputDirName
              << "/' directory.\n";

    const std::string prompt = std::string("Choose a filter (number, name, 'a' = ")
                               + kAllMethodsChoice + ", Enter = " + kDefaultFilter + "): ";

    while (!in.eof()) {
        const std::string text = in.readLine(prompt);
        if (in.eof()) {
            return std::nullopt; // input ended: unwind rather than invent a choice
        }
        if (text.empty()) {
            return std::string(kDefaultFilter);
        }
        if (text == "a" || text == "A" || text == kAllMethodsChoice) {
            return std::string(kAllMethodsChoice);
        }

        std::istringstream parser(text);
        std::size_t choice = 0;
        if ((parser >> choice) && parser.eof() && choice >= 1 && choice <= identifiers.size()) {
            return identifiers[choice - 1];
        }

        if (DenoiserFactory::isRegistered(text)) {
            return text;
        }

        std::cout << "  '" << text << "' is not on the list: enter 1-" << identifiers.size()
                  << ", a filter name, or 'a' for all.\n";
    }
    return std::nullopt;
}

/// \brief Prompts for the parameters a given filter actually uses.
///
/// The anisotropic PDE has its own scale parameter (k) and the TV solver its
/// own regularization weight (lambda), so neither asks for intensity sigma;
/// unknown (custom) denoisers get the full parameter set.
DenoiseParams promptParams(TerminalInput& in, const std::string& filter,
                           const DenoiseParams& previous)
{
    std::cout << "Parameters (Enter accepts the bracketed default):\n";

    DenoiseParams params = previous;

    if (filter == "anisotropic") {
        params.num_iterations = promptInt(in, "PDE time steps", params.num_iterations);
        params.time_step = promptDouble(in, "time step dt (<= 0.25)", params.time_step);
        params.conduction_coefficient =
            promptDouble(in, "conduction coefficient k (edge sensitivity)", params.conduction_coefficient);
    } else if (filter == "tv") {
        params.lambda =
            promptDouble(in, "regularization lambda (larger = smoother)", params.lambda);
        params.num_iterations = promptInt(in, "projection steps", params.num_iterations);
        params.time_step = promptDouble(in, "step tau (<= 0.25)", params.time_step);
    } else {
        params.intensitySigma =
            promptDouble(in, "intensity sigma / h (noise scale)", params.intensitySigma);

        if (filter == "bilateral") {
            params.spatialSigma = promptDouble(in, "spatial sigma", params.spatialSigma);
        } else if (filter == "nlm") {
            params.searchWindow = promptInt(in, "search window side (odd)", params.searchWindow);
            params.patchSize = promptInt(in, "patch side (odd)", params.patchSize);
        } else if (filter == "dct") {
            params.blockSize = promptInt(in, "block size (8 or 16)", params.blockSize);
            params.threshold =
                promptDouble(in, "threshold multiplier (x intensity sigma)", params.threshold);
        } else {
            // Custom denoiser with unknown semantics: expose the full parameter set.
            params.spatialSigma = promptDouble(in, "spatial sigma", params.spatialSigma);
            params.searchWindow = promptInt(in, "search window side (odd)", params.searchWindow);
            params.patchSize = promptInt(in, "patch side (odd)", params.patchSize);
            params.blockSize = promptInt(in, "block size", params.blockSize);
            params.threshold = promptDouble(in, "threshold multiplier", params.threshold);
        }
    }

    params.iterations = promptInt(in, "iterations", params.iterations);
    return params;
}

/// Re-prompts until the parameters pass the library's validation.
///
/// The library validates per-algorithm, but the CLI does not know which
/// validator a (possibly custom) denoiser will apply, so the full parameter
/// set is checked here. Unused parameters keep valid defaults, and surfacing
/// mistakes before the run avoids wasting an expensive filtering pass.
DenoiseParams promptValidParams(TerminalInput& in, const std::string& filter,
                                const DenoiseParams& previous)
{
    while (!in.eof()) {
        const DenoiseParams params = promptParams(in, filter, previous);
        try {
            params.validate();
            params.validatePatchBased();
            params.validateTransform();
            params.validateDiffusion();
            params.validateVariational();
            return params;
        } catch (const InvalidParameterError& error) {
            std::cout << "  " << error.what() << " - please try again.\n";
        }
    }
    return previous;
}

/// Input "C:/dir/photo.png" with filter "nlm" -> "C:/dir/photo_nlm.png".
std::string defaultOutputPath(const std::string& inputPath, const std::string& filterName)
{
    const std::filesystem::path input(inputPath);
    std::filesystem::path output = input;
    output.replace_filename(input.stem().string() + "_" + filterName
                            + input.extension().string());
    return output.string();
}

} // namespace

int runInteractive()
{
    std::cout << "=== ImageDenoiser - interactive mode ===\n"
              << "(run with --help for non-interactive batch usage)\n\n";

    TerminalInput in;
    DenoiseParams lastParams; // tuned values carry over to the next run

    while (!in.eof()) {
        const std::string imagePath = in.readLine("Image path (Enter to quit): ");
        if (in.eof() || imagePath.empty()) {
            break;
        }

        const cv::Mat image = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
        if (image.empty()) {
            std::cout << "  could not load '" << imagePath << "' - try again.\n";
            continue;
        }
        std::cout << "  loaded " << describeImage(image) << '\n';

        // Work on a private copy so "run again" can re-pollute a pristine image.
        cv::Mat working = image.clone();
        const double noiseSigma = promptDouble(in, "Add Gaussian noise? sigma (0 = off)", 0.0);
        if (!in.eof() && noiseSigma > 0.0) {
            addGaussianNoise(working, noiseSigma);
            std::cout << "  added Gaussian noise with sigma " << noiseSigma << '\n';
        }
        if (in.eof()) {
            break;
        }

        bool workOnSameImage = true;
        while (workOnSameImage && !in.eof()) {
            const std::optional<std::string> choice = chooseFilterName(in);
            if (in.eof() || !choice.has_value()) {
                break;
            }

            if (*choice == kAllMethodsChoice) {
                // Concurrent run of every registered denoiser using the
                // current parameter set; results go to the output/ directory.
                std::string stem;
                std::string extension;
                splitInputPath(imagePath, stem, extension);
                const std::vector<RunResult> results =
                    runConcurrentSuite(working, lastParams, kOutputDirName, stem, extension);
                printResultsTable(results);
            } else {
                std::unique_ptr<ImageDenoiser> denoiser = DenoiserFactory::create(*choice);
                if (!denoiser) {
                    break;
                }

                const DenoiseParams params = promptValidParams(in, denoiser->name(), lastParams);
                if (in.eof()) {
                    break;
                }

                const std::string outputPath = promptText(
                    in, "Output path", defaultOutputPath(imagePath, denoiser->name()));
                if (in.eof()) {
                    break;
                }

                cv::Mat result;
                try {
                    const double elapsedMs = timedDenoise(*denoiser, working, result, params,
                                                          consoleProgress(denoiser->name()));
                    endProgressLine();
                    if (!cv::imwrite(outputPath, result)) {
                        throw DenoiserError("could not write result image '" + outputPath + "'");
                    }
                    lastParams = params;
                    std::cout << std::fixed << std::setprecision(1)
                              << "  denoised in " << elapsedMs << " ms -> " << outputPath << '\n';
                } catch (const std::exception& error) {
                    endProgressLine();
                    std::cout << "  error: " << error.what() << '\n';
                }
            }

            const std::string action = in.readLine(
                "Next: [r]un again on this image, [n]ew image, [q]uit (Enter = new image): ");
            if (in.eof()) {
                break;
            }
            if (action == "r" || action == "R" || action == "run") {
                continue; // same image, choose a different filter/parameters
            }
            if (action == "q" || action == "Q" || action == "quit") {
                std::cout << "Bye!\n";
                return 0;
            }
            workOnSameImage = false; // 'n', 'new' or just Enter
        }
    }

    std::cout << "Bye!\n";
    return 0;
}

} // namespace denoiser::cli
