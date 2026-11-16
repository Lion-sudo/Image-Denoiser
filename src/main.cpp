#include "CliCommon.hpp"
#include "ConcurrentRunner.hpp"
#include "InteractiveSession.hpp"

#include <denoiser/DenoiserFactory.hpp>
#include <denoiser/Error.hpp>
#include <denoiser/ImageDenoiser.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string inputPath;
    std::string outputPath = "denoised.png";
    std::string filter = "bilateral";
    denoiser::DenoiseParams params;
    double noiseSigma = 0.0; ///< Optional synthetic Gaussian noise, in native intensity units.
    bool allMethods = false; ///< Run every registered denoiser concurrently (--method all).
    bool interactive = false;
    bool listFilters = false;
    bool showHelp = false;
};

void printUsage(std::ostream& out, const std::string& programName)
{
    out << "ImageDenoiser - modular image denoising CLI\n\n"
        << "Usage: " << programName << " [options]\n"
        << "       " << programName << "            (no arguments: interactive session)\n\n"
        << "Options:\n"
        << "  -i, --input <path>        Image to denoise (PNG, JPEG, BMP, TIFF, ...).\n"
        << "  -o, --output <path>       Destination image (default: denoised.png).\n"
        << "  -f, --filter <name>       Denoiser identifier, see --list (default: bilateral).\n"
        << "      --method <name>       Alias for --filter, or 'all' to run every registered\n"
        << "                            denoiser concurrently with live progress; results\n"
        << "                            are written into the '" << denoiser::cli::kOutputDirName
        << "/' directory.\n"
        << "      --spatial-sigma <v>   Spatial Gaussian sigma (default: 3.0).\n"
        << "      --intensity-sigma <v> Intensity sigma / h, in [0,1] units (default: 0.1).\n"
        << "      --search-window <n>   NLM search window side, odd (default: 21).\n"
        << "      --patch-size <n>      NLM patch side, odd (default: 7).\n"
        << "      --block-size <n>      DCT block side (default: 8).\n"
        << "      --threshold <v>       DCT soft-threshold multiplier (default: 2.7);\n"
        << "                            per-coefficient tau = threshold * intensity-sigma.\n"
        << "      --num-iterations <n>  Iterative steps: PDE time steps (anisotropic) or\n"
        << "                            Chambolle projection steps (tv) (default: 30).\n"
        << "      --time-step <v>       Explicit step dt (anisotropic) / tau (tv),\n"
        << "                            must be <= 0.25 (default: 0.15).\n"
        << "      --conduction-coefficient <v>\n"
        << "                            Perona-Malik edge sensitivity k (default: 0.15).\n"
        << "      --lambda <v>          TV regularization weight (default: 0.1);\n"
        << "                            larger values give smoother results.\n"
        << "      --iterations <n>      Number of filtering passes (default: 1).\n"
        << "      --noise <sigma>       Add synthetic Gaussian noise before denoising;\n"
        << "                            sigma is in native intensity units (0-255 for\n"
        << "                            8-bit images, [0,1] for float images).\n"
        << "  -I, --interactive         Start the menu-driven interactive session.\n"
        << "  -l, --list                List the registered denoisers and exit.\n"
        << "  -h, --help                Show this help message and exit.\n";
}

double toDouble(const std::string& text, const std::string& option)
{
    try {
        std::size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed != text.size()) {
            throw denoiser::InvalidParameterError("option '" + option + "' expects a number, got '"
                                                  + text + "'");
        }
        return value;
    } catch (const std::invalid_argument&) {
        throw denoiser::InvalidParameterError("option '" + option + "' expects a number, got '"
                                              + text + "'");
    } catch (const std::out_of_range&) {
        throw denoiser::InvalidParameterError("option '" + option + "' value '" + text
                                              + "' is out of range");
    }
}

int toInt(const std::string& text, const std::string& option)
{
    try {
        std::size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (consumed != text.size()) {
            throw denoiser::InvalidParameterError("option '" + option + "' expects an integer, got '"
                                                  + text + "'");
        }
        return value;
    } catch (const std::invalid_argument&) {
        throw denoiser::InvalidParameterError("option '" + option + "' expects an integer, got '"
                                              + text + "'");
    } catch (const std::out_of_range&) {
        throw denoiser::InvalidParameterError("option '" + option + "' value '" + text
                                              + "' is out of range");
    }
}

Options parseArguments(int argc, char* argv[])
{
    const std::vector<std::string> args(argv + (argc > 0 ? 1 : 0), argv + (argc > 0 ? argc : 0));

    Options options;
    auto valueFor = [&args](std::size_t& index, const std::string& option) {
        if (index + 1 >= args.size()) {
            throw denoiser::InvalidParameterError("missing value for option '" + option + "'");
        }
        return args[++index];
    };

    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "-h" || arg == "--help") {
            options.showHelp = true;
        } else if (arg == "-l" || arg == "--list") {
            options.listFilters = true;
        } else if (arg == "-I" || arg == "--interactive") {
            options.interactive = true;
        } else if (arg == "-i" || arg == "--input") {
            options.inputPath = valueFor(i, arg);
        } else if (arg == "-o" || arg == "--output") {
            options.outputPath = valueFor(i, arg);
        } else if (arg == "-f" || arg == "--filter") {
            options.filter = valueFor(i, arg);
        } else if (arg == "--method") {
            const std::string method = valueFor(i, arg);
            if (method == "all") {
                options.allMethods = true;
            } else {
                options.filter = method; // convenience alias for --filter
            }
        } else if (arg == "--spatial-sigma") {
            options.params.spatialSigma = toDouble(valueFor(i, arg), arg);
        } else if (arg == "--intensity-sigma") {
            options.params.intensitySigma = toDouble(valueFor(i, arg), arg);
        } else if (arg == "--search-window") {
            options.params.searchWindow = toInt(valueFor(i, arg), arg);
        } else if (arg == "--patch-size") {
            options.params.patchSize = toInt(valueFor(i, arg), arg);
        } else if (arg == "--block-size") {
            options.params.blockSize = toInt(valueFor(i, arg), arg);
        } else if (arg == "--threshold") {
            options.params.threshold = toDouble(valueFor(i, arg), arg);
        } else if (arg == "--num-iterations") {
            options.params.num_iterations = toInt(valueFor(i, arg), arg);
        } else if (arg == "--time-step") {
            options.params.time_step = toDouble(valueFor(i, arg), arg);
        } else if (arg == "--conduction-coefficient") {
            options.params.conduction_coefficient = toDouble(valueFor(i, arg), arg);
        } else if (arg == "--lambda") {
            options.params.lambda = toDouble(valueFor(i, arg), arg);
        } else if (arg == "--iterations") {
            options.params.iterations = toInt(valueFor(i, arg), arg);
        } else if (arg == "--noise") {
            options.noiseSigma = toDouble(valueFor(i, arg), arg);
        } else {
            throw denoiser::InvalidParameterError("unknown option '" + arg
                                                  + "' (use --help for usage)");
        }
    }
    return options;
}

/// Non-interactive single-shot pipeline: load -> (optional noise) -> denoise -> save.
int runBatch(const Options& options)
{
    if (options.inputPath.empty()) {
        throw denoiser::InvalidParameterError("no input image specified (use --input <path>, "
                                              "--interactive, or --help for usage)");
    }

    const cv::Mat image = cv::imread(options.inputPath, cv::IMREAD_UNCHANGED);
    if (image.empty()) {
        throw denoiser::DenoiserError("could not load image '" + options.inputPath + "'");
    }

    cv::Mat noisyImage = image.clone();
    if (options.noiseSigma > 0.0) {
        denoiser::cli::addGaussianNoise(noisyImage, options.noiseSigma);
    }

    // Multi-method mode: dispatch every registered denoiser concurrently with
    // live progress feedback. -o does not apply here; each result goes to
    // <kOutputDirName>/<stem>_<filter>.<ext>.
    if (options.allMethods) {
        std::string stem;
        std::string extension;
        denoiser::cli::splitInputPath(options.inputPath, stem, extension);

        const std::vector<denoiser::cli::RunResult> results =
            denoiser::cli::runConcurrentSuite(noisyImage, options.params,
                                              denoiser::cli::kOutputDirName, stem, extension);
        denoiser::cli::printResultsTable(results);

        return std::all_of(results.begin(), results.end(),
                           [](const denoiser::cli::RunResult& result) {
                               return result.success;
                           })
                   ? 0
                   : 1;
    }

    auto denoiser = denoiser::DenoiserFactory::create(options.filter);

    cv::Mat denoised;
    const double elapsedMs = denoiser::cli::timedDenoise(
        *denoiser, noisyImage, denoised, options.params,
        denoiser::cli::consoleProgress(denoiser->name()));
    denoiser::cli::endProgressLine();

    if (!cv::imwrite(options.outputPath, denoised)) {
        throw denoiser::DenoiserError("could not write result image '" + options.outputPath + "'");
    }

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[denoiser] input   : " << options.inputPath << " ("
              << denoiser::cli::describeImage(noisyImage) << ")\n"
              << "[denoiser] filter  : " << denoiser->name() << "\n"
              << "[denoiser] elapsed : " << elapsedMs << " ms\n"
              << "[denoiser] output  : " << options.outputPath << '\n';

    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    try {
        // Launched bare (e.g. double-clicked or without options): be friendly
        // and open the interactive session instead of failing on missing --input.
        if (argc <= 1) {
            return denoiser::cli::runInteractive();
        }

        const Options options = parseArguments(argc, argv);

        if (options.showHelp) {
            printUsage(std::cout, argc > 0 ? argv[0] : "denoiser-cli");
            return 0;
        }

        if (options.listFilters) {
            denoiser::cli::printAvailableFilters(std::cout);
            return 0;
        }

        if (options.interactive) {
            return denoiser::cli::runInteractive();
        }

        return runBatch(options);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
