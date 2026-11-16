#include <denoiser/DenoiserFactory.hpp>

#include <denoiser/AnisotropicDiffusionFilter.hpp>
#include <denoiser/BilateralFilter.hpp>
#include <denoiser/DctFilter.hpp>
#include <denoiser/Error.hpp>
#include <denoiser/NonLocalMeansFilter.hpp>
#include <denoiser/TotalVariationFilter.hpp>

#include <algorithm>
#include <cctype>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace denoiser {
namespace {

/// Lower-cases and trims an identifier so lookups are case- and whitespace-insensitive.
std::string normalize(std::string identifier)
{
    std::transform(identifier.begin(), identifier.end(), identifier.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    const auto first = identifier.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = identifier.find_last_not_of(" \t\r\n");
    return identifier.substr(first, last - first + 1);
}

using Registry = std::map<std::string, DenoiserFactory::Creator>;

/// Function-local static: immune to static-initialization-order fiasco.
Registry& registry()
{
    static Registry instance;
    return instance;
}

/// Registers the built-in denoisers exactly once.
void registerBuiltins()
{
    static const bool initialized = [] {
        registry().emplace("anisotropic", [] { return std::make_unique<AnisotropicDiffusionFilter>(); });
        registry().emplace("bilateral", [] { return std::make_unique<BilateralFilter>(); });
        registry().emplace("nlm", [] { return std::make_unique<NonLocalMeansFilter>(); });
        registry().emplace("dct", [] { return std::make_unique<DctFilter>(); });
        registry().emplace("tv", [] { return std::make_unique<TotalVariationFilter>(); });
        return true;
    }();
    (void)initialized;
}

std::string join(const std::vector<std::string>& parts, const std::string& separator)
{
    std::ostringstream out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i != 0) {
            out << separator;
        }
        out << parts[i];
    }
    return out.str();
}

} // namespace

std::unique_ptr<ImageDenoiser> DenoiserFactory::create(DenoiserType type)
{
    switch (type) {
    case DenoiserType::AnisotropicDiffusion:
        return std::make_unique<AnisotropicDiffusionFilter>();
    case DenoiserType::Bilateral:
        return std::make_unique<BilateralFilter>();
    case DenoiserType::NonLocalMeans:
        return std::make_unique<NonLocalMeansFilter>();
    case DenoiserType::Dct:
        return std::make_unique<DctFilter>();
    case DenoiserType::TotalVariation:
        return std::make_unique<TotalVariationFilter>();
    }
    throw UnknownDenoiserError("unknown built-in denoiser type");
}

std::unique_ptr<ImageDenoiser> DenoiserFactory::create(const std::string& identifier)
{
    registerBuiltins();

    const auto it = registry().find(normalize(identifier));
    if (it == registry().end()) {
        std::ostringstream message;
        message << "unknown denoiser identifier '" << identifier << "'; available filters: "
                << join(available(), ", ");
        throw UnknownDenoiserError(message.str());
    }
    return it->second();
}

bool DenoiserFactory::registerDenoiser(const std::string& identifier, Creator creator)
{
    if (!creator) {
        throw InvalidParameterError("denoiser creator must be callable");
    }
    registerBuiltins();
    return registry().emplace(normalize(identifier), std::move(creator)).second;
}

std::vector<std::string> DenoiserFactory::available()
{
    registerBuiltins();

    std::vector<std::string> identifiers;
    identifiers.reserve(registry().size());
    for (const auto& entry : registry()) {
        identifiers.push_back(entry.first);
    }
    return identifiers;
}

bool DenoiserFactory::isRegistered(const std::string& identifier)
{
    registerBuiltins();
    return registry().count(normalize(identifier)) != 0;
}

std::string DenoiserFactory::identifierFor(DenoiserType type)
{
    switch (type) {
    case DenoiserType::AnisotropicDiffusion:
        return "anisotropic";
    case DenoiserType::Bilateral:
        return "bilateral";
    case DenoiserType::NonLocalMeans:
        return "nlm";
    case DenoiserType::Dct:
        return "dct";
    case DenoiserType::TotalVariation:
        return "tv";
    }
    throw UnknownDenoiserError("unknown built-in denoiser type");
}

} // namespace denoiser
