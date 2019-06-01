#include "prng.h"

namespace isai
{
  std::random_device prng_t::s_dev = std::random_device{};  // NOLINT
  std::default_random_engine prng_t::s_eng =                // NOLINT
    std::default_random_engine{};                           // NOLINT
}  // namespace isai
