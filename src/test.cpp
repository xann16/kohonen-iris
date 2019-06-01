#include "kohris.h"

#include <iostream>

int main()
{
  isai::prng_t::initialize();
  auto settings = isai::knc_settings_t{};
  auto clusterizer = isai::iris_clusterizer_t{ settings };
  clusterizer.run();
  return 0;
}
