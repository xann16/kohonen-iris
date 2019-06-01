#include "kohris.h"

#include <iostream>

int main()
{
  isai::prng_t::initialize();

  auto settings = isai::knc_settings_t{};

  auto clusterizer = isai::iris_clusterizer_t{ settings };

  clusterizer.run();



  // auto ds = isai::dataset_t{ 100u, 1.0, true };
  // ds.print();

  return 0;
}
