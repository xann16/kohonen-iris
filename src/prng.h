#pragma once

#ifndef ISAI_KOHRIS_PRNG_H_INCLUDED
#define ISAI_KOHRIS_PRNG_H_INCLUDED

#include <algorithm>
#include <array>
#include <cassert>
#include <random>
#include <vector>

namespace isai
{

  // random number generation utils
  class prng_t
  {
  private:
    prng_t() noexcept = default;

  public:
    // initializes prng device
    static void initialize() noexcept
    {
      s_eng = std::default_random_engine{ s_dev() };
    }

    // probability [0,1] to binary success/failure
    static bool perc_check( double perc ) noexcept
    {
      assert( perc >= 0.0 );
      assert( perc <= 1.0 );
      return perc >= std::generate_canonical< double, 64 >( s_eng );
    }

    // shuffles elements of given vector
    template < typename T >
    static void shuffle( std::vector< T > &v )
    {
      // std::random_shuffle( std::begin( v ), std::end( v ) );
      std::shuffle( std::begin( v ), std::end( v ), s_eng );
    }

    // random array to be used as initial weighs of hidden layer
    static auto init_neuron_weights( std::array< double, 5 > &weights )
    {
      weights[ 0 ] = get_plus_minus_one();
      weights[ 1 ] = get_plus_minus_one();
      weights[ 2 ] = get_plus_minus_one();
      weights[ 3 ] = get_plus_minus_one();
      weights[ 4 ] = 0.0;
    }

  private:
    static double get_plus_minus_one()
    {
      return std::generate_canonical< double, 64 >( s_eng ) * 2.0 - 1.0;
    }

    static std::random_device s_dev;
    static std::default_random_engine s_eng;
  };

}  // namespace isai

#endif  // !ISAI_GENEPI_POLY_H_INCLUDED
