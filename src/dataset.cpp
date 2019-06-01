#include "dataset.h"

namespace isai
{

  void dataset_t::print( bool is_normalized ) const
  {
    for ( auto &&dp : m_data )
    {
      std::printf( "[ %8.3f %8.3f %8.3f %8.3f ", dp.features[ 0 ],
                   dp.features[ 1 ], dp.features[ 2 ], dp.features[ 3 ] );
      if ( is_normalized )
      {
        std::printf( "%8.3f", dp.features[ 4 ] );
      }
      std::printf( " ] <- %s\n", label_to_string( dp.label ) );
    }
  }

  void dataset_t::load_from_file( char const *const path )
  {
    m_data.clear();
    m_data.reserve( 150 );

    auto fin = std::ifstream{ path, std::ios::in };

    auto line = std::string{};
    while ( std::getline( fin, line ) )
    {
      if ( line.empty() )
      {
        continue;
      }

      auto dp = data_point_t{};

      dp.features[ 0 ] = std::atof( line.substr( 0, 3 ).c_str() );   // NOLINT
      dp.features[ 1 ] = std::atof( line.substr( 4, 3 ).c_str() );   // NOLINT
      dp.features[ 2 ] = std::atof( line.substr( 8, 3 ).c_str() );   // NOLINT
      dp.features[ 3 ] = std::atof( line.substr( 12, 3 ).c_str() );  // NOLINT
      dp.features[ 4 ] = 0.0;

      auto label_str = line.substr( 16 );

      if ( label_str == "Iris-setosa" )
      {
        dp.label = label_t::setosa;
      }
      else if ( label_str == "Iris-versicolor" )
      {
        dp.label = label_t::versicolor;
      }
      else if ( label_str == "Iris-virginica" )
      {
        dp.label = label_t::virginica;
      }
      else
      {
        assert( false );
      }

      m_data.emplace_back( dp );
    }

    assert( size() == 150 );
  }

  void dataset_t::balance_signs()
  {
    auto avgs = std::accumulate( m_data.begin(), m_data.end(),
                                 std::array< double, 4 >{ 0.0, 0.0, 0.0, 0.0 },
                                 []( auto &&acc, auto &&dp ) {
                                   acc[ 0 ] += dp.features[ 0 ];
                                   acc[ 1 ] += dp.features[ 1 ];
                                   acc[ 2 ] += dp.features[ 2 ];
                                   acc[ 3 ] += dp.features[ 3 ];
                                   return acc;
                                 } );
    std::transform( avgs.begin(), avgs.end(), avgs.begin(),
                    [this]( auto avg_val ) {
                      return avg_val / static_cast< double >( size() );
                    } );
    for ( auto &&dp : m_data )
    {
      dp.features[ 0 ] -= avgs[ 0 ];
      dp.features[ 1 ] -= avgs[ 1 ];
      dp.features[ 2 ] -= avgs[ 2 ];
      dp.features[ 3 ] -= avgs[ 3 ];
    }
  }

  void normalize_stereographic( features_t &features, double radius )
  {
    auto rcoeff = 4.0 * radius * radius;
    auto sum = std::accumulate(
      features.begin(), features.end(), 0.0,
      []( auto acc, auto val ) { return acc + ( val * val ); } );
    auto den = rcoeff + sum;

    features[ 0 ] = ( rcoeff * features[ 0 ] ) / den;
    features[ 1 ] = ( rcoeff * features[ 1 ] ) / den;
    features[ 2 ] = ( rcoeff * features[ 2 ] ) / den;
    features[ 3 ] = ( rcoeff * features[ 3 ] ) / den;
    features[ 4 ] = ( sum - rcoeff ) / den;
  }

}  // namespace isai
