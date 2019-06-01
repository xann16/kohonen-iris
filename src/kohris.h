#pragma once

#ifndef ISAI_KOHRIS_KOHRIS_H_INCLUDED
#define ISAI_KOHRIS_KOHRIS_H_INCLUDED

#include "dataset.h"
#include "kohnet.h"

namespace isai
{

  class iris_clusterizer_t
  {
  public:
    explicit iris_clusterizer_t( knc_settings_t const &settings ) :
      m_settings( settings ),
      m_dataset( settings.training_set_size,
                 settings.normalization_sphere_radius,
                 settings.is_feature_sign_balanced ),
      m_solver( settings )
    {
    }

    void run()
    {
      std::printf( "Training started with following parameters:\n" );
      print_settings();

      m_solver.run( m_dataset.train_begin(), m_dataset.train_end() );
      std::printf( "Training completed.\n\n" );

      evaluate( m_dataset.test_begin(), m_dataset.test_end(),
                m_solver.get_results() );
    }

  private:
    template < typename Iterator >
    void evaluate( Iterator begin, Iterator end,
                   std::vector< kohonen_neuron_t > const &results )
    {
      // clusterize test data according to given results
      auto labels = std::vector< std::pair< std::size_t, std::size_t > >{};

      for ( auto i = begin; i != end; i++ )
      {
        auto best_label = std::size_t{ 0 };
        auto best_dist_sqr = results[ 0 ].sqr_distance_to( ( *i ).features );

        for ( auto j = std::size_t{ 1 }; j < results.size(); j++ )
        {
          auto dist_sqr = results[ j ].sqr_distance_to( ( *i ).features );
          if ( dist_sqr < best_dist_sqr )
          {
            best_dist_sqr = dist_sqr;
            best_label = j;
          }
        }

        auto act_label = static_cast< std::size_t >( ( *i ).label );

        assert( best_label >= 0 && best_label < 3 );
        assert( act_label >= 0 && act_label < 3 );

        labels.emplace_back( act_label, best_label );
      }

      // create cross reference table
      std::size_t crt[ 3 ][ 3 ];
      for ( auto i = 0; i < 9; i++ )
      {
        crt[ i / 3 ][ i % 3 ] = 0u;
      }

      for ( auto &&lp : labels )
      {
        crt[ lp.first ][ lp.second ]++;
      }

      // print crt
      std::printf( "*----------------------*------*------*------*\n" );
      std::printf( "|  actual \\ predicted  |  #1  |  #2  |  #3  |\n" );
      std::printf( "*----------------------*------*------*------*\n" );
      std::printf( "| #1 - %15s | %4lu | %4lu | %4lu |\n", iris_label_strs[ 0 ],
                   crt[ 0 ][ 0 ], crt[ 0 ][ 1 ], crt[ 0 ][ 2 ] );
      std::printf( "*----------------------*------*------*------*\n" );
      std::printf( "| #2 - %15s | %4lu | %4lu | %4lu |\n", iris_label_strs[ 1 ],
                   crt[ 1 ][ 0 ], crt[ 1 ][ 1 ], crt[ 1 ][ 2 ] );
      std::printf( "*----------------------*------*------*------*\n" );
      std::printf( "| #3 - %15s | %4lu | %4lu | %4lu |\n", iris_label_strs[ 2 ],
                   crt[ 2 ][ 0 ], crt[ 2 ][ 1 ], crt[ 2 ][ 2 ] );
      std::printf( "*----------------------*------*------*------*\n" );

      // tbc...
    }

    void print_settings()
    {
      std::printf( " - training set size:                              %3lu\n",
                   m_settings.training_set_size );
      std::printf( " - initial no of neurons:                          %3lu\n",
                   m_settings.hidden_layer_size );

      std::printf( " - radius of normalization sphere:                 %6.2f\n",
                   m_settings.normalization_sphere_radius );
      std::printf( " - initial radius of input influence:              %6.2f\n",
                   m_settings.init_radius_threshold );
      std::printf( " - alpha (neuron adjustment strength):             %6.2f\n",
                   m_settings.alpha );
      std::printf( " - beta (radius of input influence reduction):     %6.2f\n",
                   m_settings.beta );
      std::printf( " - ratio of neurons killed per iteration:          %6.2f\n",
                   m_settings.kill_perc );
      std::printf( " - minimal number of neurons killed per iteration: %3lu\n",
                   m_settings.kill_min );
      std::puts( "" );
    }


  private:
    knc_settings_t m_settings;
    dataset_t m_dataset;
    kohonen_network_t m_solver;
  };


}  // namespace isai

#endif  // !ISAI_KOHRIS_KOHNET_H_INCLUDED
