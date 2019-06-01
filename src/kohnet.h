#pragma once

#ifndef ISAI_KOHRIS_KOHNET_H_INCLUDED
#define ISAI_KOHRIS_KOHNET_H_INCLUDED

#include "dataset.h"

#include <cmath>

namespace isai
{

  struct knc_settings_t
  {
    std::size_t hidden_layer_size = 30u;
    std::size_t training_set_size = 100u;
    std::size_t expected_cluster_count = 3u;

    double normalization_sphere_radius = 1.0;
    double init_radius_threshold = 1.0;
    double alpha = 0.2;
    double beta = 0.2;
    double kill_perc = 0.1;
    std::size_t kill_min = 3u;

    bool is_feature_sign_balanced = true;
  };

  class kohonen_neuron_t
  {
  public:
    explicit kohonen_neuron_t( double radius = 1.0 )
    {
      prng_t::init_neuron_weights( m_weights );
      normalize_stereographic( m_weights, radius );
    }

    double operator[]( std::size_t pos ) const { return m_weights[ pos ]; }
    double &operator[]( std::size_t pos ) { return m_weights[ pos ]; }
    auto begin() const { m_weights.begin(); }
    auto end() const { m_weights.end(); }

    features_t const &weights() const noexcept { return m_weights; }

    void normalize()
    {
      auto n = norm();
      for ( auto &&w : m_weights )
      {
        w /= n;
      }
    }

    double sqr_distance_to( features_t const &other ) const
    {
      assert( other.size() == m_weights.size() );
      auto res = 0.0;
      for ( auto i = std::size_t{ 0 }; i < m_weights.size(); i++ )
      {
        res +=
          ( other[ i ] - m_weights[ i ] ) * ( other[ i ] - m_weights[ i ] );
      }
      return res;
    }

    double distance_to( features_t const &other ) const
    {
      return std::sqrt( sqr_distance_to( other ) );
    }

    void adjust_to( features_t const &other, double alpha = 0.2 )
    {
      for ( auto i = std::size_t{ 0 }; i < m_weights.size(); i++ )
      {
        m_weights[ i ] += alpha * ( other[ i ] - m_weights[ i ] );
      }
      normalize();
    }

    double adjust_to_ex( features_t const &other, double alpha = 0.2 )
    {
      auto prev = m_weights;
      adjust_to( other, alpha );
      return sqr_distance_to( prev );
    }

  private:
    double norm()
    {
      auto sqrsum = std::accumulate(
        m_weights.begin(), m_weights.end(), 0.0,
        []( auto &&acc, auto &&val ) { return acc + ( val * val ); } );
      return std::sqrt( sqrsum );
    }

  private:
    features_t m_weights = features_t{};
  };

  // stores iris dataset and provides basic helper functionalities
  class kohonen_network_t
  {
  public:
    explicit kohonen_network_t( knc_settings_t const &settings ) :
      m_radius_threshold( settings.init_radius_threshold ),
      m_iteration_no( 0u ),
      m_settings( settings )
    {
      m_hidden_layer.reserve( m_settings.hidden_layer_size );
      m_statuses.reserve( m_settings.hidden_layer_size );
      for ( auto i = std::size_t{ 0 }; i < m_settings.hidden_layer_size; i++ )
      {
        m_hidden_layer.emplace_back( m_settings.normalization_sphere_radius );
        m_statuses.emplace_back( 0.0 );
      }
    }

    template < typename Iterator >
    void run( Iterator begin, Iterator end )
    {
      while ( !is_completed() )
      {
        clear_statuses();
        for ( auto i = begin; i != end; i++ )
        {
          process_input( ( *i ).features );
        }
        kill_lazy();
        adjust_radius_threshold();
        m_iteration_no++;
        print_status();
      }
    }

    auto get_results() const
    {
      auto res = std::vector< kohonen_neuron_t >{};
      for ( auto i = std::size_t{ 0 }; i < size(); i++ )
      {
        if ( m_statuses[ i ] >= 0.0 )
        {
          res.emplace_back( m_hidden_layer[ i ] );
        }
      }

      assert( res.size() == m_settings.expected_cluster_count );

      return res;
    }

  private:
    void clear_statuses()
    {
      for ( auto &&s : m_statuses )
      {
        if ( s > 0.0 )
        {
          s = 0.0;
        }
      }
    }

    void process_input( features_t const &input )
    {
      auto sqr_radius_threshold = m_radius_threshold * m_radius_threshold;
      for ( auto i = std::size_t{ 0 }; i < size(); i++ )
      {
        if ( m_statuses[ i ] >= 0.0 && m_hidden_layer[ i ].sqr_distance_to(
                                         input ) < sqr_radius_threshold )
        {
          auto sqr_diff_dist =
            m_hidden_layer[ i ].adjust_to_ex( input, m_settings.alpha );
          if ( sqr_diff_dist > m_statuses[ i ] )
          {
            m_statuses[ i ] = sqr_diff_dist;
          }
        }
      }
    }

    void kill_lazy()
    {
      auto alive = alive_count();
      auto remaining = alive;

      auto to_kill =
        std::max( static_cast< std::size_t >( alive * m_settings.kill_perc ),
                  m_settings.kill_min );

      auto victims = kill_untouched( alive );
      remaining -= victims;

      while ( victims < to_kill &&
              remaining > m_settings.expected_cluster_count )
      {
        kill_worst();
        victims++;
        remaining--;
      }
    }

    std::size_t kill_untouched( std::size_t alive_count )
    {
      auto victim_count = std::size_t{ 0 };
      for ( auto &&s : m_statuses )
      {
        if ( s == 0.0 )
        {
          s = -1.0;
          victim_count++;
          if ( alive_count - victim_count <= m_settings.expected_cluster_count )
          {
            break;
          }
        }
      }
      return victim_count;
    }

    std::size_t alive_count() const
    {
      auto res = std::size_t{ 0 };
      for ( auto &&s : m_statuses )
      {
        if ( s >= 0.0 )
        {
          res++;
        }
      }
      return res;
    }

    void kill_worst()
    {
      auto max_ix = std::size_t{ 0 };
      auto max_val = m_statuses[ 0 ];

      for ( auto i = std::size_t{ 1 }; i < size(); i++ )
      {
        if ( m_statuses[ i ] > max_val )
        {
          max_val = m_statuses[ i ];
          max_ix = i;
        }
      }
      m_statuses[ max_ix ] = -1.0;
    }


    /*
    void kill_worst()
    {
      auto min_ix = std::size_t{ 0 };
      auto min_val = std::numeric_limits< double >::max();

      for ( auto i = std::size_t{ 0 }; i < size(); i++ )
      {
        if ( m_statuses[ i ] >= 0.0 && m_statuses[ i ] < min_val )
        {
          min_val = m_statuses[ i ];
          min_ix = i;
        }
      }
      m_statuses[ min_ix ] = -1.0;
    }
    */

    void print_status() const
    {
      auto alive = alive_count();
      auto perc =
        ( static_cast< double >( alive ) / static_cast< double >( size() ) ) *
        100.0;
      std::printf(
        "ITERATION #%03lu - live neurons remaining: %3lu/%3lu (%6.2f%%)\n",
        m_iteration_no, alive, size(), perc );
    }

    void adjust_radius_threshold()
    {
      m_radius_threshold *= ( 1.0 - m_settings.beta );
    }

    bool is_completed()
    {
      return alive_count() <= m_settings.expected_cluster_count;
      // return m_iteration_no == 100u;
    }

    std::size_t size() const noexcept { return m_hidden_layer.size(); }

  private:
    std::vector< kohonen_neuron_t > m_hidden_layer =
      std::vector< kohonen_neuron_t >{};
    std::vector< double > m_statuses = std::vector< double >{};

    double m_radius_threshold;
    std::size_t m_iteration_no;

    knc_settings_t m_settings;
  };

}  // namespace isai

#endif  // !ISAI_KOHRIS_KOHNET_H_INCLUDED
