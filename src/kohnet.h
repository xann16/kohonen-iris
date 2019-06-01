#pragma once

#ifndef ISAI_KOHRIS_KOHNET_H_INCLUDED
#define ISAI_KOHRIS_KOHNET_H_INCLUDED

#include "dataset.h"

#include <cmath>

namespace isai
{

  struct knc_settings_t
  {
    std::size_t hidden_layer_size = 10000u;
    std::size_t training_set_size = 100u;
    std::size_t expected_cluster_count = 3u;

    double normalization_sphere_radius = 4.0;
    double alpha = 0.3;
    std::size_t coalesce_interval = 10u;

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

    features_t const &weights() const { return m_weights; }

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

    void average_with( kohonen_neuron_t const &other )
    {
      for ( auto i = std::size_t{ 0 }; i < m_weights.size(); i++ )
      {
        m_weights[ i ] += ( m_weights[ i ] + other.m_weights[ i ] ) / 2.0;
      }
      normalize();
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
      m_iteration_no( 0u ),
      m_alive_count( 0u ),
      m_kill_count( 0u ),
      m_coalesce_count( 0u ),
      m_settings( settings )
    {
      m_hidden_layer.reserve( m_settings.hidden_layer_size );
      m_statuses.reserve( m_settings.hidden_layer_size );
      for ( auto i = std::size_t{ 0 }; i < m_settings.hidden_layer_size; i++ )
      {
        m_hidden_layer.emplace_back( m_settings.normalization_sphere_radius );
        m_statuses.emplace_back( 0 );
        m_alive_count++;
      }
      assert( m_alive_count == m_settings.hidden_layer_size );
    }

    template < typename Iterator >
    void run( Iterator begin, Iterator end )
    {
      while ( !is_completed() )
      {
        prepare();
        for ( auto i = begin; i != end; i++ )
        {
          process_input( ( *i ).features );
        }
        kill();
        coalesce();
        print_status();
      }
    }

    auto get_results() const
    {
      auto res = std::vector< kohonen_neuron_t >{};
      for ( auto i = std::size_t{ 0 }; i < size(); i++ )
      {
        if ( is_alive( i ) )
        {
          res.emplace_back( m_hidden_layer[ i ] );
        }
      }

      assert( res.size() == m_settings.expected_cluster_count );

      return res;
    }

  private:
    void prepare()
    {
      m_kill_count = 0u;
      m_coalesce_count = 0u;
      m_iteration_no++;

      auto sum = 0;

      for ( auto &&s : m_statuses )
      {
        if ( s > 0 )
        {
          sum += s;
          s = 0;
        }
      }

      assert( sum = static_cast< int >( m_settings.training_set_size ) );
    }

    std::size_t find_winner( features_t const &input ) const
    {
      auto best_ix = std::size_t{ 0 };
      auto best_val = std::numeric_limits< double >::max();

      for ( auto i = std::size_t{ 0 }; i < size(); i++ )
      {
        auto sqr_distance = m_hidden_layer[ i ].sqr_distance_to( input );
        if ( is_alive( i ) && sqr_distance < best_val )
        {
          best_ix = i;
          best_val = sqr_distance;
        }
      }

      return best_ix;
    }

    void process_input( features_t const &input )
    {
      auto winner_ix = find_winner( input );
      m_hidden_layer[ winner_ix ].adjust_to( input );
      m_statuses[ winner_ix ]++;
    }

    void kill()
    {
      assert( m_kill_count == 0 );
      for ( auto &&s : m_statuses )
      {
        if ( s == 0 )
        {
          s = -1;
          m_kill_count++;
          m_alive_count--;
          if ( is_completed() )
          {
            break;
          }
        }
      }
    }

    void coalesce()
    {
      if ( is_completed() ||
           m_iteration_no % m_settings.coalesce_interval != 0 )
      {
        return;
      }

      auto best_i = std::size_t{ 0 };
      auto best_j = std::size_t{ 0 };
      auto best_val = std::numeric_limits< double >::min();

      for ( auto i = std::size_t{ 0 }; i < size(); i++ )
      {
        if ( !is_alive( i ) )
        {
          continue;
        }

        for ( auto j = i + 1; j < size(); j++ )
        {
          auto dot_prod =
            m_hidden_layer[ i ].weights() * m_hidden_layer[ j ].weights();
          if ( is_alive( j ) && dot_prod > best_val )
          {
            best_i = i;
            best_j = j;
            best_val = dot_prod;
          }
        }
      }
      coalesce( best_i, best_j );
    }

    void coalesce( std::size_t i, std::size_t j )
    {
      m_hidden_layer[ j ].average_with( m_hidden_layer[ i ] );
      m_statuses[ i ] = -1;
      m_coalesce_count++;
      m_alive_count--;
    }

    void print_status() const
    {
      auto perc = ( static_cast< double >( m_alive_count ) /
                    static_cast< double >( size() ) ) *
                  100.0;
      std::printf( "ITERATION #%03lu - live neurons remaining: %lu/%lu "
                   "(%.2f%%) [killed: %lu, coalesced: %lu]\n",
                   m_iteration_no, m_alive_count, size(), perc, m_kill_count,
                   m_coalesce_count );
    }

    bool is_completed()
    {
      assert( m_alive_count >= m_settings.expected_cluster_count );
      return m_alive_count == m_settings.expected_cluster_count;
      // return m_iteration_no == 100u;
    }

    bool is_alive( std::size_t index ) const
    {
      return m_statuses[ index ] >= 0;
    }

    std::size_t size() const noexcept { return m_hidden_layer.size(); }

  private:
    std::vector< kohonen_neuron_t > m_hidden_layer =
      std::vector< kohonen_neuron_t >{};
    std::vector< int > m_statuses = std::vector< int >{};

    std::size_t m_iteration_no;
    std::size_t m_alive_count;
    std::size_t m_kill_count;
    std::size_t m_coalesce_count;

    knc_settings_t m_settings;
  };

}  // namespace isai

#endif  // !ISAI_KOHRIS_KOHNET_H_INCLUDED
