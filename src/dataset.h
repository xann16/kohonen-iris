#pragma once

#ifndef ISAI_KOHRIS_DATASET_H_INCLUDED
#define ISAI_KOHRIS_DATASET_H_INCLUDED

#include "prng.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <vector>

namespace isai
{

  // types of iris (labels)
  enum class label_t
  {
    setosa,
    versicolor,
    virginica
  };

  // human-readable labels
  constexpr char const *const iris_label_strs[] = { "iris setosa",
                                                    "iris versicolor",
                                                    "iris virginica" };
  constexpr char const *label_to_string( label_t label )
  {
    return iris_label_strs[ static_cast< int >( label ) ];
  }

  // array of features
  using features_t = std::array< double, 5 >;

  // point of data from iris dataset
  struct data_point_t
  {
    features_t features;
    label_t label;
  };

  // general utility stereographic normalization
  void normalize_stereographic( features_t &features, double radius );

  // stores iris dataset and provides basic helper functionalities
  class dataset_t
  {
  public:
    // basic constructor
    explicit dataset_t( std::size_t training_count,
                        double proj_sphere_radius = 1.0,
                        bool do_sign_balancing = false ) :
      m_training_count( training_count )
    {
      // load data from file
      load_from_file( "data/iris.csv" );

      // preprocessing
      if ( do_sign_balancing )
      {
        balance_signs();
      }
      normalize( proj_sphere_radius );

      // randomly split to training and test sets
      prng_t::shuffle( m_data );
    }

    // default copy/move constructors/assignments
    dataset_t( dataset_t const & ) = default;
    dataset_t( dataset_t && ) noexcept = default;
    dataset_t &operator=( dataset_t const & ) = default;
    dataset_t &operator=( dataset_t && ) noexcept = default;

    // size and iterators for whole dataset
    std::size_t size() const noexcept { return m_data.size(); }
    auto begin() const noexcept { return m_data.begin(); }
    auto end() const noexcept { return m_data.end(); }

    // size and iterators for training set
    std::size_t train_size() const noexcept { return m_training_count; }
    auto train_begin() const noexcept { return m_data.begin(); }
    auto train_end() const noexcept
    {
      return m_data.begin() +
             static_cast< typename std::vector<
               data_point_t >::iterator::difference_type >( train_size() );
    }

    // size and iterators for test set
    std::size_t test_size() const noexcept
    {
      return m_data.size() - m_training_count;
    }
    auto test_begin() const noexcept
    {
      return m_data.begin() +
             static_cast< typename std::vector<
               data_point_t >::iterator::difference_type >( train_size() );
    }
    auto test_end() const noexcept { return m_data.end(); }

    // debug print
    void print( bool is_normalized = true ) const;

  private:
    // loads iris dataset form file
    void load_from_file( char const *path );

    // normalizes each featurre vector using stereographic projection
    void normalize( double radius )
    {
      for ( auto &&dp : m_data )
      {
        normalize_stereographic( dp.features, radius );
      }
    }

    // balance values to have both signs
    void balance_signs();

  private:
    std::vector< data_point_t > m_data = std::vector< data_point_t >{};
    std::size_t m_training_count;
  };

}  // namespace isai

#endif  // !ISAI_KOHRIS_DATASET_H_INCLUDED
