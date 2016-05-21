#pragma once

#include <mkl.h>

#include <blam/detail/config.h>

#include <blam/system/cblas/execution_policy.h>

// XXX TODO: Inherit from cblas_execution_policy for refined dispatch

namespace blam
{
// put the canonical tag in the same ns as the backend's entry points
namespace mkl
{

// this awkward sequence of definitions arise
// from the desire both for tag to derive
// from execution_policy and for execution_policy
// to convert to tag (when execution_policy is not
// an ancestor of tag)

// forward declaration of tag
struct tag;

// forward declaration of execution_policy
template <typename>
struct execution_policy;

// specialize execution_policy for tag
template <>
struct execution_policy<tag>
    : blam::cblas::execution_policy<tag>
{};

// tag's definition comes before the
// generic definition of execution_policy
struct tag : execution_policy<tag> {};

// allow conversion to tag when it is not a successor
template <typename Derived>
struct execution_policy
    : blam::cblas::execution_policy<Derived>
{
  // allow conversion to tag
  inline operator tag () const {
    return tag();
  }
};

static const tag par{};

} // end namespace mkl
} // end namespace blam

// Include algorithms
#include <blam/system/mkl/batch_gemm.h>
