#ifndef BLAS_UTIL_HH
#define BLAS_UTIL_HH

#include <complex>
#include <cstdarg>

#include <cassert>

namespace blam {

// -----------------------------------------------------------------------------
/// Exception class for BLAS errors.
class Error: public std::exception {
public:
    /// Constructs BLAS error
    Error():
        std::exception()
    {}

    /// Constructs BLAS error with message
    Error( std::string const& msg ):
        std::exception(),
        msg_( msg )
    {}

    /// Constructs BLAS error with message: "msg, in function func"
    Error( const char* msg, const char* func ):
        std::exception(),
        msg_( std::string(msg) + ", in function " + func )
    {}

    /// Returns BLAS error message
    virtual const char* what() const noexcept override
        { return msg_.c_str(); }

private:
    std::string msg_;
};

// -----------------------------------------------------------------------------
//  traits
/// Given a type, defines corresponding real and complex types.
/// E.g., for float,          real_t = float, complex_t = std::complex<float>,
///       for complex<float>, real_t = float, complex_t = std::complex<float>.

template< typename T >
class traits
{
public:
    typedef T real_t;
    typedef std::complex<T> complex_t;
};

// ----------------------------------------
template< typename T >
class traits< std::complex<T> >
{
public:
    typedef T real_t;
    typedef std::complex<T> complex_t;
};

// -----------------------------------------------------------------------------
//  traits2
/// Given two types, defines scalar and real types compatible with both types.
/// E.g., for pair (float, complex<float>),
/// scalar_t = complex<float>, real_t = float.

// By default, scalars and reals are T1.
// Later classes specialize if it should be T2 or something else
template< typename T1, typename T2 >
class traits2
{
public:
    typedef T1 scalar_t;
    typedef T1 real_t;
};

// ----------------------------------------
// int
template<>
class traits2< int, int64_t >
{
public:
    typedef int64_t scalar_t;
    typedef int64_t real_t;
};

// ---------------
template<>
class traits2< int, float >
{
public:
    typedef float scalar_t;
    typedef float real_t;
};

// ---------------
template<>
class traits2< int, double >
{
public:
    typedef double scalar_t;
    typedef double real_t;
};

// ----------------------------------------
// float
template<>
class traits2< float, double >
{
public:
    typedef double scalar_t;
    typedef double real_t;
};

// ---------------
template<>
class traits2< float, std::complex<float> >
{
public:
    typedef std::complex<float> scalar_t;
    typedef float real_t;
};

// ---------------
template<>
class traits2< float, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

// ----------------------------------------
// double
template<>
class traits2< double, std::complex<float> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

// ---------------
template<>
class traits2< double, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

// ----------------------------------------
// complex<float>
template<>
class traits2< std::complex<float>, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

template<>
class traits2< std::complex<float>, std::complex<float> >
{
public:
    typedef std::complex<float> scalar_t;
    typedef float real_t;
};

// ----------------------------------------
// complex<double>
template<>
class traits2< std::complex<double>, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

// -----------------------------------------------------------------------------
// traits3
/// Given three types, defines scalar and real types compatible with all types.
/// E.g., for the triple (float, complex<float>, double),
/// scalar_t = complex<double>, real_t = double.

// ----------------------------------------
template< typename T1, typename T2, typename T3 >
class traits3
{
public:
    typedef typename
        traits2< typename traits2<T1,T2>::scalar_t, T3 >::scalar_t scalar_t;

    typedef typename
        traits2< typename traits2<T1,T2>::scalar_t, T3 >::real_t real_t;
};

} // end namespace blam

#endif        //  #ifndef BLAS_UTIL_HH
