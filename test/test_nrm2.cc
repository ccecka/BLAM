#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"

// -----------------------------------------------------------------------------
template< typename T, typename Policy = decltype(blam_default_policy) >
void test_nrm2_work( Params& params, bool run, Policy policy = blam_default_policy )
{
    namespace blas = blam;
    using namespace blam;

    typedef typename traits<T>::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.gbytes.value();
    params.ref_time.value();
    params.ref_gflops.value();
    params.ref_gbytes.value();

    // adjust header to msec
    params.time.name( "BLAS++\ntime (ms)" );
    params.ref_time.name( "Ref.\ntime (ms)" );

    if ( ! run)
        return;

    // setup
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    T* x = new T[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );

    // test error exits
    real_t result;
    assert_throw( blas::nrm2( policy, -1, x, incx, result ), blas::Error );
    assert_throw( blas::nrm2( policy,  n, x,    0, result ), blas::Error );
    assert_throw( blas::nrm2( policy,  n, x,   -1, result ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n",
                (lld) n, (lld) incx, (lld) size_x );
    }
    if (verbose >= 2) {
        printf( "x = " ); print_vector( n, x, incx );
    }

    // run test
    double time;
    {
      T* dx = create_device_copy(policy, x, size_x);

      libtest::flush_cache( params.cache.value() );
      auto timer = get_timer(policy);
      blas::nrm2( policy, n, dx, incx, result );
      time = timer.seconds();

      destroy(policy, dx);
    }

    double gflop = Gflop < T >::nrm2( n );
    double gbyte = Gbyte < T >::nrm2( n );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;
    params.gbytes.value() = gbyte / time;

    if (verbose >= 2) {
        printf( "result = %.4e\n", result );
    }

    if (params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        real_t ref = cblas_nrm2( n, x, std::abs(incx) );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;
        params.ref_gbytes.value() = gbyte / time;

        if (verbose >= 2) {
            printf( "ref    = %.4e\n", ref );
        }

        // error = |ref - result| / |result|
        real_t error = std::abs( ref - result ) / std::abs( result );
        params.error.value() = error;

        real_t eps = std::numeric_limits< real_t >::epsilon();
        real_t tol = params.tol.value() * eps;
        params.okay.value() = (error < tol);
    }

    delete[] x;
}

// -----------------------------------------------------------------------------
void test_nrm2( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_nrm2_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_nrm2_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_nrm2_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_nrm2_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_nrm2_work< std::complex<double> >( params, run );
            break;
    }
}
