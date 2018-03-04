#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template< typename TX, typename TY, typename Policy = decltype(blam_default_policy) >
void test_axpy_work( Params& params, bool run, Policy policy = blam_default_policy )
{
    namespace blas = blam;
    using namespace blam;
    using std::real; using std::imag;

    typedef typename traits2< TX, TY >::scalar_t scalar_t;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    scalar_t alpha  = params.alpha.value();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx.value();
    int64_t incy    = params.incy.value();
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
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    TX* x    = new TX[ size_x ];
    TY* y    = new TY[ size_y ];
    TY* yref = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, y, incy, yref, incy );

    // test error exits
    assert_throw( blas::axpy( policy, -1, alpha, x, incx, y, incy ), blas::Error );
    assert_throw( blas::axpy( policy,  n, alpha, x,    0, y, incy ), blas::Error );
    assert_throw( blas::axpy( policy,  n, alpha, x, incx, y,    0 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n"
                "y n=%5lld, inc=%5lld, size=%10lld\n",
                (lld) n, (lld) incx, (lld) size_x,
                (lld) n, (lld) incy, (lld) size_y );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "x    = " ); print_vector( n, x, incx );
        printf( "y    = " ); print_vector( n, y, incy );
    }

    // run test
    double time;
    {
      TX* dx = create_device_copy(policy, x, size_x);
      TY* dy = create_device_copy(policy, y, size_y);

      libtest::flush_cache( params.cache.value() );
      auto timer = get_timer(policy);
      blas::axpy( policy, n, alpha, dx, incx, dy, incy );
      time = timer.seconds();

      copy_from_device(policy, dy, y, size_y);
      destroy(policy, dx);
      destroy(policy, dy);
    }

    double gflop = Gflop < scalar_t >::axpy( n );
    double gbyte = Gbyte< scalar_t >::axpy( n );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;
    params.gbytes.value() = gbyte / time;

    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( n, y, incy );
    }

    if (params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        cblas_axpy( n, alpha, x, incx, yref, incy );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;
        params.ref_gbytes.value() = gbyte / time;

        if (verbose >= 2) {
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        // error = ||yref - y|| / ||y|| ... todo
        cblas_axpy( n, -1.0, y, incy, yref, incy );
        real_t error = cblas_nrm2( n, yref, std::abs(incy) );
        real_t ynorm = cblas_nrm2( n, y,    std::abs(incy) );
        error /= ynorm;
        params.error.value() = error;

        real_t eps = std::numeric_limits< real_t >::epsilon();
        real_t tol = params.tol.value() * eps;
        params.okay.value() = (error < tol);
    }

    delete[] x;
    delete[] y;
    delete[] yref;
}

// -----------------------------------------------------------------------------
void test_axpy( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_axpy_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_axpy_work< float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_axpy_work< double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_axpy_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_axpy_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;
    }
}
