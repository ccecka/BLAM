#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"

// -----------------------------------------------------------------------------
template< typename TX, typename TY, typename Policy = decltype(blam_default_policy) >
void test_swap_work( Params& params, bool run, Policy policy = blam_default_policy )
{
    namespace blas = blam;
    using namespace blam;
    using std::real; using std::imag;

    typedef typename traits2< TX, TY >::scalar_t scalar_t;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
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
    TX* xref = new TX[ size_x ];
    TY* y    = new TY[ size_y ];
    TY* yref = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, x, incx, xref, incx );
    cblas_copy( n, y, incy, yref, incy );

    // test error exits
    assert_throw( blas::swap( policy, -1, x, incx, y, incy ), blas::Error );
    assert_throw( blas::swap( policy,  n, x,    0, y, incy ), blas::Error );
    assert_throw( blas::swap( policy,  n, x, incx, y,    0 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n"
                "y n=%5lld, inc=%5lld, size=%10lld\n",
                (lld) n, (lld) incx, (lld) size_x,
                (lld) n, (lld) incy, (lld) size_y );
    }
    if (verbose >= 2) {
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
      blas::swap( policy, n, dx, incx, dy, incy );
      time = timer.seconds();

      copy_from_device(policy, dx, x, size_x);
      copy_from_device(policy, dy, y, size_y);
      destroy(policy, dx);
      destroy(policy, dy);
    }

    double gflop = Gflop < scalar_t >::swap( n );
    double gbyte = Gbyte < scalar_t >::swap( n );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;
    params.gbytes.value() = gbyte / time;

    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
        printf( "y2   = " ); print_vector( n, y, incy );
    }

    if (params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        cblas_swap( n, xref, incx, yref, incy );
        time = omp_get_wtime() - time;
        if (verbose >= 2) {
            printf( "xref = " ); print_vector( n, xref, incx );
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;
        params.ref_gbytes.value() = gbyte / time;

        // error = ||xref - x|| + ||yref - y||
        cblas_axpy( n, -1.0, x, incx, xref, incx );
        cblas_axpy( n, -1.0, y, incy, yref, incy );
        real_t error = cblas_nrm2( n, xref, std::abs(incx) )
                     + cblas_nrm2( n, yref, std::abs(incy) );
        params.error.value() = error;

        // swap must be exact!
        params.okay.value() = (error == 0);
    }

    delete[] x;
    delete[] y;
    delete[] xref;
    delete[] yref;
}

// -----------------------------------------------------------------------------
void test_swap( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_swap_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_swap_work< float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_swap_work< double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_swap_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_swap_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;
    }
}
