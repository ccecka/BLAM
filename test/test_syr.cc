#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template< typename TA, typename TX,
          typename Policy = decltype(blam_default_policy) >
void test_syr_work( Params& params, bool run, Policy policy = blam_default_policy )
{
    namespace blas = blam;
    using namespace blam;
    using std::real; using std::imag;

    typedef typename traits2< TA, TX >::scalar_t scalar_t;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Uplo uplo = params.uplo .value();
    scalar_t alpha  = params.alpha.value();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx.value();
    int64_t align   = params.align.value();
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
    int64_t lda = roundup( n, align );
    size_t size_A = size_t(lda)*n;
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    TA* A    = new TA[ size_A ];
    TA* Aref = new TA[ size_A ];
    TX* x    = new TX[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    lapack_lacpy( "g", n, n, A, lda, Aref, lda );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lansy( "f", to_string(uplo), n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );

    // test error exits
    assert_throw( blas::syr( policy, Layout(0), uplo,     n, alpha, x, incx, A, lda ), blas::Error );
    assert_throw( blas::syr( policy, layout,    Uplo(0),  n, alpha, x, incx, A, lda ), blas::Error );
    assert_throw( blas::syr( policy, layout,    uplo,    -1, alpha, x, incx, A, lda ), blas::Error );
    assert_throw( blas::syr( policy, layout,    uplo,     n, alpha, x,    0, A, lda ), blas::Error );
    assert_throw( blas::syr( policy, layout,    uplo,     n, alpha, x, incx, A, n-1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                (lld) n, (lld) lda,  (lld) size_A, Anorm,
                (lld) n, (lld) incx, (lld) size_x, Xnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "A = " ); print_matrix( n, n, A, lda );
        printf( "x = " ); print_vector( n, x, incx );
    }

    // run test
    double time;
    {
      TA* dA = create_device_copy(policy, A, size_A);
      TX* dx = create_device_copy(policy, x, size_x);

      libtest::flush_cache( params.cache.value() );
      auto timer = get_timer(policy);
      blas::syr( policy, layout, uplo, n, alpha, dx, incx, dA, lda );
      time = timer.seconds();

      copy_from_device(policy, dA, A, size_A);
      destroy(policy, dA);
      destroy(policy, dx);
    }

    double gflop = Gflop < scalar_t >::syr( n );
    double gbyte = Gbyte < scalar_t >::syr( n );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;
    params.gbytes.value() = gbyte / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( n, n, A, lda );
    }

    if (params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        cblas_syr( cblas_layout_const(layout), cblas_uplo_const(uplo),
                   n, alpha, x, incx, Aref, lda );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;
        params.ref_gbytes.value() = gbyte / time;

        if (verbose >= 2) {
            printf( "Aref = " ); print_matrix( n, n, Aref, lda );
        }

        // check error compared to reference
        // beta = 1
        real_t error;
        bool okay;
        check_herk( uplo, n, 1, alpha, scalar_t(1), Xnorm, Xnorm, Anorm,
                    Aref, lda, A, lda, verbose, &error, &okay );
        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] A;
    delete[] Aref;
    delete[] x;
}

// -----------------------------------------------------------------------------
void test_syr( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_syr_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_syr_work< float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_syr_work< double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
          //throw blas::Error( "syr< complex > in LAPACK++", __func__ );
            //test_syr_work< std::complex<float>, std::complex<float> >
            //    ( params, run );
            break;

        case libtest::DataType::DoubleComplex:
          //throw blas::Error( "syr< complex > in LAPACK++", __func__ );
            //test_syr_work< std::complex<double>, std::complex<double> >
            //    ( params, run );
            break;
    }
}
