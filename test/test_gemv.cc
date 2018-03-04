#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template< typename TA, typename TX, typename TY,
          typename Policy = decltype(blam_default_policy) >
void test_gemv_work( Params& params, bool run, Policy policy = blam_default_policy )
{
    namespace blas = blam;
    using namespace blam;
    using std::real; using std::imag;

    typedef typename traits3< TA, TX, TY >::scalar_t scalar_t;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Op trans  = params.trans.value();
    scalar_t alpha  = params.alpha.value();
    scalar_t beta   = params.beta.value();
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx.value();
    int64_t incy    = params.incy.value();
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
    int64_t Am = (layout == Layout::ColMajor ? m : n);
    int64_t An = (layout == Layout::ColMajor ? n : m);
    int64_t lda = roundup( Am, align );
    int64_t Xm = (trans == Op::NoTrans ? n : m);
    int64_t Ym = (trans == Op::NoTrans ? m : n);
    size_t size_A = size_t(lda)*An;
    size_t size_x = (Xm - 1) * std::abs(incx) + 1;
    size_t size_y = (Ym - 1) * std::abs(incy) + 1;
    TA* A    = new TA[ size_A ];
    TX* x    = new TX[ size_x ];
    TY* y    = new TY[ size_y ];
    TY* yref = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( Ym, y, incy, yref, incy );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Xnorm = cblas_nrm2( Xm, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( Ym, y, std::abs(incy) );

    // test error exits
    assert_throw( blas::gemv( policy, Layout(0), trans,  m,  n, alpha, A, lda, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::gemv( policy, layout,    Op(0),  m,  n, alpha, A, lda, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::gemv( policy, layout,    trans, -1,  n, alpha, A, lda, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::gemv( policy, layout,    trans,  m, -1, alpha, A, lda, x, incx, beta, y, incy ), blas::Error );

    assert_throw( blas::gemv( policy, Layout::ColMajor, trans,  m,  n, alpha, A, m-1, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::gemv( policy, Layout::RowMajor, trans,  m,  n, alpha, A, n-1, x, incx, beta, y, incy ), blas::Error );

    assert_throw( blas::gemv( policy, layout,    trans,  m,  n, alpha, A, lda, x, 0,    beta, y, incy ), blas::Error );
    assert_throw( blas::gemv( policy, layout,    trans,  m,  n, alpha, A, lda, x, incx, beta, y, 0    ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x Xm=%5lld, inc=%5lld,           size=%10lld, norm=%.2e\n"
                "y Ym=%5lld, inc=%5lld,           size=%10lld, norm=%.2e\n",
                (lld) Am, (lld) An, (lld) lda, (lld) size_A, Anorm,
                (lld) Xm, (lld) incx,          (lld) size_x, Xnorm,
                (lld) Ym, (lld) incy,          (lld) size_y, Ynorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( m, n, A, lda );
        printf( "x    = " ); print_vector( Xm, x, incx );
        printf( "y    = " ); print_vector( Ym, y, incy );
    }

    // run test
    double time;
    {
      TA* dA = create_device_copy(policy, A, size_A);
      TX* dx = create_device_copy(policy, x, size_x);
      TY* dy = create_device_copy(policy, y, size_y);

      libtest::flush_cache( params.cache.value() );
      auto timer = get_timer(policy);
      blas::gemv( policy, layout, trans, m, n, alpha, dA, lda, dx, incx, beta, dy, incy );
      time = timer.seconds();

      copy_from_device(policy, dy, y, size_y);
      destroy(policy, dA);
      destroy(policy, dx);
      destroy(policy, dy);
    }

    double gflop = Gflop< scalar_t >::gemv( m, n );
    double gbyte = Gbyte< scalar_t >::gemv( m, n );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;
    params.gbytes.value() = gbyte / time;

    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( n, y, incy );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        cblas_gemv( cblas_layout_const(layout), cblas_trans_const(trans), m, n,
                    alpha, A, lda, x, incx, beta, yref, incy );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;
        params.ref_gbytes.value() = gbyte / time;

        if (verbose >= 2) {
            printf( "yref = " ); print_vector( Ym, yref, incy );
        }

        // check error compared to reference
        // treat y as 1 x Ym matrix with ld = incy; k = Xm is reduction dimension
        real_t error;
        bool okay;
        check_gemm( 1, Ym, Xm, alpha, beta, Anorm, Xnorm, Ynorm,
                    yref, std::abs(incy), y, std::abs(incy), verbose, &error, &okay );
        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] A;
    delete[] x;
    delete[] y;
    delete[] yref;
}

// -----------------------------------------------------------------------------
void test_gemv( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_gemv_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gemv_work< float, float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gemv_work< double, double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gemv_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gemv_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;
    }
}
