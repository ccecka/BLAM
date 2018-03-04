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
void test_trsv_work( Params& params, bool run, Policy policy = blam_default_policy )
{
    #define A(i_, j_) (A + (i_) + (j_)*lda)

    namespace blas = blam;
    using namespace blam;
    using std::real; using std::imag;

    typedef typename traits2< TA, TX >::scalar_t scalar_t;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Uplo uplo = params.uplo.value();
    blas::Op trans  = params.trans.value();
    blas::Diag diag = params.diag.value();
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
    size_t size_x = size_t(n - 1) * std::abs(incx) + 1;
    TA* A    = new TA[ size_A ];
    TX* x    = new TX[ size_x ];
    TX* xref = new TX[ size_x ];
    int* ipiv = new int[ n ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    cblas_copy( n, x, incx, xref, incx );

    // make A a well-conditioned triangle
    int info = 0;
    lapack_getrf( n, n, A, lda, ipiv, &info );
    if (diag == Diag::NonUnit) {
        // copy upper => lower
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < j; ++i) {
                *A(j,i) = *A(i,j);
            }
        }
    }
    else {
        // copy lower => upper
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < j; ++i) {
                *A(i,j) = *A(j,i);
            }
        }
    }
    assert( info == 0 );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lantr( "f", to_string(uplo), to_string(diag),
                                 n, n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );

    // test error exits
    assert_throw( blas::trsv( policy, Layout(0), uplo,    trans, diag,     n, A, lda, x, incx ), blas::Error );
    assert_throw( blas::trsv( policy, layout,    Uplo(0), trans, diag,     n, A, lda, x, incx ), blas::Error );
    assert_throw( blas::trsv( policy, layout,    uplo,    Op(0), diag,     n, A, lda, x, incx ), blas::Error );
    assert_throw( blas::trsv( policy, layout,    uplo,    trans, Diag(0),  n, A, lda, x, incx ), blas::Error );
    assert_throw( blas::trsv( policy, layout,    uplo,    trans, diag,    -1, A, lda, x, incx ), blas::Error );
    assert_throw( blas::trsv( policy, layout,    uplo,    trans, diag,     n, A, n-1, x, incx ), blas::Error );
    assert_throw( blas::trsv( policy, layout,    uplo,    trans, diag,     n, A, lda, x,    0 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                (lld) n, (lld) lda,  (lld) size_A, Anorm,
                (lld) n, (lld) incx, (lld) size_x, Xnorm );
    }
    if (verbose >= 2) {
        printf( "A = "    ); print_matrix( n, n, A, lda );
        printf( "x    = " ); print_vector( n, x, incx );
    }

    // run test
    double time;
    {
      TA* dA = create_device_copy(policy, A, size_A);
      TX* dx = create_device_copy(policy, x, size_x);

      libtest::flush_cache( params.cache.value() );
      auto timer = get_timer(policy);
      blas::trsv( policy, layout, uplo, trans, diag, n, dA, lda, dx, incx );
      time = timer.seconds();

      copy_from_device(policy, dx, x, size_x);
      destroy(policy, dA);
      destroy(policy, dx);
    }

    double gflop = Gflop < scalar_t >::trsv( n );
    double gbyte = Gbyte < scalar_t >::trsv( n );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;
    params.gbytes.value() = gbyte / time;

    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
    }

    if (params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        cblas_trsv( cblas_layout_const(layout),
                    cblas_uplo_const(uplo),
                    cblas_trans_const(trans),
                    cblas_diag_const(diag),
                    n, A, lda, xref, incx );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;
        params.ref_gbytes.value() = gbyte / time;

        if (verbose >= 2) {
            printf( "xref = " ); print_vector( n, xref, incx );
        }

        // check error compared to reference
        // treat x as 1 x n matrix with ld = incx; k = n is reduction dimension
        // alpha = 1, beta = 0.
        real_t error;
        bool okay;
        check_gemm( 1, n, n, scalar_t(1), scalar_t(0), Anorm, Xnorm, real_t(0),
                    xref, std::abs(incx), x, std::abs(incx), verbose, &error, &okay );
        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] A;
    delete[] x;
    delete[] xref;
    delete[] ipiv;

    #undef A
}

// -----------------------------------------------------------------------------
void test_trsv( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_trsv_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_trsv_work< float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_trsv_work< double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_trsv_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_trsv_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;
    }
}
