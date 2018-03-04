#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template< typename TA, typename TB, typename TC,
          typename Policy = decltype(blam_default_policy) >
void test_gemm_work( Params& params, bool run, Policy policy = blam_default_policy )
{
    namespace blas = blam;
    using namespace blam;
    using std::real; using std::imag;

    typedef typename traits3< TA, TB, TC >::scalar_t scalar_t;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Op transA = params.transA.value();
    blas::Op transB = params.transB.value();
    scalar_t alpha  = params.alpha.value();
    scalar_t beta   = params.beta.value();
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t k       = params.dim.k();
    int64_t align   = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    if ( ! run)
        return;

    // setup
    int64_t Am = (transA == Op::NoTrans ? m : k);
    int64_t An = (transA == Op::NoTrans ? k : m);
    int64_t Bm = (transB == Op::NoTrans ? k : n);
    int64_t Bn = (transB == Op::NoTrans ? n : k);
    int64_t Cm = m;
    int64_t Cn = n;
    if (layout == Layout::RowMajor) {
        std::swap( Am, An );
        std::swap( Bm, Bn );
        std::swap( Cm, Cn );
    }
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Bm, align );
    int64_t ldc = roundup( Cm, align );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*Bn;
    size_t size_C = size_t(ldc)*Cn;
    TA* A    = new TA[ size_A ];
    TB* B    = new TB[ size_B ];
    TC* C    = new TC[ size_C ];
    TC* Cref = new TC[ size_C ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", Cm, Cn, C, ldc, Cref, ldc );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B, ldb, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C, ldc, work );

    // test error exits
    assert_throw( blas::gemm( policy, Layout(0), transA, transB,  m,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, layout,    Op(0),  transB,  m,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, layout,    transA, Op(0),   m,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, layout,    transA, transB, -1,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, layout,    transA, transB,  m, -1,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, layout,    transA, transB,  m,  n, -1, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::gemm( policy, Layout::ColMajor, Op::NoTrans,   Op::NoTrans, m, n, k, alpha, A, m-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, Layout::ColMajor, Op::Trans,     Op::NoTrans, m, n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, Layout::ColMajor, Op::ConjTrans, Op::NoTrans, m, n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::gemm( policy, Layout::RowMajor, Op::NoTrans,   Op::NoTrans, m, n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, Layout::RowMajor, Op::Trans,     Op::NoTrans, m, n, k, alpha, A, m-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, Layout::RowMajor, Op::ConjTrans, Op::NoTrans, m, n, k, alpha, A, m-1, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::gemm( policy, Layout::ColMajor, Op::NoTrans, Op::NoTrans,   m, n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, Layout::ColMajor, Op::NoTrans, Op::Trans,     m, n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, Layout::ColMajor, Op::NoTrans, Op::ConjTrans, m, n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );

    assert_throw( blas::gemm( policy, Layout::RowMajor, Op::NoTrans, Op::NoTrans,   m, n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, Layout::RowMajor, Op::NoTrans, Op::Trans,     m, n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( policy, Layout::RowMajor, Op::NoTrans, Op::ConjTrans, m, n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );

    assert_throw( blas::gemm( policy, Layout::ColMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, m-1 ), blas::Error );
    assert_throw( blas::gemm( policy, Layout::RowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, n-1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B Bm=%5lld, Bn=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n"
                "C Cm=%5lld, Cn=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                (lld) Am, (lld) An, (lld) lda, (lld) size_A, Anorm,
                (lld) Bm, (lld) Bn, (lld) ldb, (lld) size_B, Bnorm,
                (lld) Cm, (lld) Cn, (lld) ldc, (lld) size_C, Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( Am, An, A, lda );
        printf( "B = "    ); print_matrix( Bm, Bn, B, ldb );
        printf( "C = "    ); print_matrix( Cm, Cn, C, ldc );
    }

    // run test
    double time;
    {
      TA* dA = create_device_copy(policy, A, size_A);
      TB* dB = create_device_copy(policy, B, size_B);
      TC* dC = create_device_copy(policy, C, size_C);

      libtest::flush_cache( params.cache.value() );
      auto timer = get_timer(policy);
      blas::gemm( policy, layout, transA, transB, m, n, k,
                  alpha, dA, lda, dB, ldb, beta, dC, ldc );
      time = timer.seconds();

      copy_from_device(policy, dC, C, size_C);
      destroy(policy, dA);
      destroy(policy, dB);
      destroy(policy, dC);
    }

    double gflop = Gflop < scalar_t >::gemm( m, n, k );
    params.time.value()   = time;
    params.gflops.value() = gflop / time;

    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( Cm, Cn, C, ldc );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        cblas_gemm( cblas_layout_const(layout),
                    cblas_trans_const(transA),
                    cblas_trans_const(transB),
                    m, n, k, alpha, A, lda, B, ldb, beta, Cref, ldc );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time;
        params.ref_gflops.value() = gflop / time;

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( Cm, Cn, Cref, ldc );
        }

        // check error compared to reference
        real_t error;
        bool okay;
        check_gemm( Cm, Cn, k, alpha, beta, Anorm, Bnorm, Cnorm,
                    Cref, ldc, C, ldc, verbose, &error, &okay );
        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] Cref;
}

// -----------------------------------------------------------------------------
void test_gemm( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_gemm_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_gemm_work< float, float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gemm_work< double, double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gemm_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gemm_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;
    }
}
