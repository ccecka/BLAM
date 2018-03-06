BLAM: Basic Linear Algebra Module
=================================

BLAM (name unstable) is a unifying C++ interface to BLAS implementations and their extensions that uses a high(er)-level interface to enhance programmer productivity and enable performance portability.

BLAM uses execution policies (see C++17 STL and Thrust) to allow multiple backends for each operation while maintaining common, generic transformations and simplified interfaces to every backend as well.

Documentation
-------------

* [SLATE Working Note 2: C++ API for BLAS and LAPACK](http://www.icl.utk.edu/publications/swan-002)
* [SLATE Working Note 4: C++ API for Batch BLAS](http://www.icl.utk.edu/publications/swan-004)
* BLAM Doxygen [in progress]
* [BLAS++: C++ API for the Basic Linear Algebra Subroutines](https://bitbucket.org/icl/blaspp)

Design Choices
--------------

### Type Dispatching

BLAM wraps cblas and cublas to provide overloaded interfaces which can be accessed directly via the `blam::cblas::` and blam::cublas::` namespaces. Note that these wrappers do not change the interfaces -- e.g. cuBLAS requires a `cublasHandle_t` object and alpha/beta parameters are often passed by pointer -- and only act to provide convenient type overloading.

### Execution Policies

Execution policies are extensible and composable objects that may be used to specify how an algorithm should be executed. Execution policies are used throughout the C++17 Parallel STL and we largely follow the design laid out in [N3554](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3554.pdf).

BLAM currently provides two primary execution policies, `blam::cblas::par` and `blam::cublas::par`, that can be used with a a unified BLAS interface in the `blam::` namespace. These policies dispatch to the appropriate backend and convert the unified interface to the backend interface.

In the future, we will show how these policies can be composed with other libraries, extended for custom behavior, and modified for improved control over execution.

### Customization Points

BLAM functions are Neibler-style customization points via function objects. This forces qualified and "two-step" qualified calls into the library to be captured and entered into dispatch without ADL. That is, BLAM always gets a chance to inspect parameters, even in the case that ADL (e.g. using an external execution_policy from thrust, agency, or the standard library) would otherwise select another algorithm. This could be useful for error messages and static failures.

Each customization point is currently tasked with finding an appropriate dispatch via ADL or falling back to a "generic" implementation. A "generic" backend is called when a suitable dispatch cannot be found. The `generic` implementation may:

1. Map the call to an equivalent call that can be sent back to the customization point. This can be used to easily provide new entry point interfaces that default parameters or provide interface conveniences.

2. "Decay" the operation to a weaker operation. e.g. GEMM can be evaluated with multiple GEMVs. This can be used to implement/test new backends quickly: implementating AXPBY/DOT could provide the necessary functionality for nearly all of BLAS.

3. Statically fail, with function signature + error message.

While (2) seems useful in some cases, it should be done conservatively to avoid performance degradation. Currently, BLAM uses an opt-in with the BLAM_DECAY preprocessor flag. This is currently being ported to a 'blam/system/decay' wrapper policy for better modularity.

BLAM supports external execution policies (e.g. thrust/agency/C++17 backends).


TODO
----

* Guidance for `__host__ __device__` annotations.

* Guidance for `__managed__` customization points.

* Accept abstracted pointers (e.g. thrust::device_ptr)?

* Guidance for default execution policies.
** Currently algorithms invoked without an execution policy fail. Alternatively, this could default to sequential or a use-defined policy.

* Guidance for synchronous/asychronous execution.

[ ] Public Sequencial implementations.

[ ] Public ThrustBlas/Agency implementations.
