BLAM: Basic Linear Algebra Module
=================================

BLAM (name unstable) is a unifying interface to BLAS implementations and their extensions that uses a high(er)-level interface to enhance programmer productivity and enable performance portability.

BLAM uses execution policies (see Thrust and C++17 STL) to allow multiple backends for each operation while maintaining common, generic transformations and simplified interfaces to every backend as well.


Design Choices
--------------

### Entry Points

BLAM entry points are Neibler-style customization points via function objects. This forces qualified and "two-step" qualified calls into the library to be captured and entered into dispatch without ADL. That is, BLAM always gets a chance to inspect parameters, even in the case that ADL (e.g. using an external execution_policy from thrust, agency, or the standard library) would otherwise select another algorithm. This could be useful for error messages and static failures.

### `blam/adl`

The `blam/adl` is the true entry point and is currently tasked with finding an appropriate dispatch via ADL or falling back to `blam/system/generic`.

BLAM supports external execution policies (e.g. thrust/agency/C++17 backends).

### `blam/system/generic`

The `blam/system/generic` backend is called when a suitable dispatch cannot be found. The `generic` implementation may:

1. Map the call to an equivalent call that can be sent back to `blam/adl`. This can be used to easily provide new entry point interfaces that default parameters or provide interface conveniences.

2. "Decay" the operation to a weaker operation. e.g. GEMM can be evaluated with multiple GEMVs. This can be used to implement/test new backends quickly: implementating AXPBY could provide the necessary functionality for nearly all of BLAS.

3. Statically fail, with function signature + error message

While (2) seems useful in some cases, it should be done conservatively to avoid performance degradation. Currently, BLAM uses an opt-in with the BLAM_DECAY preprocessor flag. This is currently being ported to a 'blam/system/decay' wrapper policy for better modularity.


TODO
----

* Guidance for `__host__ __device__` annotations?

* Accept abstracted pointers (e.g. thrust::device_ptr) or iterators?

* Guidance for default execution policies?

* Guidance for return types and error checking.

[ ] Better name?

[x] Write a generator that yeilds the majority boilerplate based on the cblas/cublas specifications. Only `generic` and some of the BLAS extensions should be need to be written by hand.

[ ] More ThrustBlas/Agency implementations.
