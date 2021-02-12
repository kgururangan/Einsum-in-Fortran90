# Einsum-in-Fortran90
A repository containing codes used to develop an einsum tensor contraction function in Modern Fortran. The syntax is meant to mirror Numpy's einsum function and uses the TTGT (transpose-transpose-GEMM-transpose) scheme. The function can be called with the following syntax
`call einsum(string,A,B,C)` where `string` is a string containing the indical scheme for the contraction. For instance, a regular 2D matrix multiplication between matrices A and B such that C(i,j) = A(i,k)B(k,j) can be executed as `call einsum('ik,kj->ij',A,B,C)`. Of course, higher-order tensor contraction can be analogously defined. Currently, the `einsum` function only supports contractions between even-rank tensors and there is no support for complete contractions (e.g. dot products) or outer products.
