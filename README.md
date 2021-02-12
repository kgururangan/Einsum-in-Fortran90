# Einsum-in-Fortran90
A repository containing codes used to develop an einsum tensor construction function in Modern Fortran. The syntax is meant to mirror Numpy's einsum function. The function can be called with the following syntax
`call einsum(string,A,B,C)` where `string` is a string containing the indical scheme for the contraction. For instance, a regular 2D matrix multiplication between matrices $A$ and $B$ such that $C_{ij} = \sum_k A_{ik}B_{kj}$ can be executed as `call einsum('ik,kj->ij',A,B,C)`.
