## Expression Generation

The expression generation strategy is different from well-known works such as [CSmith](https://embed.cs.utah.edu/csmith) due to being in a different situation. 
1. TVM has no specification and thus it is not possible to produce canonical programs
2. There are not multiple implementations of TVM as there are in C (gcc, CompCert, Clang, etc.). Differential testing is therefore not possible.
3. TIR represents a very small subset of a language such as C so it is feasible to construct a ground-truth.

For these reasons the situation is considerably simplified. 

## Generation Strategy

Rather than using rules with predicates and backtracing, TVMFuzz cajoles the inputs to an operator to be valid. That is to say that it is impossible for an operator to fail to generate after it has been selected in TVMFuzz. For example the `tir.Any` operator requires that every operand is of type `bool`. Rather than having a predicate that checks whether each operand is of type `bool` and backtracking if that check fails, TVMFuzz simply casts each operand to `bool`. 

This approach has the advantage that when the table of operators and their probabilities is defined, each probability is accurate. However, the disadvantage of this approach is that there is a considerable amount of *glue* between operators. In practice, this has not stopped TVMFuzz from finding significant bugs. 
