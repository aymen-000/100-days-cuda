# Sigmoid Activation (Element-wise)

Apply the Sigmoid activation function element-wise to an input matrix A to produce output matrix C:

Mathematical definition:
$$
C[i][j] \;=\; \sigma\big(A[i][j]\big)
$$

where the Sigmoid function is
$$
\sigma(x) \;=\; \frac{1}{1 + e^{-x}}.
$$

Inputs
- A: matrix of shape M × N containing floating-point values.

Output
- C: matrix of shape M × N containing the Sigmoid activations.

