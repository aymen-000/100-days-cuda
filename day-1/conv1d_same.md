# 1D Convolution

Compute the 1D convolution of an input signal A with a kernel B using zero padding.

Mathematical definition (kernel centered, K odd):

$$
C[i] = \sum_{j=0}^{K-1} A\!\left[i + j - \left\lfloor\frac{K}{2}\right\rfloor\right]\; \cdot\; B[j]
$$

Out-of-range accesses to A are treated as zero (zero padding), i.e. A[t] = 0 for t < 0 or t ≥ N.

Inputs
- A: vector of length N (input signal)
- B: vector of length K (convolution kernel), K is odd and K < N

Output
- C: vector of length N (convolved signal)

Notes
- The kernel is centered at each output position i with (K-1)/2 elements on each side.
- Use zero padding when the kernel extends beyond the input bounds.

Example (conceptual)
- N = 7, K = 3, center = 1
- For K = 3:
$$
C[i] = B[0]\;A[i-1] + B[1]\;A[i] + B[2]\;A[i+1], \quad \text{with } A[-1]=A[N]=0.
$$

Implementation hint
- For each i in 0..N-1:
  - Let center = floor(K/2).
  - Accumulate sum over j in 0..K-1 with a_idx = i + j - center.
  - If 0 ≤ a_idx < N then add A[a_idx] * B[j], otherwise