# 1D Average Pooling

Perform 1D average pooling on an input tensor. The operation slides a window of size k over the input with stride S and padding P, computing the average inside each window.

Mathematical definition:

$$
\text{output}[i] \;=\; \frac{1}{k}\sum_{m=0}^{k-1} \text{input}\big[S\cdot i + m - P\big]
$$

Out-of-range input indices (due to padding) are treated as zero.

Inputs
- input: 1D tensor of length H
- kernel_size k: size of the pooling window
- stride S: step between window positions
- padding P: number of zero elements added at both ends

Output
- output: 1D tensor of length H_out, where
$$
H_{\text{out}} \;=\; \left\lfloor\frac{H + 2P - k}{S}\right\rfloor + 1
$$

Notes
- If the window extends beyond the (padded) input, those positions contribute 0 to the sum.
- Common choices: P = 0 (no padding), S = k (non-overlapping pooling).

Example
- H = 10, k = 3, S = 2, P = 1
$$
H_{\text{out}} = \left\lfloor\frac{10 + 2\cdot1 - 3}{2}\right\rfloor + 1 = \left\lfloor\frac{9}{2}\right\rfloor + 1 = 5
$$

Implementation hint (pseudo-code)
- For i in 0..H_out-1:
  - sum = 0
  - for m in 0..k-1:
    - idx = S*i + m - P
    - if 0 ≤ idx < H: sum += input[idx]
    - else: sum += 0
  - output[i]