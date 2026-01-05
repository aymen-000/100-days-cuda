# CUDA Mandelbrot Set Generator

A high-performance GPU-accelerated Mandelbrot set renderer using CUDA.

## What is the Mandelbrot Set?

The Mandelbrot set is a fractal—a mathematical set of points that creates infinitely complex patterns. For each point `c` in the complex plane, we iterate the formula:
```
z(n+1) = z(n)² + c
```

Starting with `z(0) = 0`. If the sequence remains bounded (doesn't escape to infinity), the point is in the set (colored black). Otherwise, we color it based on how quickly it escapes.

