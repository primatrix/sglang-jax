# JIT Cache Optimization Test

This document describes how to test the JIT cache optimization that replaces double partial wrapping with closure-based approach.

## 🎯 What This Tests

The optimization addresses the core issue where JAX cache was being invalidated due to:
1. **Double partial wrapping** creating functions with different hashes
2. **Static arguments changing** between calls
3. **Function recreation** during precompile loops

## 🧪 Test Suite

### Core Tests

1. **`test_cache_hit_rate()`** - Validates that identical calls hit the cache
2. **`test_precompile_cache_effectiveness()`** - Tests precompile speedup
3. **`test_different_batch_sizes()`** - Verifies cache behavior across configurations
4. **`test_functional_correctness()`** - Ensures optimization doesn't break functionality
5. **`test_memory_efficiency()`** - Checks memory usage remains reasonable

## 🚀 Running the Tests

### Quick Start

```bash
# Basic test run
python test_jit_cache_optimization.py

# With custom model path
python test_jit_cache_optimization.py --model-path /path/to/your/model

# With tensor parallelism
python test_jit_cache_optimization.py --tp-size 4

# Verbose output
python test_jit_cache_optimization.py --verbose

# Run specific test
python test_jit_cache_optimization.py --specific-test test_cache_hit_rate
```

### Manual Test Run

```bash
# Set environment
export MODEL_PATH="/models/Qwen-7B"
export TP_SIZE=1
export JAX_PLATFORMS=tpu
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_test_cache"

# Run tests
python -m unittest python.sgl_jax.test.model_executor.test_jit_cache_optimization -v
```

## 📊 Expected Results

### Before Optimization (Double Partial)
```
First call: 15.234s, cache misses: 8
Second call: 14.891s, cache misses: 8  ❌ No cache hits!
```

### After Optimization (Closure)
```
First call: 15.234s, cache misses: 8
Second call: 0.012s, cache misses: 0   ✅ Cache hit!
Speedup: 1269.5x
```

### Precompile Performance
```
First precompile time: 45.123s
Second precompile time: 2.456s         ✅ 18.4x speedup
Total cache misses in second run: 0
```

## 🔧 Test Configuration

The test uses reduced precompile configurations for faster execution:
- **Token paddings**: `[64, 128]` (vs default `[64, 128, 256, 512, 1024, 2048, 4096, 8192]`)
- **Batch paddings**: `[1, 2]` (vs default `[1, 2, 4, 8, 16, 32, 64, 128, 256]`)

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```bash
   # Make sure model path exists or use a different model
   python test_jit_cache_optimization.py --model-path /path/to/existing/model
   ```

2. **TPU Not Available**
   ```bash
   # Use CPU for testing
   python test_jit_cache_optimization.py --device cpu
   ```

3. **Memory Issues**
   ```bash
   # Reduce TP size
   python test_jit_cache_optimization.py --tp-size 1
   ```

### Debug Mode

To debug cache misses in detail:
```bash
export JAX_LOG_COMPILES=1
python test_jit_cache_optimization.py --verbose
```

## 📈 Performance Interpretation

### Good Cache Performance
- **Second call cache misses**: 0
- **Speed improvement**: >100x for repeated calls
- **Precompile speedup**: >10x for second run

### Poor Cache Performance (indicates issues)
- **Second call cache misses**: >1
- **Speed improvement**: <10x
- **Precompile speedup**: <5x

## 🎛️ Environment Variables

- `MODEL_PATH`: Path to the model
- `TP_SIZE`: Tensor parallel size
- `JAX_PLATFORMS`: Device type (tpu/gpu/cpu)
- `JAX_COMPILATION_CACHE_DIR`: Cache directory
- `ENABLE_PRECISION_TRACER`: Set to "0" for faster testing

## ✅ Success Criteria

The optimization is working correctly if:
1. ✅ Cache hit rate is 100% for identical calls
2. ✅ Second forward call is >100x faster
3. ✅ Second precompile run is >10x faster
4. ✅ Functional correctness is maintained
5. ✅ Memory usage stays reasonable

## 🔄 Next Steps

If tests pass, the optimization is ready for production use. If tests fail:

1. Check the closure implementation in `model_runner.py`
2. Verify no static arguments are changing between calls
3. Ensure function hashes remain stable
4. Consider implementing fallback mechanisms
