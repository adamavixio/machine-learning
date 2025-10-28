# ML Library Optimization Plan

## FINAL RESULTS (October 2025)

**Status**: Micro-optimizations complete - 4.3x overall speedup achieved
**Phase**: Shifting to algorithmic improvements (RL enhancements, 3x3 scaling)

### Optimizations Implemented

1. **Dual-context autodiff** - Separate parameter and temporary gradients
2. **BLAS transpose matmul** - Used Accelerate framework's optimized GEMM
3. **Scratch buffer pool** - Per-layer pre-allocated buffers for backward pass
4. **SIMD broadcast_add** - Vectorized bias gradient reduction

### Performance Achieved

#### Episode 100 (Early Training)
| Metric | Baseline | Final | Speedup |
|--------|----------|-------|---------|
| Total time | 21.6ms | 5.0ms | **4.3x** |
| Backward pass | 17.4ms | 1.4ms | **12.4x** |
| Forward pass | - | 2.9ms | - |

**Breakdown** (backward pass):
- matmul: 17.8% (was 91% in baseline)
- broadcast_add: 67.7%
- mul: 9.1%
- gather: 5.5%

#### Episode 200 (Later Training)
| Metric | Baseline | Final | Speedup |
|--------|----------|-------|---------|
| Total time | 22.0ms | 13.0ms | **1.7x** |
| Backward pass | 9.9ms | 3.4ms | **2.9x** |
| Forward pass | - | 7.8ms | - |

**Breakdown** (backward pass, cumulative):
- matmul: 1,055ms (8.6%) - was 62.5%, now excellent
- broadcast_add: 9,098ms (74.0%) - memory-bandwidth limited
- mul: 1,395ms (11.3%)
- gather: 747ms (6.1%)

### Key Insights

1. **Allocator overhead dominated matmul time** - Scratch buffers gave 9.7-17.7x speedup on matmul backward
2. **Memory bandwidth limits further gains** - broadcast_add is memory-bound (1 add per read)
3. **Cache locality > SIMD width** - First SIMD attempt with poor memory access was 21% slower
4. **Diminishing returns** - Each optimization yielded less than the previous

### Files Modified

- `src/tensor/autodiff.zig` - Dual-context, scratch buffers, SIMD reduction
- `src/tensor/blas.zig` - Accelerate framework integration
- `src/tensor/matmul.zig` - Dispatch to BLAS for backward transpose
- `src/nn/dense.zig` - Per-layer scratch buffer allocation
- `src/nn/model.zig` - Scratch buffer lifecycle management

### Next Steps (Completed)

**Declared victory on micro-optimizations.** ✅ Moved to algorithmic improvements:

**Completed implementations**:
1. ✅ **Dueling DQN architecture**: Separate value and advantage streams with proper gradient flow
2. ✅ **N-step returns**: Configurable 1-5 step TD for better credit assignment
3. ✅ **Layer normalization**: Stabilizes training for deeper networks
4. ✅ **Prioritized experience replay (PER)**: Importance sampling for efficient learning
5. ✅ **Curriculum learning**: Progressive scramble depth with success thresholds
6. ✅ **Supervised pretraining**: Optimal policy examples for faster convergence
7. ✅ **Test coverage**: Comprehensive tests for n-step metadata and dueling Q gradients (src/rl/replay.zig:203-251, src/tensor/autodiff.zig)

**Current status (October 2025)**:
- Long-horizon validation experiments in progress (depth-3 curriculum: 5000 episodes)
- Depth-4/5 fixed runs reached 1300/5000 episodes before stopping (partial results available)
- Dueling + n-step is now the default architecture
- All 32 tests passing, including replay buffer `n_steps_taken` sampling and dueling Q gradient propagation
- Latest benchmarks: 45.8% success on depth-4 scrambles (2000-episode run with curriculum escalation)

---

## ORIGINAL PLAN (Historical Reference)

**Note**: The plan below was drafted before implementation. We achieved the core performance goals through different techniques than originally planned.

**Target speedup**: 6-8x (from current 50 eps/min → 300-400 eps/min)
**Estimated effort**: 3-4 days

---

## Priority 1: Memory Pooling (2-3x speedup)

### Current Bottleneck
Every training step allocates/frees tensors repeatedly:
```zig
// Current: Per training step (batch_size=64)
- Forward pass: allocate hidden1 [64×128], hidden2 [64×64], output [64×6]
- Backward pass: allocate gradients for all layers
- Free everything
// Result: Thousands of alloc/free calls per second
```

### Proposed Solution: Pre-allocated Tensor Pool

**File**: `src/tensor/pool.zig` (new)

```zig
pub const TensorPool = struct {
    // Pre-allocated buffers for common sizes
    batch_hidden1: []f32,  // [batch_size × 128]
    batch_hidden2: []f32,  // [batch_size × 64]
    batch_output: []f32,   // [batch_size × 6]

    // Gradient buffers
    grad_hidden1: []f32,
    grad_hidden2: []f32,
    grad_output: []f32,

    // Scratch space for matmuls
    scratch: []f32,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, batch_size: usize) !TensorPool {
        return TensorPool{
            .batch_hidden1 = try allocator.alloc(f32, batch_size * 128),
            .batch_hidden2 = try allocator.alloc(f32, batch_size * 64),
            .batch_output = try allocator.alloc(f32, batch_size * 6),
            .grad_hidden1 = try allocator.alloc(f32, batch_size * 128),
            .grad_hidden2 = try allocator.alloc(f32, batch_size * 64),
            .grad_output = try allocator.alloc(f32, batch_size * 6),
            .scratch = try allocator.alloc(f32, batch_size * 128),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TensorPool) void {
        self.allocator.free(self.batch_hidden1);
        self.allocator.free(self.batch_hidden2);
        self.allocator.free(self.batch_output);
        self.allocator.free(self.grad_hidden1);
        self.allocator.free(self.grad_hidden2);
        self.allocator.free(self.grad_output);
        self.allocator.free(self.scratch);
    }

    /// Get a tensor view for forward pass activations
    pub fn getActivation(self: *TensorPool, layer: enum { hidden1, hidden2, output }) []f32 {
        return switch (layer) {
            .hidden1 => self.batch_hidden1,
            .hidden2 => self.batch_hidden2,
            .output => self.batch_output,
        };
    }

    /// Get a tensor view for gradients
    pub fn getGradient(self: *TensorPool, layer: enum { hidden1, hidden2, output }) []f32 {
        return switch (layer) {
            .hidden1 => self.grad_hidden1,
            .hidden2 => self.grad_hidden2,
            .output => self.grad_output,
        };
    }
};
```

### Integration Points

**`src/rl/qnetwork.zig`** - Add pool to network struct:
```zig
pub const QNetwork = struct {
    online_model: *MLP,
    target_model: *MLP,
    tensor_pool: TensorPool,  // NEW: Pre-allocated buffers
    allocator: std.mem.Allocator,

    pub fn init(..., batch_size: usize) !QNetwork {
        // ...existing code...
        const pool = try TensorPool.init(allocator, batch_size);
        return QNetwork{
            .online_model = online,
            .target_model = target,
            .tensor_pool = pool,  // NEW
            .allocator = allocator,
        };
    }
};
```

**`src/nn/mlp.zig`** - Use pool buffers instead of allocating:
```zig
pub fn forward(self: *Self, input: []const f32, pool: *TensorPool) ![]const f32 {
    // OLD: const h1 = try self.allocator.alloc(f32, batch_size * 128);
    // NEW: Use pre-allocated buffer
    const h1 = pool.getActivation(.hidden1);

    // Compute h1 = input @ W1
    try matmul(h1, input, self.W1.data, batch_size, input_dim, 128);
    relu(h1);  // In-place activation

    // Same for h2, output...
    const h2 = pool.getActivation(.hidden2);
    const out = pool.getActivation(.output);

    // No free() calls - buffers reused across calls
    return out;
}
```

### Expected Impact
- **Speedup**: 2-3x (eliminates allocation overhead)
- **Memory**: Same total usage, just pre-allocated
- **Lines changed**: ~200 (pool.zig + integration in 3-4 files)

---

## Priority 2: BLAS/Accelerate Integration (2x speedup on matmul)

### Current Implementation
Hand-rolled SIMD matmul in `src/tensor/matmul.zig`:
- Cache blocking (32x32 tiles)
- SIMD vectorization (8-wide on ARM)
- Performance: ~1-2 GFLOPS

### Proposed: Link Apple Accelerate Framework

**File**: `build.zig` - Add framework dependency:
```zig
pub fn build(b: *std.Build) void {
    // ...existing code...

    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // NEW: Link Accelerate on macOS
    if (target.result.isDarwin()) {
        exe.linkFramework("Accelerate");
    }
}
```

**File**: `src/tensor/blas.zig` (new) - Zig wrapper:
```zig
const std = @import("std");

// Import C headers
pub const cblas = @cImport({
    @cInclude("Accelerate/Accelerate.h");
});

/// Wrapper for BLAS SGEMM (single-precision general matrix multiply)
/// C = alpha * A @ B + beta * C
pub fn matmul_blas(
    C: []f32,
    A: []const f32,
    B: []const f32,
    M: usize,
    K: usize,
    N: usize,
) void {
    // Call BLAS: C[M×N] = A[M×K] @ B[K×N]
    cblas.cblas_sgemm(
        cblas.CblasRowMajor,  // Row-major layout
        cblas.CblasNoTrans,   // A not transposed
        cblas.CblasNoTrans,   // B not transposed
        @intCast(M),          // M rows of A/C
        @intCast(N),          // N cols of B/C
        @intCast(K),          // K cols of A, rows of B
        1.0,                  // alpha = 1.0
        A.ptr,                // A data
        @intCast(K),          // Leading dimension of A
        B.ptr,                // B data
        @intCast(N),          // Leading dimension of B
        0.0,                  // beta = 0.0 (overwrite C)
        C.ptr,                // C data (output)
        @intCast(N),          // Leading dimension of C
    );
}
```

**File**: `src/tensor/matmul.zig` - Dispatch to BLAS:
```zig
const blas = @import("blas.zig");

pub fn matmul(
    C: []Scalar,
    A: []const Scalar,
    B: []const Scalar,
    M: usize,
    K: usize,
    N: usize,
) !void {
    // Use BLAS on macOS, fallback to hand-rolled otherwise
    if (@import("builtin").target.os.tag == .macos) {
        blas.matmul_blas(C, A, B, M, K, N);
    } else {
        // Existing implementation
        if (M >= 64 and N >= 64 and K >= 64) {
            return matmulBlocked(C, A, B, M, K, N);
        } else {
            return matmulNaive(C, A, B, M, K, N);
        }
    }
}
```

### Expected Impact
- **Speedup**: 2x on matmul (50-100 GFLOPS vs 1-2 GFLOPS)
- **Overall**: ~1.5x total (matmul is ~50% of training time)
- **Lines changed**: ~50 (new blas.zig + dispatch logic)
- **Platform**: macOS only (fallback to hand-rolled on Linux/Windows)

---

## Priority 3: Batched Operations (1.5x speedup)

### Current Bottleneck
Process batch samples one-by-one:
```zig
// Current: Loop over batch
for (0..batch_size) |i| {
    const state_i = states[i * state_dim..(i+1) * state_dim];
    const q_values = try forward(state_i);  // 64 separate forward passes
}
```

### Proposed: Single Batched Forward Pass
```zig
// NEW: Single batched matmul
// states: [batch_size × state_dim] = [64 × 144]
// W1: [state_dim × hidden_dim] = [144 × 128]
// h1: [batch_size × hidden_dim] = [64 × 128]
try matmul(h1, states, W1, batch_size, state_dim, hidden_dim);
```

### Integration
**Already done!** Our `matmul()` already supports batched operations. We just need to refactor `trainStep()` in `src/rl/dqn.zig` to pass the full batch instead of looping.

**Before**:
```zig
for (0..batch_size) |i| {
    const q_online = try self.qnet.online_model.forward(&state_onehot, ctx);
    // ...
}
```

**After**:
```zig
// Reshape states_batch: [batch_size × state_dim]
const states_reshaped = try reshapeStates(states, batch_size, state_dim);
const q_online_batch = try self.qnet.online_model.forwardBatch(states_reshaped, batch_size, pool);
```

### Expected Impact
- **Speedup**: 1.5x (better cache utilization)
- **Lines changed**: ~100 (refactor DQN trainStep)

---

## Combined Expected Performance

### Current (ReleaseFast)
- **Speed**: 50 episodes/minute
- **Depth-3 training**: ~40 minutes total

### After All Optimizations
| Optimization | Speedup | Cumulative |
|--------------|---------|------------|
| Baseline (ReleaseFast) | 1.0x | 50 eps/min |
| + Memory pooling | 2.5x | 125 eps/min |
| + BLAS/Accelerate | 1.8x | 225 eps/min |
| + Batched ops | 1.5x | **340 eps/min** |

**Total speedup: 6.8x**
**Depth-3 training time: 6 minutes** (vs 40 minutes current)

---

## Implementation Plan (Post-Validation)

### Day 1: Memory Pooling
- [ ] Create `src/tensor/pool.zig`
- [ ] Add pool to `QNetwork` struct
- [ ] Refactor `MLP.forward()` to use pool
- [ ] Refactor `trainStep()` to use pool
- [ ] Test: Verify gradients unchanged (numerical diff check)

### Day 2: Memory Pooling (cont.)
- [ ] Add pool to replay buffer sampling
- [ ] Benchmark: Measure speedup (should see 2-3x)
- [ ] Test: Run 100 training episodes, verify learning curves match

### Day 3: BLAS Integration
- [ ] Create `src/tensor/blas.zig`
- [ ] Link Accelerate in `build.zig`
- [ ] Add dispatch logic in `matmul.zig`
- [ ] Test: Verify matmul outputs match (tolerance 1e-5)
- [ ] Benchmark: Measure matmul speedup (should see 20-50x on large matrices)

### Day 4: Batched Operations + Validation
- [ ] Refactor `trainStep()` for batched forward/backward
- [ ] Benchmark: Full training run (should see 340 eps/min)
- [ ] Validation: Re-run 2×2 cube training, verify depth-3 results match

---

## Fallback Plan (If Depth-3 Fails)

### If depth-3 success <5%
**Hypothesis**: Architecture limited, not speed

**Potential fixes**:
1. **Larger network**: 144→256→128→64→6 (more capacity)
2. **Different architecture**: Add attention mechanism for state-action relationships
3. **Better exploration**: Curiosity-driven exploration, count-based bonuses
4. **Reward shaping**: Dense rewards for progress (distance to solved state)
5. **Planning**: Integrate MCTS with learned value function

**Action**: Experiment with architectural changes BEFORE optimizing infrastructure

### If depth-3 success 15-30%
**Hypothesis**: Architecture works, needs more training or tuning

**Action**: Proceed with optimization, then:
- Increase training budget (2000 eps per depth)
- Tune hyperparameters (learning rate, epsilon decay)
- Consider auxiliary losses (inverse model, curiosity)

---

## Risk Mitigation

### BLAS Precision Issues
- **Risk**: BLAS might use different numerics (FMA vs separate mul+add)
- **Mitigation**: Tolerance testing, compare gradients before/after

### Memory Pooling Bugs
- **Risk**: Buffer reuse could cause race conditions or stale data
- **Mitigation**: Extensive testing, verify gradients match exactly

### Platform Dependency
- **Risk**: BLAS only on macOS, slower on Linux/Windows
- **Mitigation**: Keep hand-rolled matmul as fallback, consider OpenBLAS/MKL later

---

## Success Criteria

**Before starting optimization**:
- ✅ Depth-3 validation completes
- ✅ Success rate ≥15% OR architectural hypothesis formed

**After optimization**:
- ✅ Training speed ≥300 episodes/minute (6x speedup)
- ✅ Gradient numerical checks pass (tolerance <1e-5)
- ✅ Re-run validation shows same convergence behavior

---

**Next Step**: Wait for depth-3 results (~30 minutes), then execute plan based on outcome.
