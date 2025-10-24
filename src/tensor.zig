/// Tensor operations with SIMD acceleration
///
/// This module provides:
/// - TensorContext: Arena-based memory management for tensors
/// - Tensor: Multi-dimensional array view with shape/stride
/// - SIMD-accelerated operations: matmul, elementwise ops, reductions
///
/// Design follows ChatGPT's guidance:
/// - Context owns allocator and scratch buffers
/// - SIMD width auto-detected via std.simd.suggestVectorLength
/// - Aligned allocations for SIMD efficiency
/// - Safe wrappers validate dimensions, kernels use raw pointers

const std = @import("std");

pub const config = @import("tensor/config.zig");
pub const tensor = @import("tensor/tensor.zig");
pub const context = @import("tensor/context.zig");
pub const ops = @import("tensor/ops.zig");
pub const matmul = @import("tensor/matmul.zig");
pub const grad = @import("tensor/grad.zig");
pub const autodiff = @import("tensor/autodiff.zig");

// Re-export common types
pub const TensorConfig = config.TensorConfig;
pub const Tensor = tensor.Tensor;
pub const TensorShape = tensor.TensorShape;
pub const TensorContext = context.TensorContext;

// Re-export autodiff types
pub const GradContext = grad.GradContext;
pub const GradHandle = grad.GradHandle;
pub const AutodiffContext = autodiff.AutodiffContext;
pub const TrackedTensor = autodiff.TrackedTensor;
pub const OpType = autodiff.OpType;

// Re-export operations
pub const addInto = ops.addInto;
pub const mulInto = ops.mulInto;
pub const reluInto = ops.reluInto;
pub const tanhInto = ops.tanhInto;
pub const sum = ops.sum;
pub const max = ops.max;
pub const mean = ops.mean;
pub const broadcastAddInto = ops.broadcastAddInto;
pub const gatherActions = ops.gatherActions;
pub const maxAlongAxis1 = ops.maxAlongAxis1;
pub const matmulInto = matmul.matmul;

test {
    std.testing.refAllDecls(@This());
}
