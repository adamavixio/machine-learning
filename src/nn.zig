/// Neural Network layer implementations
///
/// This module provides:
/// - DenseLayer: Fully connected layer with He initialization
/// - Model: Sequential container for multiple layers
/// - Activation functions: ReLU, Tanh
/// - Loss functions: MSE, Huber
///
/// Design follows ChatGPT's guidance:
/// - Struct-based layers that own parameter handles
/// - Layers allocate parameters during init via TensorContext
/// - Explicit activation functions for flexible composition
/// - Model container for clean multi-layer forward passes

const std = @import("std");

pub const dense = @import("nn/dense.zig");
pub const activations = @import("nn/activations.zig");
pub const loss = @import("nn/loss.zig");
pub const model = @import("nn/model.zig");
pub const optimizer = @import("nn/optimizer.zig");

// Re-export common types
pub const DenseLayer = dense.DenseLayer;
pub const Model = model.Model;

// Re-export activation functions
pub const relu = activations.relu;
pub const tanh = activations.tanh;

// Re-export loss functions
pub const mse = loss.mse;
pub const huber = loss.huber;

// Re-export optimizer functions
pub const sgdStep = optimizer.sgdStep;
pub const sgdStepClipped = optimizer.sgdStepClipped;

test {
    std.testing.refAllDecls(@This());
}
