const std = @import("std");
const tensor_mod = @import("../tensor.zig");

const Tensor = tensor_mod.Tensor;
const AutodiffContext = tensor_mod.AutodiffContext;
const TrackedTensor = tensor_mod.TrackedTensor;
const Scalar = tensor_mod.TensorConfig.Scalar;

/// Mean Squared Error loss: MSE = mean((pred - target)^2)
pub fn mse(
    ad_ctx: *AutodiffContext,
    pred: TrackedTensor,
    target: TrackedTensor,
    diff_tensor: Tensor,
    squared_tensor: Tensor,
) !TrackedTensor {
    // diff = pred - target (implemented as pred + (-1 * target))
    const allocator = std.heap.page_allocator;
    const neg_target_data = try allocator.alloc(Scalar, target.data().len);
    defer allocator.free(neg_target_data);

    for (target.data(), 0..) |val, i| {
        neg_target_data[i] = -val;
    }

    const neg_target_tensor = try Tensor.init(target.shape().dims, neg_target_data);
    const neg_target_tracked = try ad_ctx.track(neg_target_tensor);

    // diff = pred + neg_target
    const diff = try ad_ctx.trackedAdd(pred, neg_target_tracked, diff_tensor);

    // squared = diff * diff
    const squared = try ad_ctx.trackedMul(diff, diff, squared_tensor);

    return squared;
}

/// Huber loss: smooth combination of L1 and L2 loss
/// For |x| <= delta: L = 0.5 * x^2
/// For |x| > delta: L = delta * (|x| - 0.5 * delta)
pub fn huber(
    pred: TrackedTensor,
    target: TrackedTensor,
    delta: Scalar,
) !TrackedTensor {
    _ = pred;
    _ = target;
    _ = delta;
    return error.NotImplemented;
}
