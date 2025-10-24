const std = @import("std");
const tensor_mod = @import("../tensor.zig");

const Tensor = tensor_mod.Tensor;
const AutodiffContext = tensor_mod.AutodiffContext;
const TrackedTensor = tensor_mod.TrackedTensor;

/// ReLU activation: y = max(0, x)
pub fn relu(
    ad_ctx: *AutodiffContext,
    x: TrackedTensor,
    output_tensor: Tensor,
) !TrackedTensor {
    return ad_ctx.trackedRelu(x, output_tensor);
}

/// Tanh activation: y = tanh(x)
/// Note: This is a placeholder. Full implementation would require trackedTanh in autodiff
pub fn tanh(
    x: TrackedTensor,
    output_tensor: Tensor,
) !TrackedTensor {
    _ = x;
    _ = output_tensor;
    return error.NotImplemented;
}
