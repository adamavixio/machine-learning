const std = @import("std");
const tensor_mod = @import("../tensor.zig");

const GradContext = tensor_mod.GradContext;
const GradHandle = tensor_mod.GradHandle;
const Tensor = tensor_mod.Tensor;

/// SGD optimizer step with optional gradient clipping: param -= lr * grad
pub fn sgdStep(
    params: []Tensor,
    param_handles: []const GradHandle,
    grad_ctx: *GradContext,
    learning_rate: f32,
) void {
    sgdStepClipped(params, param_handles, grad_ctx, learning_rate, null);
}

/// SGD optimizer step with gradient clipping
pub fn sgdStepClipped(
    params: []Tensor,
    param_handles: []const GradHandle,
    grad_ctx: *GradContext,
    learning_rate: f32,
    max_grad_norm: ?f32,
) void {
    std.debug.assert(params.len == param_handles.len);

    for (params, param_handles) |param, handle| {
        const grad = grad_ctx.getGrad(handle);

        // Update with optional gradient clipping: param -= lr * clip(grad)
        for (param.data, grad) |*p, g| {
            const clipped_g = if (max_grad_norm) |max_norm|
                @max(-max_norm, @min(max_norm, g))
            else
                g;
            p.* -= learning_rate * clipped_g;
        }
    }
}

test "sgd step" {
    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    // Create a simple parameter
    const allocator = std.testing.allocator;
    const param_data = try allocator.alloc(f32, 3);
    defer allocator.free(param_data);

    param_data[0] = 1.0;
    param_data[1] = 2.0;
    param_data[2] = 3.0;

    const dims = [_]usize{3};
    const param = try Tensor.init(&dims, param_data);

    // Register gradient
    const handle = try grad_ctx.allocGrad(param.shape);

    // Set gradient
    const grad = grad_ctx.getGrad(handle);
    grad[0] = 0.1;
    grad[1] = 0.2;
    grad[2] = 0.3;

    // SGD step with lr=0.1
    var params = [_]Tensor{param};
    const handles = [_]GradHandle{handle};
    sgdStep(&params, &handles, &grad_ctx, 0.1);

    // Check updated values: param -= 0.1 * grad
    try std.testing.expectApproxEqAbs(@as(f32, 0.99), param_data[0], 1e-5); // 1.0 - 0.1*0.1
    try std.testing.expectApproxEqAbs(@as(f32, 1.98), param_data[1], 1e-5); // 2.0 - 0.1*0.2
    try std.testing.expectApproxEqAbs(@as(f32, 2.97), param_data[2], 1e-5); // 3.0 - 0.1*0.3
}
