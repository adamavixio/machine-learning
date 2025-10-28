const std = @import("std");
const config = @import("config.zig");
const tensor_mod = @import("tensor.zig");

const Scalar = config.TensorConfig.Scalar;
const Tensor = tensor_mod.Tensor;
const TensorShape = tensor_mod.TensorShape;

/// Handle to a gradient in the GradContext
pub const GradHandle = usize;

/// Gradient context for managing gradients during backward pass
pub const GradContext = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    /// Gradient storage: maps handle to gradient tensor data
    grads: std.ArrayList([]Scalar),
    /// Shapes for each gradient
    shapes: std.ArrayList(TensorShape),

    pub fn init(base_allocator: std.mem.Allocator) GradContext {
        return GradContext{
            .allocator = base_allocator,
            .arena = std.heap.ArenaAllocator.init(base_allocator),
            .grads = std.ArrayList([]Scalar){},
            .shapes = std.ArrayList(TensorShape){},
        };
    }

    pub fn deinit(self: *GradContext) void {
        self.grads.deinit(self.allocator);
        self.shapes.deinit(self.allocator);
        self.arena.deinit();
    }

    /// Allocate a new gradient tensor with the given shape
    pub fn allocGrad(self: *GradContext, shape: TensorShape) !GradHandle {
        const arena_allocator = self.arena.allocator();
        const total_size = shape.size();

        const grad_data = try arena_allocator.alloc(Scalar, total_size);
        @memset(grad_data, 0.0);

        // Copy shape
        const dims_copy = try arena_allocator.dupe(usize, shape.dims);
        const strides_copy = try arena_allocator.dupe(usize, shape.strides);

        const shape_copy = TensorShape{
            .dims = dims_copy,
            .strides = strides_copy,
        };

        const handle = self.grads.items.len;
        try self.grads.append(self.allocator, grad_data);
        try self.shapes.append(self.allocator, shape_copy);

        return handle;
    }

    /// Get gradient data by handle
    pub fn getGrad(self: *GradContext, handle: GradHandle) []Scalar {
        return self.grads.items[handle];
    }

    /// Get gradient shape by handle
    pub fn getShape(self: *GradContext, handle: GradHandle) TensorShape {
        return self.shapes.items[handle];
    }

    /// Accumulate gradient: grad[handle] += value
    pub fn accumulate(self: *GradContext, handle: GradHandle, value: []const Scalar) void {
        const grad = self.grads.items[handle];
        const n = @min(grad.len, value.len);
        for (0..n) |i| {
            grad[i] += value[i];
        }
    }

    /// Set gradient to a specific value (used for seeding loss gradient)
    pub fn setGrad(self: *GradContext, handle: GradHandle, value: []const Scalar) void {
        const grad = self.grads.items[handle];
        const n = @min(grad.len, value.len);
        @memcpy(grad[0..n], value[0..n]);
    }

    /// Get total bytes in all gradient tensors (for diagnostics)
    pub fn getTotalBytes(self: *GradContext) usize {
        var total_bytes: usize = 0;
        for (self.grads.items) |grad| {
            total_bytes += grad.len * @sizeOf(Scalar);
        }
        return total_bytes;
    }

    /// Reset all gradients to zero
    pub fn zeroGrads(self: *GradContext) void {
        for (self.grads.items) |grad| {
            @memset(grad, 0.0);
        }
    }

    /// Clear all gradients and reset arena
    pub fn reset(self: *GradContext) void {
        self.grads.clearRetainingCapacity();
        self.shapes.clearRetainingCapacity();
        _ = self.arena.reset(.retain_capacity);
    }
};

test "grad context allocation" {
    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    const shape = TensorShape{
        .dims = &[_]usize{ 2, 3 },
        .strides = &[_]usize{ 3, 1 },
    };

    const handle = try grad_ctx.allocGrad(shape);
    try std.testing.expectEqual(@as(GradHandle, 0), handle);

    const grad = grad_ctx.getGrad(handle);
    try std.testing.expectEqual(@as(usize, 6), grad.len);

    // Should be zero-initialized
    for (grad) |val| {
        try std.testing.expectEqual(@as(Scalar, 0.0), val);
    }
}

test "grad context accumulate" {
    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    const shape = TensorShape{
        .dims = &[_]usize{3},
        .strides = &[_]usize{1},
    };

    const handle = try grad_ctx.allocGrad(shape);

    const values = [_]Scalar{ 1.0, 2.0, 3.0 };
    grad_ctx.accumulate(handle, &values);
    grad_ctx.accumulate(handle, &values);

    const grad = grad_ctx.getGrad(handle);
    try std.testing.expectEqual(@as(Scalar, 2.0), grad[0]);
    try std.testing.expectEqual(@as(Scalar, 4.0), grad[1]);
    try std.testing.expectEqual(@as(Scalar, 6.0), grad[2]);
}
