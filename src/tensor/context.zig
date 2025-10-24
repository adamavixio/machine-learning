const std = @import("std");
const config = @import("config.zig");
const tensor_mod = @import("tensor.zig");

const Scalar = config.TensorConfig.Scalar;
const Tensor = tensor_mod.Tensor;

/// Arena-based tensor context for efficient memory management
pub const TensorContext = struct {
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,

    pub fn init(base_allocator: std.mem.Allocator) TensorContext {
        return TensorContext{
            .allocator = base_allocator,
            .arena = std.heap.ArenaAllocator.init(base_allocator),
        };
    }

    pub fn deinit(self: *TensorContext) void {
        self.arena.deinit();
    }

    /// Allocate a new tensor with the given shape, aligned for SIMD
    pub fn allocTensor(self: *TensorContext, dims: []const usize) !Tensor {
        const arena_allocator = self.arena.allocator();

        // Calculate total size
        var total_size: usize = 1;
        for (dims) |d| {
            total_size *= d;
        }

        // Allocate memory (Zig's allocator handles alignment for SIMD types)
        const data = try arena_allocator.alloc(Scalar, total_size);

        // Initialize to zero
        @memset(data, 0.0);

        // Create dims copy
        const dims_copy = try arena_allocator.dupe(usize, dims);

        // Calculate row-major strides
        const strides = try arena_allocator.alloc(usize, dims.len);
        var stride: usize = 1;
        var i = dims.len;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= dims[i];
        }

        return Tensor{
            .shape = tensor_mod.TensorShape{
                .dims = dims_copy,
                .strides = strides,
            },
            .data = data,
        };
    }

    /// Reset the arena, freeing all allocated tensors
    pub fn reset(self: *TensorContext) void {
        _ = self.arena.reset(.retain_capacity);
    }
};

test "tensor context allocation" {
    var ctx = TensorContext.init(std.testing.allocator);
    defer ctx.deinit();

    const t1 = try ctx.allocTensor(&[_]usize{ 2, 3 });
    try std.testing.expectEqual(@as(usize, 6), t1.data.len);
    try std.testing.expectEqual(@as(usize, 2), t1.shape.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), t1.shape.dims[1]);

    // Check alignment
    const required_alignment = @alignOf(@Vector(config.TensorConfig.simd_lanes, Scalar));
    const addr = @intFromPtr(t1.data.ptr);
    try std.testing.expectEqual(@as(usize, 0), addr % required_alignment);
}
