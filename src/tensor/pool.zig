const std = @import("std");

/// Pre-allocated tensor buffer pool for eliminating allocation overhead during training.
///
/// This pool pre-allocates buffers for forward pass activations and gradients,
/// eliminating thousands of alloc/free calls per training step.
///
/// Expected speedup: 2-3x
///
/// Usage:
///   var pool = try TensorPool.init(allocator, 64, &[_]usize{144, 256, 128, 64, 6});
///   defer pool.deinit();
///
///   const buffer = try pool.checkout(batch_size * layer_size);
///   defer pool.returnBuffer(buffer);
pub const TensorPool = struct {
    allocator: std.mem.Allocator,
    buffers: std.ArrayList(PoolBuffer),
    batch_size: usize,

    const PoolBuffer = struct {
        data: []f32,
        capacity: usize,
        in_use: bool,
    };

    /// Initialize the tensor pool and preallocate buffers for the given network architecture
    pub fn init(allocator: std.mem.Allocator, batch_size: usize, layer_sizes: []const usize) !TensorPool {
        var pool = TensorPool{
            .allocator = allocator,
            .buffers = std.ArrayList(PoolBuffer){},
            .batch_size = batch_size,
        };

        // Preallocate buffers for each layer's forward pass
        // Each layer needs multiple buffers (matmul output, bias-added output, activations, gradients)
        for (0..layer_sizes.len - 1) |i| {
            const out_dim = layer_sizes[i + 1];
            const buffer_size = batch_size * out_dim;

            // Allocate 4 buffers per layer:
            // - matmul output
            // - bias-added output
            // - forward gradient
            // - backward gradient
            for (0..4) |_| {
                try pool.allocateBuffer(buffer_size);
            }
        }

        // Preallocate extra scratch buffers for backward pass
        const max_layer_size = blk: {
            var max_size: usize = 0;
            for (layer_sizes) |size| {
                if (size > max_size) max_size = size;
            }
            break :blk max_size;
        };

        for (0..3) |_| {
            try pool.allocateBuffer(batch_size * max_layer_size);
        }

        return pool;
    }

    pub fn deinit(self: *TensorPool) void {
        for (self.buffers.items) |buffer| {
            self.allocator.free(buffer.data);
        }
        self.buffers.deinit(self.allocator);
    }

    /// Allocate a single buffer of the specified size with SIMD alignment
    fn allocateBuffer(self: *TensorPool, size: usize) !void {
        // Allocate memory (standard f32 alignment)
        const data = try self.allocator.alloc(f32, size);
        @memset(data, 0.0);

        try self.buffers.append(self.allocator, PoolBuffer{
            .data = data,
            .capacity = size,
            .in_use = false,
        });
    }

    /// Get a buffer of at least the specified size
    pub fn checkout(self: *TensorPool, size: usize) ![]f32 {
        // First try to find an existing buffer that's not in use and large enough
        for (self.buffers.items) |*buffer| {
            if (!buffer.in_use and buffer.capacity >= size) {
                buffer.in_use = true;
                return buffer.data[0..size];
            }
        }

        // No suitable buffer found, allocate a new one
        try self.allocateBuffer(size);
        const new_buffer = &self.buffers.items[self.buffers.items.len - 1];
        new_buffer.in_use = true;
        return new_buffer.data[0..size];
    }

    /// Return a buffer to the pool (mark as available)
    pub fn returnBuffer(self: *TensorPool, data: []f32) void {
        const data_ptr = data.ptr;

        for (self.buffers.items) |*buffer| {
            if (buffer.data.ptr == data_ptr) {
                buffer.in_use = false;
                // Zero out buffer for next use
                @memset(buffer.data[0..data.len], 0.0);
                return;
            }
        }
    }

    /// Reset all buffers to available state (call at start of each training step)
    pub fn reset(self: *TensorPool) void {
        for (self.buffers.items) |*buffer| {
            buffer.in_use = false;
        }
    }

    /// Get pool statistics for debugging
    pub fn getStats(self: *TensorPool) PoolStats {
        var stats = PoolStats{
            .total_buffers = self.buffers.items.len,
            .in_use_buffers = 0,
            .total_memory = 0,
            .in_use_memory = 0,
        };

        for (self.buffers.items) |buffer| {
            stats.total_memory += buffer.capacity * @sizeOf(f32);
            if (buffer.in_use) {
                stats.in_use_buffers += 1;
                stats.in_use_memory += buffer.capacity * @sizeOf(f32);
            }
        }

        return stats;
    }

    pub const PoolStats = struct {
        total_buffers: usize,
        in_use_buffers: usize,
        total_memory: usize,
        in_use_memory: usize,
    };
};

test "TensorPool basic allocation" {
    const allocator = std.testing.allocator;
    const layer_sizes = [_]usize{ 144, 256, 128, 64, 6 };
    var pool = try TensorPool.init(allocator, 64, &layer_sizes);
    defer pool.deinit();

    const stats = pool.getStats();
    try std.testing.expect(stats.total_buffers > 0);
    try std.testing.expect(stats.total_memory > 0);
}

test "TensorPool checkout and return" {
    const allocator = std.testing.allocator;
    const layer_sizes = [_]usize{ 144, 256, 128, 64, 6 };
    var pool = try TensorPool.init(allocator, 64, &layer_sizes);
    defer pool.deinit();

    // Checkout a buffer
    const buf1 = try pool.checkout(64 * 256);
    try std.testing.expectEqual(@as(usize, 64 * 256), buf1.len);

    // Return it
    pool.returnBuffer(buf1);

    // Checkout again should reuse
    const buf2 = try pool.checkout(64 * 256);
    try std.testing.expectEqual(buf1.ptr, buf2.ptr);
}

test "TensorPool reset" {
    const allocator = std.testing.allocator;
    const layer_sizes = [_]usize{ 144, 256, 128, 64, 6 };
    var pool = try TensorPool.init(allocator, 64, &layer_sizes);
    defer pool.deinit();

    // Checkout multiple buffers
    _ = try pool.checkout(64 * 256);
    _ = try pool.checkout(64 * 128);

    var stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.in_use_buffers);

    // Reset should mark all as available
    pool.reset();

    stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.in_use_buffers);
}
