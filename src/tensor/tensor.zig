const std = @import("std");
const config = @import("config.zig");
const Scalar = config.TensorConfig.Scalar;

/// Shape and stride information for tensors
pub const TensorShape = struct {
    dims: []const usize,
    strides: []const usize,

    pub fn ndim(self: TensorShape) usize {
        return self.dims.len;
    }

    pub fn size(self: TensorShape) usize {
        var total: usize = 1;
        for (self.dims) |d| {
            total *= d;
        }
        return total;
    }

    pub fn isContiguous(self: TensorShape) bool {
        if (self.dims.len == 0) return true;
        var expected_stride: usize = 1;
        var i = self.dims.len;
        while (i > 0) {
            i -= 1;
            if (self.strides[i] != expected_stride) return false;
            expected_stride *= self.dims[i];
        }
        return true;
    }
};

/// Tensor view backed by arena-allocated memory
pub const Tensor = struct {
    shape: TensorShape,
    data: []Scalar,

    pub fn init(dims: []const usize, data: []Scalar) !Tensor {
        const total_size = blk: {
            var size: usize = 1;
            for (dims) |d| size *= d;
            break :blk size;
        };

        if (data.len != total_size) {
            return error.ShapeMismatch;
        }

        // Row-major strides
        const strides = try std.heap.page_allocator.alloc(usize, dims.len);
        var stride: usize = 1;
        var i = dims.len;
        while (i > 0) {
            i -= 1;
            strides[i] = stride;
            stride *= dims[i];
        }

        return Tensor{
            .shape = TensorShape{
                .dims = dims,
                .strides = strides,
            },
            .data = data,
        };
    }

    pub fn at(self: Tensor, indices: []const usize) !*Scalar {
        if (indices.len != self.shape.ndim()) {
            return error.InvalidIndex;
        }

        var offset: usize = 0;
        for (indices, 0..) |idx, i| {
            if (idx >= self.shape.dims[i]) {
                return error.IndexOutOfBounds;
            }
            offset += idx * self.shape.strides[i];
        }

        return &self.data[offset];
    }

    pub fn get(self: Tensor, indices: []const usize) !Scalar {
        const ptr = try self.at(indices);
        return ptr.*;
    }

    pub fn set(self: Tensor, indices: []const usize, value: Scalar) !void {
        const ptr = try self.at(indices);
        ptr.* = value;
    }
};

test "tensor shape size" {
    const dims = [_]usize{ 2, 3, 4 };
    const shape = TensorShape{
        .dims = &dims,
        .strides = &[_]usize{ 12, 4, 1 },
    };
    try std.testing.expectEqual(@as(usize, 24), shape.size());
}

test "tensor creation and access" {
    const allocator = std.testing.allocator;
    const dims = [_]usize{ 2, 3 };
    const data = try allocator.alloc(Scalar, 6);
    defer allocator.free(data);

    for (data, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }

    const tensor = try Tensor.init(&dims, data);
    try std.testing.expectEqual(@as(Scalar, 0.0), try tensor.get(&[_]usize{ 0, 0 }));
    try std.testing.expectEqual(@as(Scalar, 5.0), try tensor.get(&[_]usize{ 1, 2 }));
}
