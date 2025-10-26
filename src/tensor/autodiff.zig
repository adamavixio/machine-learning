const std = @import("std");
const config = @import("config.zig");
const tensor_mod = @import("tensor.zig");
const grad_mod = @import("grad.zig");
const ops_mod = @import("ops.zig");
const matmul_mod = @import("matmul.zig");

const Scalar = config.TensorConfig.Scalar;
const Tensor = tensor_mod.Tensor;
const TensorShape = tensor_mod.TensorShape;
const GradContext = grad_mod.GradContext;
const GradHandle = grad_mod.GradHandle;

/// Type of operation for backward pass
pub const OpType = enum {
    add,
    mul,
    matmul,
    relu,
    broadcast_add,
    gather_actions,
};

/// Backward function signature
pub const BackwardFn = *const fn (grad_ctx: *GradContext, op: *const Op) void;

/// Operation recorded on the tape
pub const Op = struct {
    op_type: OpType,
    result_handle: GradHandle,
    operand_handles: []const GradHandle,
    /// Cached data needed for backward pass
    cache: OpCache,

    pub const OpCache = union(OpType) {
        add: void,
        mul: MulCache,
        matmul: MatmulCache,
        relu: ReluCache,
        broadcast_add: BroadcastAddCache,
        gather_actions: GatherActionsCache,

        pub const MulCache = struct {
            a_data: []const Scalar,
            b_data: []const Scalar,
        };

        pub const MatmulCache = struct {
            a_data: []const Scalar,
            b_data: []const Scalar,
            M: usize,
            K: usize,
            N: usize,
        };

        pub const ReluCache = struct {
            input: []const Scalar,
        };

        pub const BroadcastAddCache = struct {
            a_shape: []const usize,
            b_shape: []const usize,
        };

        pub const GatherActionsCache = struct {
            actions: []const u8,
            batch_size: usize,
            num_actions: usize,
        };
    };
};

/// Tracked tensor that participates in autodiff
pub const TrackedTensor = struct {
    tensor: Tensor,
    grad_handle: GradHandle,

    pub fn data(self: TrackedTensor) []Scalar {
        return self.tensor.data;
    }

    pub fn shape(self: TrackedTensor) TensorShape {
        return self.tensor.shape;
    }
};

/// Autodiff context that owns the tape
pub const AutodiffContext = struct {
    allocator: std.mem.Allocator,
    tape: std.ArrayList(Op),
    grad_ctx: *GradContext,

    pub fn init(base_allocator: std.mem.Allocator, grad_ctx: *GradContext) AutodiffContext {
        return AutodiffContext{
            .allocator = base_allocator,
            .tape = std.ArrayList(Op){},
            .grad_ctx = grad_ctx,
        };
    }

    pub fn deinit(self: *AutodiffContext) void {
        // Free operand_handles arrays
        for (self.tape.items) |op| {
            self.allocator.free(op.operand_handles);
        }
        self.tape.deinit(self.allocator);
        // Don't deinit grad_ctx - it's owned by caller
    }

    /// Clear tape for next iteration (don't reset gradients - caller manages that)
    pub fn reset(self: *AutodiffContext) void {
        for (self.tape.items) |op| {
            self.allocator.free(op.operand_handles);
        }
        self.tape.clearRetainingCapacity();
        // Don't reset grad_ctx - caller manages gradients
    }

    /// Create a tracked tensor from raw data
    pub fn track(self: *AutodiffContext, tensor: Tensor) !TrackedTensor {
        const grad_handle = try self.grad_ctx.allocGrad(tensor.shape);
        return TrackedTensor{
            .tensor = tensor,
            .grad_handle = grad_handle,
        };
    }

    /// Seed the gradient for backward pass (typically loss = 1.0)
    pub fn seedGrad(self: *AutodiffContext, tracked: TrackedTensor, grad_value: Scalar) void {
        const grad = self.grad_ctx.getGrad(tracked.grad_handle);
        @memset(grad, grad_value);
    }

    /// Run backward pass
    pub fn backward(self: *AutodiffContext) void {
        var i = self.tape.items.len;
        while (i > 0) {
            i -= 1;
            const op = &self.tape.items[i];
            backwardOp(self.grad_ctx, op);
        }
    }

    /// Record an operation on the tape
    fn recordOp(
        self: *AutodiffContext,
        op_type: OpType,
        result_handle: GradHandle,
        operands: []const GradHandle,
        cache: Op.OpCache,
    ) !void {
        const operands_copy = try self.allocator.dupe(GradHandle, operands);
        try self.tape.append(self.allocator, Op{
            .op_type = op_type,
            .result_handle = result_handle,
            .operand_handles = operands_copy,
            .cache = cache,
        });
    }

    // ============ Tracked Operations ============

    /// Tracked element-wise addition: c = a + b
    pub fn trackedAdd(
        self: *AutodiffContext,
        a: TrackedTensor,
        b: TrackedTensor,
        c_tensor: Tensor,
    ) !TrackedTensor {
        // Forward pass
        try ops_mod.addInto(c_tensor.data, a.tensor.data, b.tensor.data);

        // Track gradient
        const c = try self.track(c_tensor);

        // Record operation
        const operands = [_]GradHandle{ a.grad_handle, b.grad_handle };
        try self.recordOp(.add, c.grad_handle, &operands, .{ .add = {} });

        return c;
    }

    /// Tracked element-wise multiplication: c = a * b
    pub fn trackedMul(
        self: *AutodiffContext,
        a: TrackedTensor,
        b: TrackedTensor,
        c_tensor: Tensor,
    ) !TrackedTensor {
        // Forward pass
        try ops_mod.mulInto(c_tensor.data, a.tensor.data, b.tensor.data);

        // Track gradient
        const c = try self.track(c_tensor);

        // Record operation with cached inputs for backward
        const operands = [_]GradHandle{ a.grad_handle, b.grad_handle };
        try self.recordOp(.mul, c.grad_handle, &operands, .{
            .mul = .{
                .a_data = a.tensor.data,
                .b_data = b.tensor.data,
            },
        });

        return c;
    }

    /// Tracked matrix multiplication: C = A @ B
    pub fn trackedMatmul(
        self: *AutodiffContext,
        a: TrackedTensor,
        b: TrackedTensor,
        M: usize,
        K: usize,
        N: usize,
        c_tensor: Tensor,
    ) !TrackedTensor {
        // Forward pass
        try matmul_mod.matmul(c_tensor.data, a.tensor.data, b.tensor.data, M, K, N);

        // Track gradient
        const c = try self.track(c_tensor);

        // Record operation
        const operands = [_]GradHandle{ a.grad_handle, b.grad_handle };
        try self.recordOp(.matmul, c.grad_handle, &operands, .{
            .matmul = .{
                .a_data = a.tensor.data,
                .b_data = b.tensor.data,
                .M = M,
                .K = K,
                .N = N,
            },
        });

        return c;
    }

    /// Tracked ReLU activation: y = max(0, x)
    pub fn trackedRelu(
        self: *AutodiffContext,
        x: TrackedTensor,
        y_tensor: Tensor,
    ) !TrackedTensor {
        // Forward pass
        try ops_mod.reluInto(y_tensor.data, x.tensor.data);

        // Track gradient
        const y = try self.track(y_tensor);

        // Record operation
        const operands = [_]GradHandle{x.grad_handle};
        try self.recordOp(.relu, y.grad_handle, &operands, .{
            .relu = .{
                .input = x.tensor.data,
            },
        });

        return y;
    }

    /// Tracked broadcast addition: c = a + b (with broadcasting)
    pub fn trackedBroadcastAdd(
        self: *AutodiffContext,
        a: TrackedTensor,
        b: TrackedTensor,
        c_tensor: Tensor,
    ) !TrackedTensor {
        // Forward pass
        try ops_mod.broadcastAddInto(
            c_tensor.data,
            a.tensor.data,
            b.tensor.data,
            a.shape().dims,
            b.shape().dims,
        );

        // Track gradient
        const c = try self.track(c_tensor);

        // Cache shapes for backward pass
        const allocator = std.heap.page_allocator;
        const a_shape_copy = try allocator.dupe(usize, a.shape().dims);
        const b_shape_copy = try allocator.dupe(usize, b.shape().dims);

        // Record operation
        const operands = [_]GradHandle{ a.grad_handle, b.grad_handle };
        try self.recordOp(.broadcast_add, c.grad_handle, &operands, .{
            .broadcast_add = .{
                .a_shape = a_shape_copy,
                .b_shape = b_shape_copy,
            },
        });

        return c;
    }

    /// Tracked gather operation for action selection
    /// Given q_all [batch_size, num_actions] and actions [batch_size],
    /// returns q_sa [batch_size] where q_sa[i] = q_all[i, actions[i]]
    pub fn trackedGatherActions(
        self: *AutodiffContext,
        q_all: TrackedTensor,
        actions: []const u8,
        batch_size: usize,
        num_actions: usize,
        q_sa_tensor: Tensor,
    ) !TrackedTensor {
        // Forward pass: gather Q-values for selected actions
        for (0..batch_size) |i| {
            const action = actions[i];
            const offset = i * num_actions + action;
            q_sa_tensor.data[i] = q_all.tensor.data[offset];
        }

        // Track gradient
        const q_sa = try self.track(q_sa_tensor);

        // Cache actions for backward pass
        const allocator = std.heap.page_allocator;
        const actions_copy = try allocator.dupe(u8, actions);

        // Record operation
        const operands = [_]GradHandle{q_all.grad_handle};
        try self.recordOp(.gather_actions, q_sa.grad_handle, &operands, .{
            .gather_actions = .{
                .actions = actions_copy,
                .batch_size = batch_size,
                .num_actions = num_actions,
            },
        });

        return q_sa;
    }
};

/// Execute backward pass for a single operation
fn backwardOp(grad_ctx: *GradContext, op: *const Op) void {
    switch (op.op_type) {
        .add => backwardAdd(grad_ctx, op),
        .mul => backwardMul(grad_ctx, op),
        .matmul => backwardMatmul(grad_ctx, op),
        .relu => backwardRelu(grad_ctx, op),
        .broadcast_add => backwardBroadcastAdd(grad_ctx, op),
        .gather_actions => backwardGatherActions(grad_ctx, op),
    }
}

/// Backward pass for addition: da = dc, db = dc
fn backwardAdd(grad_ctx: *GradContext, op: *const Op) void {
    const dc = grad_ctx.getGrad(op.result_handle);
    const a_handle = op.operand_handles[0];
    const b_handle = op.operand_handles[1];

    grad_ctx.accumulate(a_handle, dc);
    grad_ctx.accumulate(b_handle, dc);
}

/// Backward pass for multiplication: da = dc * b, db = dc * a
fn backwardMul(grad_ctx: *GradContext, op: *const Op) void {
    const dc = grad_ctx.getGrad(op.result_handle);
    const a_handle = op.operand_handles[0];
    const b_handle = op.operand_handles[1];

    const cache = op.cache.mul;

    // Allocate temporary buffers for gradients
    const allocator = std.heap.page_allocator;
    const da_temp = allocator.alloc(Scalar, dc.len) catch return;
    defer allocator.free(da_temp);
    const db_temp = allocator.alloc(Scalar, dc.len) catch return;
    defer allocator.free(db_temp);

    // da = dc * b
    ops_mod.mulInto(da_temp, dc, cache.b_data) catch return;
    grad_ctx.accumulate(a_handle, da_temp);

    // db = dc * a
    ops_mod.mulInto(db_temp, dc, cache.a_data) catch return;
    grad_ctx.accumulate(b_handle, db_temp);
}

/// Backward pass for matmul: dA = dC @ B^T, dB = A^T @ dC
fn backwardMatmul(grad_ctx: *GradContext, op: *const Op) void {
    const dC = grad_ctx.getGrad(op.result_handle);
    const a_handle = op.operand_handles[0];
    const b_handle = op.operand_handles[1];

    const cache = op.cache.matmul;
    const M = cache.M;
    const K = cache.K;
    const N = cache.N;

    // Allocate temporary buffers
    const allocator = std.heap.page_allocator;
    const dA = allocator.alloc(Scalar, M * K) catch return;
    defer allocator.free(dA);
    const dB = allocator.alloc(Scalar, K * N) catch return;
    defer allocator.free(dB);

    // dA = dC @ B^T (M x N @ N x K = M x K)
    matmulTransposeB(dA, dC, cache.b_data, M, N, K) catch return;
    grad_ctx.accumulate(a_handle, dA);

    // dB = A^T @ dC (K x M @ M x N = K x N)
    matmulTransposeA(dB, cache.a_data, dC, M, K, N) catch return;
    grad_ctx.accumulate(b_handle, dB);
}

/// Backward pass for ReLU: dx = dc * (x > 0)
fn backwardRelu(grad_ctx: *GradContext, op: *const Op) void {
    const dy = grad_ctx.getGrad(op.result_handle);
    const x_handle = op.operand_handles[0];
    const cache = op.cache.relu;

    const allocator = std.heap.page_allocator;
    const dx = allocator.alloc(Scalar, dy.len) catch return;
    defer allocator.free(dx);

    for (dy, 0..) |grad_val, i| {
        dx[i] = if (cache.input[i] > 0.0) grad_val else 0.0;
    }

    grad_ctx.accumulate(x_handle, dx);
}

/// Backward pass for broadcast add
/// For [batch, features] + [features]:
/// - da = dc (gradient passes through)
/// - db = sum(dc over batch dimension)
fn backwardBroadcastAdd(grad_ctx: *GradContext, op: *const Op) void {
    const dc = grad_ctx.getGrad(op.result_handle);
    const a_handle = op.operand_handles[0];
    const b_handle = op.operand_handles[1];

    const cache = op.cache.broadcast_add;
    const a_shape = cache.a_shape;
    const b_shape = cache.b_shape;

    // Gradient for 'a' passes through as-is
    grad_ctx.accumulate(a_handle, dc);

    // Gradient for 'b' needs to be summed over broadcast dimensions
    // Handle [batch, features] + [features] case
    if (a_shape.len == 2 and b_shape.len == 1) {
        const batch = a_shape[0];
        const features = a_shape[1];

        const allocator = std.heap.page_allocator;
        const db = allocator.alloc(Scalar, features) catch return;
        defer allocator.free(db);

        @memset(db, 0.0);

        // Sum gradients across batch dimension
        for (0..batch) |i| {
            const offset = i * features;
            for (0..features) |j| {
                db[j] += dc[offset + j];
            }
        }

        grad_ctx.accumulate(b_handle, db);
        return;
    }

    // Handle [batch, features] + [1, features] case
    if (a_shape.len == 2 and b_shape.len == 2 and b_shape[0] == 1) {
        const batch = a_shape[0];
        const features = a_shape[1];

        const allocator = std.heap.page_allocator;
        const db = allocator.alloc(Scalar, features) catch return;
        defer allocator.free(db);

        @memset(db, 0.0);

        // Sum gradients across batch dimension
        for (0..batch) |i| {
            const offset = i * features;
            for (0..features) |j| {
                db[j] += dc[offset + j];
            }
        }

        grad_ctx.accumulate(b_handle, db);
        return;
    }
}

/// Backward pass for gather_actions: scatter gradients back to source tensor
/// For q_sa[i] = q_all[i, actions[i]], we have:
/// dq_all[i, actions[i]] += dq_sa[i]
fn backwardGatherActions(grad_ctx: *GradContext, op: *const Op) void {
    const dq_sa = grad_ctx.getGrad(op.result_handle);
    const q_all_handle = op.operand_handles[0];

    const cache = op.cache.gather_actions;
    const actions = cache.actions;
    const batch_size = cache.batch_size;
    const num_actions = cache.num_actions;

    // Allocate gradient buffer for q_all [batch_size, num_actions]
    const allocator = std.heap.page_allocator;
    const dq_all = allocator.alloc(Scalar, batch_size * num_actions) catch return;
    defer allocator.free(dq_all);

    @memset(dq_all, 0.0);

    // Scatter gradients: dq_all[i, actions[i]] = dq_sa[i]
    for (0..batch_size) |i| {
        const action = actions[i];
        const offset = i * num_actions + action;
        dq_all[offset] = dq_sa[i];
    }

    grad_ctx.accumulate(q_all_handle, dq_all);
}

// ============ Helper matmul variants for backward pass ============

/// Matrix multiply with B transposed: C = A @ B^T
fn matmulTransposeB(
    C: []Scalar,
    A: []const Scalar,
    B: []const Scalar,
    M: usize,
    N: usize,
    K: usize,
) !void {
    @memset(C, 0.0);
    for (0..M) |i| {
        for (0..K) |k| {
            for (0..N) |n| {
                // A[i,n] * B[k,n] (B is transposed)
                C[i * K + k] += A[i * N + n] * B[k * N + n];
            }
        }
    }
}

/// Matrix multiply with A transposed: C = A^T @ B
fn matmulTransposeA(
    C: []Scalar,
    A: []const Scalar,
    B: []const Scalar,
    M: usize,
    K: usize,
    N: usize,
) !void {
    @memset(C, 0.0);
    for (0..K) |k| {
        for (0..N) |n| {
            for (0..M) |m| {
                // A[m,k] transposed = A^T[k,m], B[m,n]
                C[k * N + n] += A[m * K + k] * B[m * N + n];
            }
        }
    }
}

test "autodiff add" {
    var grad_ctx = GradContext.init(std.testing.allocator);
    defer grad_ctx.deinit();

    var ad_ctx = AutodiffContext.init(std.testing.allocator, &grad_ctx);
    defer ad_ctx.deinit();

    // Create input tensors
    const allocator = std.testing.allocator;
    const a_data = try allocator.alloc(Scalar, 3);
    defer allocator.free(a_data);
    const b_data = try allocator.alloc(Scalar, 3);
    defer allocator.free(b_data);
    const c_data = try allocator.alloc(Scalar, 3);
    defer allocator.free(c_data);

    a_data[0] = 1.0;
    a_data[1] = 2.0;
    a_data[2] = 3.0;
    b_data[0] = 4.0;
    b_data[1] = 5.0;
    b_data[2] = 6.0;

    const dims = [_]usize{3};
    const a_tensor = try Tensor.init(&dims, a_data);
    const b_tensor = try Tensor.init(&dims, b_data);
    const c_tensor = try Tensor.init(&dims, c_data);

    const a = try ad_ctx.track(a_tensor);
    const b = try ad_ctx.track(b_tensor);
    const c = try ad_ctx.trackedAdd(a, b, c_tensor);

    // Seed gradient
    ad_ctx.seedGrad(c, 1.0);

    // Backward pass
    ad_ctx.backward();

    // Check gradients
    const da = ad_ctx.grad_ctx.getGrad(a.grad_handle);
    const db = ad_ctx.grad_ctx.getGrad(b.grad_handle);

    try std.testing.expectApproxEqAbs(@as(Scalar, 1.0), da[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 1.0), db[0], 1e-5);
}
