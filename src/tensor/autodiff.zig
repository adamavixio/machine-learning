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
    dueling_q,
};

/// Backward function signature (updated to use both contexts)
pub const BackwardFn = *const fn (param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext, op: *const Op) void;

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
        dueling_q: DuelingQCache,

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
            // Optional scratch buffers to avoid allocation overhead in backward pass
            // If null, backwardMatmul will fall back to page allocator
            scratch_dA: ?[]Scalar = null,
            scratch_dB: ?[]Scalar = null,
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

        pub const DuelingQCache = struct {
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
    param_grad_ctx: *GradContext,  // For long-lived parameter gradients (W, b)
    temp_grad_ctx: *GradContext,   // For short-lived intermediate gradients
    /// Number of parameter gradients - used to distinguish handle contexts
    /// Handles [0, param_count) -> param_grad_ctx
    /// Handles [param_count, ...) -> temp_grad_ctx (with offset subtraction)
    param_count: usize,

    pub fn init(base_allocator: std.mem.Allocator, param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext) AutodiffContext {
        return AutodiffContext{
            .allocator = base_allocator,
            .tape = std.ArrayList(Op){},
            .param_grad_ctx = param_grad_ctx,
            .temp_grad_ctx = temp_grad_ctx,
            .param_count = param_grad_ctx.grads.items.len,
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

    /// Create a tracked tensor from raw data (uses temp_grad_ctx for intermediates)
    pub fn track(self: *AutodiffContext, tensor: Tensor) !TrackedTensor {
        const temp_handle = try self.temp_grad_ctx.allocGrad(tensor.shape);
        // Offset handle to distinguish from parameter handles
        const grad_handle = self.param_count + temp_handle;
        return TrackedTensor{
            .tensor = tensor,
            .grad_handle = grad_handle,
        };
    }

    /// Seed the gradient for backward pass (typically loss = 1.0)
    pub fn seedGrad(self: *AutodiffContext, tracked: TrackedTensor, grad_value: Scalar) void {
        // Tracked tensors from track() are always in temp_grad_ctx with offset
        const temp_handle = tracked.grad_handle - self.param_count;
        const grad = self.temp_grad_ctx.getGrad(temp_handle);
        @memset(grad, grad_value);
    }

    /// Run backward pass (needs both contexts: temp for intermediates, param for parameters)
    pub fn backward(self: *AutodiffContext) void {
        // Timing instrumentation for backward pass
        const BackwardTiming = struct {
            var matmul_time: i128 = 0;
            var relu_time: i128 = 0;
            var broadcast_add_time: i128 = 0;
            var gather_actions_time: i128 = 0;
            var add_time: i128 = 0;
            var mul_time: i128 = 0;
            var call_count: usize = 0;
        };

        BackwardTiming.call_count += 1;

        var i = self.tape.items.len;
        while (i > 0) {
            i -= 1;
            const op = &self.tape.items[i];

            const t_start = std.time.nanoTimestamp();
            backwardOp(self.param_grad_ctx, self.temp_grad_ctx, self.param_count, op);
            const t_elapsed = std.time.nanoTimestamp() - t_start;

            // Accumulate timing by op type
            switch (op.op_type) {
                .matmul => BackwardTiming.matmul_time += t_elapsed,
                .relu => BackwardTiming.relu_time += t_elapsed,
                .broadcast_add => BackwardTiming.broadcast_add_time += t_elapsed,
                .gather_actions => BackwardTiming.gather_actions_time += t_elapsed,
                .add => BackwardTiming.add_time += t_elapsed,
                .mul => BackwardTiming.mul_time += t_elapsed,
                .dueling_q => {}, // No separate timing tracking for dueling_q yet
            }
        }

        // Print breakdown every 100 backward calls (after warmup)
        if (BackwardTiming.call_count % 100 == 0 and BackwardTiming.call_count > 100) {
            const total_time = BackwardTiming.matmul_time + BackwardTiming.relu_time +
                BackwardTiming.broadcast_add_time + BackwardTiming.gather_actions_time +
                BackwardTiming.add_time + BackwardTiming.mul_time;

            std.debug.print("  [BACKWARD BREAKDOWN] Call {d}:\n", .{BackwardTiming.call_count});
            std.debug.print("    matmul:        {d:.1}ms ({d:.1}%)\n", .{
                @as(f64, @floatFromInt(BackwardTiming.matmul_time)) / 1_000_000.0,
                @as(f64, @floatFromInt(BackwardTiming.matmul_time)) * 100.0 / @as(f64, @floatFromInt(total_time)),
            });
            std.debug.print("    relu:          {d:.1}ms ({d:.1}%)\n", .{
                @as(f64, @floatFromInt(BackwardTiming.relu_time)) / 1_000_000.0,
                @as(f64, @floatFromInt(BackwardTiming.relu_time)) * 100.0 / @as(f64, @floatFromInt(total_time)),
            });
            std.debug.print("    broadcast_add: {d:.1}ms ({d:.1}%)\n", .{
                @as(f64, @floatFromInt(BackwardTiming.broadcast_add_time)) / 1_000_000.0,
                @as(f64, @floatFromInt(BackwardTiming.broadcast_add_time)) * 100.0 / @as(f64, @floatFromInt(total_time)),
            });
            std.debug.print("    gather:        {d:.1}ms ({d:.1}%)\n", .{
                @as(f64, @floatFromInt(BackwardTiming.gather_actions_time)) / 1_000_000.0,
                @as(f64, @floatFromInt(BackwardTiming.gather_actions_time)) * 100.0 / @as(f64, @floatFromInt(total_time)),
            });
            std.debug.print("    add:           {d:.1}ms ({d:.1}%)\n", .{
                @as(f64, @floatFromInt(BackwardTiming.add_time)) / 1_000_000.0,
                @as(f64, @floatFromInt(BackwardTiming.add_time)) * 100.0 / @as(f64, @floatFromInt(total_time)),
            });
            std.debug.print("    mul:           {d:.1}ms ({d:.1}%)\n", .{
                @as(f64, @floatFromInt(BackwardTiming.mul_time)) / 1_000_000.0,
                @as(f64, @floatFromInt(BackwardTiming.mul_time)) * 100.0 / @as(f64, @floatFromInt(total_time)),
            });
            std.debug.print("    TOTAL:         {d:.1}ms\n", .{
                @as(f64, @floatFromInt(total_time)) / 1_000_000.0,
            });
        }
    }

    /// Record an operation on the tape
    pub fn recordOp(
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
    /// Optional scratch buffers can be provided to avoid allocation overhead in backward pass
    pub fn trackedMatmul(
        self: *AutodiffContext,
        a: TrackedTensor,
        b: TrackedTensor,
        M: usize,
        K: usize,
        N: usize,
        c_tensor: Tensor,
        scratch_dA: ?[]Scalar,
        scratch_dB: ?[]Scalar,
    ) !TrackedTensor {
        // Forward pass
        try matmul_mod.matmul(c_tensor.data, a.tensor.data, b.tensor.data, M, K, N);

        // Track gradient
        const c = try self.track(c_tensor);

        // Record operation with optional scratch buffers
        const operands = [_]GradHandle{ a.grad_handle, b.grad_handle };
        try self.recordOp(.matmul, c.grad_handle, &operands, .{
            .matmul = .{
                .a_data = a.tensor.data,
                .b_data = b.tensor.data,
                .M = M,
                .K = K,
                .N = N,
                .scratch_dA = scratch_dA,
                .scratch_dB = scratch_dB,
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
fn backwardOp(param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext, param_count: usize, op: *const Op) void {
    switch (op.op_type) {
        .add => backwardAdd(param_grad_ctx, temp_grad_ctx, param_count, op),
        .mul => backwardMul(param_grad_ctx, temp_grad_ctx, param_count, op),
        .matmul => backwardMatmul(param_grad_ctx, temp_grad_ctx, param_count, op),
        .relu => backwardRelu(param_grad_ctx, temp_grad_ctx, param_count, op),
        .broadcast_add => backwardBroadcastAdd(param_grad_ctx, temp_grad_ctx, param_count, op),
        .gather_actions => backwardGatherActions(param_grad_ctx, temp_grad_ctx, param_count, op),
        .dueling_q => backwardDuelingQ(param_grad_ctx, temp_grad_ctx, param_count, op),
    }
}

/// Helper to get gradient from the correct context
fn getGrad(param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext, param_count: usize, handle: GradHandle) []Scalar {
    if (handle < param_count) {
        // Parameter gradient
        return param_grad_ctx.getGrad(handle);
    } else {
        // Temporary gradient (adjust offset)
        return temp_grad_ctx.getGrad(handle - param_count);
    }
}

/// Helper to accumulate gradient to the correct context
fn accumulateGrad(param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext, param_count: usize, handle: GradHandle, value: []const Scalar) void {
    if (handle < param_count) {
        // Parameter gradient
        param_grad_ctx.accumulate(handle, value);
    } else {
        // Temporary gradient (adjust offset)
        temp_grad_ctx.accumulate(handle - param_count, value);
    }
}

/// Backward pass for addition: da = dc, db = dc
fn backwardAdd(param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext, param_count: usize, op: *const Op) void {
    const dc = getGrad(param_grad_ctx, temp_grad_ctx, param_count, op.result_handle);
    const a_handle = op.operand_handles[0];
    const b_handle = op.operand_handles[1];

    accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, a_handle, dc);
    accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, b_handle, dc);
}

/// Backward pass for multiplication: da = dc * b, db = dc * a
fn backwardMul(param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext, param_count: usize, op: *const Op) void {
    const dc = getGrad(param_grad_ctx, temp_grad_ctx, param_count, op.result_handle);
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
    accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, a_handle, da_temp);

    // db = dc * a
    ops_mod.mulInto(db_temp, dc, cache.a_data) catch return;
    accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, b_handle, db_temp);
}

/// Backward pass for matmul: dA = dC @ B^T, dB = A^T @ dC
fn backwardMatmul(param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext, param_count: usize, op: *const Op) void {
    const dC = getGrad(param_grad_ctx, temp_grad_ctx, param_count, op.result_handle);
    const a_handle = op.operand_handles[0];
    const b_handle = op.operand_handles[1];

    const cache = op.cache.matmul;
    const M = cache.M;
    const K = cache.K;
    const N = cache.N;

    // Use pre-allocated scratch buffers if available, otherwise fall back to page allocator
    const allocator = std.heap.page_allocator;
    const use_scratch = cache.scratch_dA != null and cache.scratch_dB != null;

    var dA: []Scalar = undefined;
    var dB: []Scalar = undefined;
    var allocated_dA = false;
    var allocated_dB = false;

    if (use_scratch) {
        // Use pre-allocated scratch buffers (zero-cost, no allocator overhead)
        dA = cache.scratch_dA.?[0 .. M * K];
        dB = cache.scratch_dB.?[0 .. K * N];
    } else {
        // Fall back to page allocator for operations without scratch buffers
        dA = allocator.alloc(Scalar, M * K) catch return;
        allocated_dA = true;
        dB = allocator.alloc(Scalar, K * N) catch return;
        allocated_dB = true;
    }
    defer {
        if (allocated_dA) allocator.free(dA);
        if (allocated_dB) allocator.free(dB);
    }

    // dA = dC @ B^T (M x N @ N x K = M x K)
    matmul_mod.matmulTransposeB(dA, dC, cache.b_data, M, N, K) catch return;
    accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, a_handle, dA);

    // dB = A^T @ dC (K x M @ M x N = K x N)
    matmul_mod.matmulTransposeA(dB, cache.a_data, dC, M, K, N) catch return;
    accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, b_handle, dB);
}

/// Backward pass for ReLU: dx = dc * (x > 0)
fn backwardRelu(param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext, param_count: usize, op: *const Op) void {
    const dy = getGrad(param_grad_ctx, temp_grad_ctx, param_count, op.result_handle);
    const x_handle = op.operand_handles[0];
    const cache = op.cache.relu;

    const allocator = std.heap.page_allocator;
    const dx = allocator.alloc(Scalar, dy.len) catch return;
    defer allocator.free(dx);

    for (dy, 0..) |grad_val, i| {
        dx[i] = if (cache.input[i] > 0.0) grad_val else 0.0;
    }

    accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, x_handle, dx);
}

/// Backward pass for broadcast add
/// For [batch, features] + [features]:
/// - da = dc (gradient passes through)
/// - db = sum(dc over batch dimension)
/// SIMD-optimized reduction across batch dimension for bias gradients
/// Computes: db[j] = sum over i of dc[i * features + j]
/// This is the hot path for dense layer bias gradients in DQN training
inline fn reduceBatchToFeatures(db: []Scalar, dc: []const Scalar, batch: usize, features: usize) void {
    const Vec = config.Vec;
    const simd_lanes = config.TensorConfig.simd_lanes;

    // Zero-initialize output
    @memset(db, 0.0);

    // Process batch rows, SIMD-izing across features for better cache locality
    const vec_count = features / simd_lanes;
    const remainder = features % simd_lanes;

    for (0..batch) |i| {
        const row_offset = i * features;

        // SIMD fast path: accumulate features in vector chunks
        for (0..vec_count) |v| {
            const j_start = v * simd_lanes;
            const grad_vec: Vec = dc[row_offset + j_start ..][0..simd_lanes].*;
            var db_vec: Vec = db[j_start..][0..simd_lanes].*;
            db_vec += grad_vec;
            const db_array: [simd_lanes]Scalar = db_vec;
            @memcpy(db[j_start..][0..simd_lanes], &db_array);
        }

        // Scalar tail: remaining features
        if (remainder > 0) {
            const j_start = vec_count * simd_lanes;
            for (0..remainder) |idx| {
                const j = j_start + idx;
                db[j] += dc[row_offset + j];
            }
        }
    }
}

fn backwardBroadcastAdd(param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext, param_count: usize, op: *const Op) void {
    const dc = getGrad(param_grad_ctx, temp_grad_ctx, param_count, op.result_handle);
    const a_handle = op.operand_handles[0];
    const b_handle = op.operand_handles[1];

    const cache = op.cache.broadcast_add;
    const a_shape = cache.a_shape;
    const b_shape = cache.b_shape;

    // Gradient for 'a' passes through as-is
    accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, a_handle, dc);

    // Gradient for 'b' needs to be summed over broadcast dimensions
    // Handle [batch, features] + [features] case (common in DQN layers)
    if (a_shape.len == 2 and b_shape.len == 1) {
        const batch = a_shape[0];
        const features = a_shape[1];

        const allocator = std.heap.page_allocator;
        const db = allocator.alloc(Scalar, features) catch return;
        defer allocator.free(db);

        // Use SIMD-optimized reduction
        reduceBatchToFeatures(db, dc, batch, features);

        accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, b_handle, db);
        return;
    }

    // Handle [batch, features] + [1, features] case
    if (a_shape.len == 2 and b_shape.len == 2 and b_shape[0] == 1) {
        const batch = a_shape[0];
        const features = a_shape[1];

        const allocator = std.heap.page_allocator;
        const db = allocator.alloc(Scalar, features) catch return;
        defer allocator.free(db);

        // Use SIMD-optimized reduction
        reduceBatchToFeatures(db, dc, batch, features);

        accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, b_handle, db);
        return;
    }
}

/// Backward pass for gather_actions: scatter gradients back to source tensor
/// For q_sa[i] = q_all[i, actions[i]], we have:
/// dq_all[i, actions[i]] += dq_sa[i]
fn backwardGatherActions(param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext, param_count: usize, op: *const Op) void {
    const dq_sa = getGrad(param_grad_ctx, temp_grad_ctx, param_count, op.result_handle);
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

    accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, q_all_handle, dq_all);
}

fn backwardDuelingQ(param_grad_ctx: *GradContext, temp_grad_ctx: *GradContext, param_count: usize, op: *const Op) void {
    // Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
    // dL/dV = sum_a dL/dQ(s,a)  (sum across actions for each batch)
    // dL/dA(s,a) = dL/dQ(s,a) - (1/n_actions) * sum_a' dL/dQ(s,a')
    const dq = getGrad(param_grad_ctx, temp_grad_ctx, param_count, op.result_handle);
    const value_handle = op.operand_handles[0];
    const advantage_handle = op.operand_handles[1];

    const cache = op.cache.dueling_q;
    const batch_size = cache.batch_size;
    const num_actions = cache.num_actions;

    const allocator = std.heap.page_allocator;

    // Gradient for value: [batch_size, 1]
    // Each V(s) contributes to all Q(s,a), so sum gradients across actions
    const dv = allocator.alloc(Scalar, batch_size) catch return;
    defer allocator.free(dv);

    for (0..batch_size) |i| {
        var sum: Scalar = 0.0;
        for (0..num_actions) |a| {
            sum += dq[i * num_actions + a];
        }
        dv[i] = sum;
    }

    // Gradient for advantage: [batch_size, num_actions]
    // With mean-centering: dA(s,a) = dQ(s,a) - mean(dQ(s,:))
    const da = allocator.alloc(Scalar, batch_size * num_actions) catch return;
    defer allocator.free(da);

    const n_act_f = @as(Scalar, @floatFromInt(num_actions));
    for (0..batch_size) |i| {
        // Compute mean gradient across actions for this sample
        var mean_grad: Scalar = 0.0;
        for (0..num_actions) |a| {
            mean_grad += dq[i * num_actions + a];
        }
        mean_grad /= n_act_f;

        // Subtract mean from each action gradient
        for (0..num_actions) |a| {
            da[i * num_actions + a] = dq[i * num_actions + a] - mean_grad;
        }
    }

    accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, value_handle, dv);
    accumulateGrad(param_grad_ctx, temp_grad_ctx, param_count, advantage_handle, da);
}

test "autodiff add" {
    var param_grad_ctx = GradContext.init(std.testing.allocator);
    defer param_grad_ctx.deinit();

    var temp_grad_ctx = GradContext.init(std.testing.allocator);
    defer temp_grad_ctx.deinit();

    var ad_ctx = AutodiffContext.init(std.testing.allocator, &param_grad_ctx, &temp_grad_ctx);
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

    // Check gradients (tracked tensors are in temp_grad_ctx with offset)
    const da = ad_ctx.temp_grad_ctx.getGrad(a.grad_handle - ad_ctx.param_count);
    const db = ad_ctx.temp_grad_ctx.getGrad(b.grad_handle - ad_ctx.param_count);

    try std.testing.expectApproxEqAbs(@as(Scalar, 1.0), da[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 1.0), db[0], 1e-5);
}

test "autodiff dueling Q gradient propagation" {
    // Test that dueling Q operation (Q = V + (A - mean(A))) propagates gradients correctly
    // dL/dV = sum_a dL/dQ(s,a)
    // dL/dA(s,a) = dL/dQ(s,a) - (1/n_actions) * sum_a' dL/dQ(s,a')

    var param_grad_ctx = GradContext.init(std.testing.allocator);
    defer param_grad_ctx.deinit();

    var temp_grad_ctx = GradContext.init(std.testing.allocator);
    defer temp_grad_ctx.deinit();

    const batch_size = 2;
    const num_actions = 3;

    // Create value tensor shape: [batch_size, 1]
    const v_shape = TensorShape{
        .dims = &[_]usize{ batch_size, 1 },
        .strides = &[_]usize{ 1, 1 },
    };
    const v_handle = try temp_grad_ctx.allocGrad(v_shape);

    // Create advantage tensor shape: [batch_size, num_actions]
    const a_shape = TensorShape{
        .dims = &[_]usize{ batch_size, num_actions },
        .strides = &[_]usize{ num_actions, 1 },
    };
    const a_handle = try temp_grad_ctx.allocGrad(a_shape);

    // Create Q-value tensor shape: [batch_size, num_actions]
    const q_shape = TensorShape{
        .dims = &[_]usize{ batch_size, num_actions },
        .strides = &[_]usize{ num_actions, 1 },
    };
    const q_handle = try temp_grad_ctx.allocGrad(q_shape);

    // Seed gradient on Q (uniform gradient of 1.0 for simplicity)
    const dq = temp_grad_ctx.getGrad(q_handle);
    for (dq) |*grad| {
        grad.* = 1.0;
    }

    // Create the dueling Q operation
    var operand_handles = try std.testing.allocator.alloc(GradHandle, 2);
    defer std.testing.allocator.free(operand_handles);
    operand_handles[0] = v_handle;
    operand_handles[1] = a_handle;

    var op = Op{
        .op_type = .dueling_q,
        .result_handle = q_handle,
        .operand_handles = operand_handles,
        .cache = .{
            .dueling_q = .{
                .batch_size = batch_size,
                .num_actions = num_actions,
            },
        },
    };

    // Run backward pass
    backwardDuelingQ(&param_grad_ctx, &temp_grad_ctx, 0, &op);

    // Check gradients
    const dv = temp_grad_ctx.getGrad(v_handle);
    const da = temp_grad_ctx.getGrad(a_handle);

    // Expected gradients for value: sum of gradients across all actions
    // Batch 0: dL/dV = 1.0 + 1.0 + 1.0 = 3.0
    // Batch 1: dL/dV = 1.0 + 1.0 + 1.0 = 3.0
    try std.testing.expectApproxEqAbs(@as(Scalar, 3.0), dv[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 3.0), dv[1], 1e-5);

    // Expected gradients for advantage: dL/dA(s,a) = dL/dQ(s,a) - mean(dL/dQ(s,:))
    // Mean gradient = (1.0 + 1.0 + 1.0) / 3 = 1.0
    // Batch 0: dL/dA = [1.0 - 1.0, 1.0 - 1.0, 1.0 - 1.0] = [0.0, 0.0, 0.0]
    // Batch 1: dL/dA = [1.0 - 1.0, 1.0 - 1.0, 1.0 - 1.0] = [0.0, 0.0, 0.0]
    for (da) |grad| {
        try std.testing.expectApproxEqAbs(@as(Scalar, 0.0), grad, 1e-5);
    }

    // Now test with non-uniform gradients
    dq[0] = 2.0;  // Batch 0, action 0
    dq[1] = 1.0;  // Batch 0, action 1
    dq[2] = 0.5;  // Batch 0, action 2
    dq[3] = 1.5;  // Batch 1, action 0
    dq[4] = 1.0;  // Batch 1, action 1
    dq[5] = 0.5;  // Batch 1, action 2

    // Reset gradients
    @memset(dv, 0.0);
    @memset(da, 0.0);

    // Run backward again
    backwardDuelingQ(&param_grad_ctx, &temp_grad_ctx, 0, &op);

    // Expected gradients for value: sum across actions
    // Batch 0: dL/dV = 2.0 + 1.0 + 0.5 = 3.5
    // Batch 1: dL/dV = 1.5 + 1.0 + 0.5 = 3.0
    try std.testing.expectApproxEqAbs(@as(Scalar, 3.5), dv[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 3.0), dv[1], 1e-5);

    // Expected gradients for advantage
    // Batch 0: mean = (2.0 + 1.0 + 0.5) / 3 = 1.167
    //   dL/dA = [2.0 - 1.167, 1.0 - 1.167, 0.5 - 1.167] = [0.833, -0.167, -0.667]
    // Batch 1: mean = (1.5 + 1.0 + 0.5) / 3 = 1.0
    //   dL/dA = [1.5 - 1.0, 1.0 - 1.0, 0.5 - 1.0] = [0.5, 0.0, -0.5]
    try std.testing.expectApproxEqAbs(@as(Scalar, 0.833), da[0], 1e-2);
    try std.testing.expectApproxEqAbs(@as(Scalar, -0.167), da[1], 1e-2);
    try std.testing.expectApproxEqAbs(@as(Scalar, -0.667), da[2], 1e-2);
    try std.testing.expectApproxEqAbs(@as(Scalar, 0.5), da[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, 0.0), da[4], 1e-5);
    try std.testing.expectApproxEqAbs(@as(Scalar, -0.5), da[5], 1e-5);
}
