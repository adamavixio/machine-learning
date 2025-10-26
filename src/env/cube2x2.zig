const std = @import("std");

/// 2x2 Rubik's Cube (Pocket Cube) with 24 facelets (6 faces × 4 stickers)
/// Much smaller state space (~3.6 million) makes this tractable for DQN
pub const Cube2x2State = struct {
    facelets: [24]u8,

    /// Face indices for 2x2 cube
    /// Each face has 4 stickers numbered:
    ///   0 1
    ///   2 3
    pub const Face = enum(u8) {
        U = 0, // Up (white)
        D = 1, // Down (yellow)
        F = 2, // Front (green)
        B = 3, // Back (blue)
        L = 4, // Left (orange)
        R = 5, // Right (red)

        pub fn count() usize {
            return 6;
        }
    };

    /// Moves for 2x2 cube (only 6 moves vs 12 for 3x3)
    pub const Move = enum(u8) {
        R = 0,  // Right clockwise
        R_prime = 1, // Right counter-clockwise
        U = 2,  // Up clockwise
        U_prime = 3, // Up counter-clockwise
        F = 4,  // Front clockwise
        F_prime = 5, // Front counter-clockwise

        pub fn count() usize {
            return 6;
        }
    };

    /// Initialize a solved 2x2 cube
    pub fn initSolved() Cube2x2State {
        var state = Cube2x2State{ .facelets = undefined };

        // Each face has its color (4 stickers per face)
        for (0..6) |face| {
            for (0..4) |i| {
                state.facelets[face * 4 + i] = @intCast(face);
            }
        }

        return state;
    }

    /// Check if cube is solved
    pub fn isSolved(self: Cube2x2State) bool {
        // Check each face has uniform color
        for (0..6) |face| {
            const base_color = self.facelets[face * 4];
            for (1..4) |i| {
                if (self.facelets[face * 4 + i] != base_color) {
                    return false;
                }
            }
        }
        return true;
    }

    /// Clone the state
    pub fn clone(self: Cube2x2State) Cube2x2State {
        return Cube2x2State{ .facelets = self.facelets };
    }

    /// Hash for deduplication
    pub fn hash(self: Cube2x2State) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(std.mem.asBytes(&self.facelets));
        return h.final();
    }

    /// Execute a move using precomputed move tables
    pub fn step(self: *Cube2x2State, move: Move) void {
        const move_table = &MOVE_TABLES[@intFromEnum(move)];
        const old = self.facelets;
        for (0..24) |i| {
            self.facelets[i] = old[move_table[i]];
        }
    }

    /// Scramble the cube with random moves
    pub fn scramble(self: *Cube2x2State, num_moves: usize, rng: std.Random) void {
        for (0..num_moves) |_| {
            const move_idx = rng.intRangeAtMost(u8, 0, 5);
            const move: Move = @enumFromInt(move_idx);
            self.step(move);
        }
    }

    /// Convert state to one-hot encoding for neural network
    /// Output: 144-dim vector (24 facelets × 6 colors)
    pub fn toOneHot(self: Cube2x2State, output: []f32) void {
        std.debug.assert(output.len == 144);
        @memset(output, 0.0);

        for (self.facelets, 0..) |color, i| {
            const idx = i * 6 + color;
            output[idx] = 1.0;
        }
    }
};

/// 3D vector for facelet normals
const Vec3 = struct {
    x: i8,
    y: i8,
    z: i8,

    fn eql(self: Vec3, other: Vec3) bool {
        return self.x == other.x and self.y == other.y and self.z == other.z;
    }
};

/// Facelet normals (X→R/L, Y→F/B, Z→U/D)
const FACELET_NORMALS: [24]Vec3 = blk: {
    var normals: [24]Vec3 = undefined;
    // U (0-3): (0, 0, +1)
    for (0..4) |i| normals[i] = .{ .x = 0, .y = 0, .z = 1 };
    // D (4-7): (0, 0, -1)
    for (4..8) |i| normals[i] = .{ .x = 0, .y = 0, .z = -1 };
    // F (8-11): (0, +1, 0)
    for (8..12) |i| normals[i] = .{ .x = 0, .y = 1, .z = 0 };
    // B (12-15): (0, -1, 0)
    for (12..16) |i| normals[i] = .{ .x = 0, .y = -1, .z = 0 };
    // L (16-19): (-1, 0, 0)
    for (16..20) |i| normals[i] = .{ .x = -1, .y = 0, .z = 0 };
    // R (20-23): (+1, 0, 0)
    for (20..24) |i| normals[i] = .{ .x = 1, .y = 0, .z = 0 };
    break :blk normals;
};

/// Corner definitions
const Corner = struct {
    facelets: [3]u8,
};

const CORNERS = [8]Corner{
    .{ .facelets = .{ 1, 20, 9 } },   // 0: URF
    .{ .facelets = .{ 0, 17, 8 } },   // 1: UFL
    .{ .facelets = .{ 2, 16, 13 } },  // 2: ULB
    .{ .facelets = .{ 3, 21, 12 } },  // 3: UBR
    .{ .facelets = .{ 5, 22, 11 } },  // 4: DFR
    .{ .facelets = .{ 4, 19, 10 } },  // 5: DLF
    .{ .facelets = .{ 6, 18, 15 } },  // 6: DBL
    .{ .facelets = .{ 7, 23, 14 } },  // 7: DRB
};

/// Move specification
const MoveSpec = struct {
    cycle: [4]u8,
    rotate_fn: *const fn (Vec3) Vec3,
};

/// Rotation functions for each move
fn rotateR(v: Vec3) Vec3 {
    return .{ .x = v.x, .y = -v.z, .z = v.y };
}
fn rotateR_prime(v: Vec3) Vec3 {
    return .{ .x = v.x, .y = v.z, .z = -v.y };
}
fn rotateU(v: Vec3) Vec3 {
    return .{ .x = -v.y, .y = v.x, .z = v.z };
}
fn rotateU_prime(v: Vec3) Vec3 {
    return .{ .x = v.y, .y = -v.x, .z = v.z };
}
fn rotateF(v: Vec3) Vec3 {
    return .{ .x = v.z, .y = v.y, .z = -v.x };
}
fn rotateF_prime(v: Vec3) Vec3 {
    return .{ .x = -v.z, .y = v.y, .z = v.x };
}

const MOVE_SPECS = [6]MoveSpec{
    .{ .cycle = .{ 0, 3, 7, 4 }, .rotate_fn = &rotateR },        // R
    .{ .cycle = .{ 0, 4, 7, 3 }, .rotate_fn = &rotateR_prime },  // R'
    .{ .cycle = .{ 0, 1, 2, 3 }, .rotate_fn = &rotateU },        // U
    .{ .cycle = .{ 0, 3, 2, 1 }, .rotate_fn = &rotateU_prime },  // U'
    .{ .cycle = .{ 0, 4, 5, 1 }, .rotate_fn = &rotateF },        // F
    .{ .cycle = .{ 0, 1, 5, 4 }, .rotate_fn = &rotateF_prime },  // F'
};

/// Find which slot (0-2) in a corner has a given normal
fn indexOfNormal(corner: Corner, target: Vec3) ?usize {
    for (corner.facelets, 0..) |facelet_idx, slot| {
        const normal = FACELET_NORMALS[facelet_idx];
        if (normal.eql(target)) {
            return slot;
        }
    }
    return null;
}

/// Generate a move table using geometric rotation
fn generateMoveFromCorners(spec: MoveSpec) [24]u8 {
    var table: [24]u8 = undefined;

    // Start with identity
    for (0..24) |i| {
        table[i] = @intCast(i);
    }

    // Apply the corner cycle
    for (0..4) |i| {
        const src_corner_idx = spec.cycle[i];
        const dst_corner_idx = spec.cycle[(i + 1) % 4];

        const src_corner = CORNERS[src_corner_idx];
        const dst_corner = CORNERS[dst_corner_idx];

        // For each facelet in the source corner
        for (src_corner.facelets) |src_facelet_idx| {
            // Get its normal and rotate it
            const src_normal = FACELET_NORMALS[src_facelet_idx];
            const rotated_normal = spec.rotate_fn(src_normal);

            // Find which slot in the destination corner has this rotated normal
            if (indexOfNormal(dst_corner, rotated_normal)) |dest_slot| {
                const dest_facelet_idx = dst_corner.facelets[dest_slot];
                table[dest_facelet_idx] = src_facelet_idx;
            }
        }
    }

    return table;
}

/// Precomputed move tables for all 6 moves
/// Each table maps: new_position ← old_position
const MOVE_TABLES: [6][24]u8 = initMoveTables();

fn initMoveTables() [6][24]u8 {
    @setEvalBranchQuota(100000);

    var tables: [6][24]u8 = undefined;

    // All 6 moves with their rotation functions
    for (0..6) |i| {
        tables[i] = generateMoveFromCorners(MOVE_SPECS[i]);
    }

    return tables;
}

test "2x2 cube init solved" {
    const cube = Cube2x2State.initSolved();
    try std.testing.expect(cube.isSolved());

    // Check all facelets on each face match
    for (0..6) |face| {
        for (0..4) |i| {
            try std.testing.expectEqual(@as(u8, @intCast(face)), cube.facelets[face * 4 + i]);
        }
    }
}

test "2x2 cube moves" {
    var cube = Cube2x2State.initSolved();

    // R move then R' should return to solved
    cube.step(.R);
    try std.testing.expect(!cube.isSolved());
    cube.step(.R_prime);
    try std.testing.expect(cube.isSolved());

    // U move then U' should return to solved
    cube.step(.U);
    try std.testing.expect(!cube.isSolved());
    cube.step(.U_prime);
    try std.testing.expect(cube.isSolved());

    // F move then F' should return to solved
    cube.step(.F);
    try std.testing.expect(!cube.isSolved());
    cube.step(.F_prime);
    try std.testing.expect(cube.isSolved());
}

test "2x2 cube R move four times returns to solved" {
    var cube = Cube2x2State.initSolved();

    cube.step(.R);
    cube.step(.R);
    cube.step(.R);
    cube.step(.R);

    try std.testing.expect(cube.isSolved());
}

test "2x2 all moves + inverse return to solved" {
    // Test that each move followed by its inverse returns to solved
    const moves = [_]Cube2x2State.Move{ .R, .U, .F };
    const inverses = [_]Cube2x2State.Move{ .R_prime, .U_prime, .F_prime };

    for (moves, inverses) |move, inverse| {
        var cube = Cube2x2State.initSolved();
        const original = cube.facelets;

        cube.step(move);
        try std.testing.expect(!cube.isSolved());

        cube.step(inverse);
        try std.testing.expect(cube.isSolved());

        // Check all facelets match original
        for (cube.facelets, original) |facelet, orig| {
            try std.testing.expectEqual(orig, facelet);
        }
    }
}

test "2x2 cube scramble" {
    var prng = std.Random.DefaultPrng.init(42);
    const rng = prng.random();

    var cube = Cube2x2State.initSolved();
    cube.scramble(10, rng);

    // After scrambling, should not be solved (with very high probability)
    try std.testing.expect(!cube.isSolved());
}

test "2x2 cube one-hot encoding" {
    const cube = Cube2x2State.initSolved();
    var output: [144]f32 = undefined;

    cube.toOneHot(&output);

    // Check dimensions
    var sum: f32 = 0;
    for (output) |val| {
        sum += val;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 24.0), sum, 1e-5); // 24 facelets, each has one hot bit

    // Check first facelet (should be color 0)
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), output[0], 1e-5);
    for (1..6) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[i], 1e-5);
    }
}

test "2x2 cube hash" {
    const cube1 = Cube2x2State.initSolved();
    var cube2 = Cube2x2State.initSolved();

    try std.testing.expectEqual(cube1.hash(), cube2.hash());

    cube2.step(.R);
    try std.testing.expect(cube1.hash() != cube2.hash());
}

test "2x2 cube single moves debug" {
    // Test R then R'
    var cube = Cube2x2State.initSolved();
    cube.step(.R);
    const after_R_solved = cube.isSolved();
    cube.step(.R_prime);
    const after_R_prime_solved = cube.isSolved();

    if (!after_R_prime_solved) {
        std.debug.print("\n[DEBUG] R then R' FAILED\n", .{});
        std.debug.print("After R: solved={}\n", .{after_R_solved});
        std.debug.print("After R': solved={}\n", .{after_R_prime_solved});
        std.debug.print("Facelets: ", .{});
        for (cube.facelets) |f| {
            std.debug.print("{d} ", .{f});
        }
        std.debug.print("\n", .{});
    }

    try std.testing.expect(after_R_prime_solved);
}

test "2x2 cube any 1-move scramble is solvable" {
    // Test that every 1-move scramble has a solution (its inverse)
    const moves = [_]Cube2x2State.Move{ .R, .U, .F };
    const inverses = [_]Cube2x2State.Move{ .R_prime, .U_prime, .F_prime };

    for (moves, inverses) |move, inverse| {
        var cube = Cube2x2State.initSolved();
        cube.step(move);

        try std.testing.expect(!cube.isSolved());

        // The inverse should solve it
        cube.step(inverse);
        try std.testing.expect(cube.isSolved());
    }
}

test "2x2 cube state space exploration (BFS)" {
    const allocator = std.testing.allocator;

    // Track seen states using hash
    var seen = std.AutoHashMap(u64, void).init(allocator);
    defer seen.deinit();

    // Queue for BFS
    var queue = std.ArrayList(Cube2x2State){};
    defer queue.deinit(allocator);

    // Start with solved state
    const start = Cube2x2State.initSolved();
    try seen.put(start.hash(), {});
    try queue.append(allocator, start);

    const all_moves = [_]Cube2x2State.Move{ .R, .R_prime, .U, .U_prime, .F, .F_prime };

    var depth: usize = 0;
    var states_at_depth: usize = 1;
    var next_depth_states: usize = 0;

    std.debug.print("\n=== 2x2 Cube State Space Exploration (BFS) ===\n", .{});
    std.debug.print("Depth 0: 1 state (solved)\n", .{});

    var queue_idx: usize = 0;
    while (queue_idx < queue.items.len) : (queue_idx += 1) {
        const current = queue.items[queue_idx];

        // Try all 6 moves
        for (all_moves) |move| {
            var next_state = current;
            next_state.step(move);

            const h = next_state.hash();
            if (!seen.contains(h)) {
                try seen.put(h, {});
                try queue.append(allocator, next_state);
                next_depth_states += 1;
            }
        }

        states_at_depth -= 1;
        if (states_at_depth == 0 and next_depth_states > 0) {
            depth += 1;
            states_at_depth = next_depth_states;
            next_depth_states = 0;
            std.debug.print("Depth {d}: {d} new states (total: {d})\n", .{ depth, states_at_depth, seen.count() });
        }
    }

    std.debug.print("\nTotal unique states discovered: {d}\n", .{seen.count()});
    std.debug.print("Expected for 2x2: 3,674,160 (full state space)\n\n", .{});

    // Verify we found the correct number of states for a 2x2 cube
    try std.testing.expect(seen.count() == 3_674_160);
}
