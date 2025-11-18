/// BFS-based optimal solver for 2x2 Rubik's Cube
///
/// Explores the full state space (~3.7M states) from the solved position
/// to build a complete policy and value table for supervised learning.
///
/// Key features:
/// - CompactState: 24-byte representation matching Cube2x2State.facelets
/// - Full BFS without canonical normalization (simpler, tractable)
/// - Binary save/load for reusing the computed table
/// - SampleGenerator for supervised training data

const std = @import("std");
const root = @import("../root.zig");
const Cube2x2State = root.env.Cube2x2State;
const Move2x2 = root.env.Move2x2;

/// Compact state representation for hash table (24 bytes)
/// Directly uses the 24 facelets from Cube2x2State
pub const CompactState = struct {
    facelets: [24]u8,

    pub fn fromCube2x2State(state: *const Cube2x2State) CompactState {
        return .{ .facelets = state.facelets };
    }

    pub fn toCube2x2State(self: *const CompactState) Cube2x2State {
        return .{ .facelets = self.facelets };
    }

    pub fn hash(self: *const CompactState) u64 {
        var h = std.hash.Wyhash.init(0);
        h.update(std.mem.asBytes(&self.facelets));
        return h.final();
    }

    pub fn eql(self: *const CompactState, other: *const CompactState) bool {
        return std.mem.eql(u8, &self.facelets, &other.facelets);
    }
};

/// Context for CompactState hash map
pub const CompactStateContext = struct {
    pub fn hash(_: CompactStateContext, key: CompactState) u64 {
        return key.hash();
    }

    pub fn eql(_: CompactStateContext, a: CompactState, b: CompactState) bool {
        return a.eql(&b);
    }
};

/// Entry in the solver table
pub const SolverEntry = struct {
    depth: u8,              // Distance from solved state (0-11)
    optimal_move: u8,       // Best move to get closer to solved (0-5, or 255 if already solved)
};

/// BFS result: stores optimal policy and value for all reachable states
pub const SolverTable = struct {
    // Hash table: state → (depth, optimal_move)
    states: std.HashMap(CompactState, SolverEntry, CompactStateContext, std.hash_map.default_max_load_percentage),
    max_depth: u8,
    total_states: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !SolverTable {
        return SolverTable{
            .states = std.HashMap(CompactState, SolverEntry, CompactStateContext, std.hash_map.default_max_load_percentage).init(allocator),
            .max_depth = 0,
            .total_states = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *SolverTable) void {
        self.states.deinit();
    }

    /// Get optimal action for a state (returns null if state not in table)
    pub fn getOptimalMove(self: *SolverTable, state: *const Cube2x2State) ?u8 {
        const compact = CompactState.fromCube2x2State(state);
        if (self.states.get(compact)) |entry| {
            return entry.optimal_move;
        }
        return null;
    }

    /// Get optimal depth (value) for a state (returns null if state not in table)
    pub fn getOptimalDepth(self: *SolverTable, state: *const Cube2x2State) ?u8 {
        const compact = CompactState.fromCube2x2State(state);
        if (self.states.get(compact)) |entry| {
            return entry.depth;
        }
        return null;
    }

    /// Save table to binary file
    /// Format: [header][entries]
    /// Header: magic(4) version(4) max_depth(1) total_states(8)
    /// Entry: facelets(24) depth(1) optimal_move(1)
    pub fn saveToBinary(self: *SolverTable, path: []const u8) !void {
        const file = try std.fs.cwd().createFile(path, .{});
        defer file.close();

        // Build entire buffer in memory, then write once
        const entry_size = 24 + 1 + 1; // facelets + depth + optimal_move
        const header_size = 4 + 4 + 1 + 8; // magic + version + max_depth + total_states
        const total_size = header_size + self.total_states * entry_size;

        const buffer = try self.allocator.alloc(u8, total_size);
        defer self.allocator.free(buffer);

        var offset: usize = 0;

        // Write header
        @memcpy(buffer[offset .. offset + 4], "BFS2");
        offset += 4;
        std.mem.writeInt(u32, buffer[offset..][0..4], 1, .little); // Version
        offset += 4;
        buffer[offset] = self.max_depth;
        offset += 1;
        std.mem.writeInt(u64, buffer[offset..][0..8], self.total_states, .little);
        offset += 8;

        // Write entries
        var iter = self.states.iterator();
        while (iter.next()) |entry| {
            @memcpy(buffer[offset .. offset + 24], &entry.key_ptr.facelets);
            offset += 24;
            buffer[offset] = entry.value_ptr.depth;
            offset += 1;
            buffer[offset] = entry.value_ptr.optimal_move;
            offset += 1;
        }

        try file.writeAll(buffer);
    }

    /// Load table from binary file
    pub fn loadFromBinary(allocator: std.mem.Allocator, path: []const u8) !SolverTable {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();

        // Read entire file into memory
        const file_size = (try file.stat()).size;
        const buffer = try allocator.alloc(u8, file_size);
        defer allocator.free(buffer);
        _ = try file.readAll(buffer);

        var offset: usize = 0;

        // Read header
        if (offset + 4 > buffer.len or !std.mem.eql(u8, buffer[offset .. offset + 4], "BFS2")) {
            return error.InvalidMagic;
        }
        offset += 4;

        const version = std.mem.readInt(u32, buffer[offset..][0..4], .little);
        offset += 4;
        if (version != 1) {
            return error.UnsupportedVersion;
        }

        const max_depth = buffer[offset];
        offset += 1;

        const total_states = std.mem.readInt(u64, buffer[offset..][0..8], .little);
        offset += 8;

        // Initialize table
        var table = try SolverTable.init(allocator);
        errdefer table.deinit();

        table.max_depth = max_depth;
        table.total_states = total_states;

        // Preallocate hash map for efficiency
        try table.states.ensureTotalCapacity(@intCast(total_states));

        // Read entries
        for (0..total_states) |_| {
            var state: CompactState = undefined;
            @memcpy(&state.facelets, buffer[offset .. offset + 24]);
            offset += 24;

            const depth = buffer[offset];
            offset += 1;

            const optimal_move = buffer[offset];
            offset += 1;

            try table.states.put(state, .{
                .depth = depth,
                .optimal_move = optimal_move,
            });
        }

        return table;
    }
};

const QueueEntry = struct {
    state: Cube2x2State,
    depth: u8,
};

/// BFS solver: explores full state space from solved state
pub const BFSSolver = struct {
    allocator: std.mem.Allocator,
    table: SolverTable,

    pub fn init(allocator: std.mem.Allocator) !BFSSolver {
        return BFSSolver{
            .allocator = allocator,
            .table = try SolverTable.init(allocator),
        };
    }

    pub fn deinit(self: *BFSSolver) void {
        self.table.deinit();
    }

    /// Run BFS from solved state, populate table
    /// Returns the number of states discovered
    pub fn solve(self: *BFSSolver) !void {
        const start_time = std.time.milliTimestamp();

        var queue = std.ArrayList(QueueEntry){};
        defer queue.deinit(self.allocator);

        // Start from solved state
        const solved = Cube2x2State.initSolved();
        const solved_compact = CompactState.fromCube2x2State(&solved);

        try self.table.states.put(solved_compact, .{
            .depth = 0,
            .optimal_move = 255, // No move needed (already solved)
        });

        try queue.append(self.allocator, .{
            .state = solved,
            .depth = 0,
        });

        var states_processed: usize = 0;
        var last_report_time = start_time;

        std.debug.print("Starting BFS from solved state...\n", .{});

        // BFS expansion - iterate by index to avoid O(n²) cost
        var queue_idx: usize = 0;
        while (queue_idx < queue.items.len) : (queue_idx += 1) {
            const current = queue.items[queue_idx];
            states_processed += 1;

            // Progress reporting every 5 seconds
            const now = std.time.milliTimestamp();
            if (now - last_report_time > 5000) {
                const elapsed_sec = @as(f64, @floatFromInt(now - start_time)) / 1000.0;
                std.debug.print("Depth {}: {} states explored, {} in queue ({:.1}s)\n", .{
                    current.depth,
                    self.table.states.count(),
                    queue.items.len,
                    elapsed_sec,
                });
                last_report_time = now;
            }

            // Try all 6 moves
            for (0..6) |move_idx| {
                var next_state = current.state.clone();
                const move: Move2x2 = @enumFromInt(move_idx);
                next_state.step(move);

                const next_compact = CompactState.fromCube2x2State(&next_state);

                // If not visited, add to table and queue
                if (!self.table.states.contains(next_compact)) {
                    const next_depth = current.depth + 1;

                    // Store the INVERSE move - the move that takes us back toward solved
                    // For 2x2 cube: R/R' are inverses (0/1), U/U' (2/3), F/F' (4/5)
                    // Inverse is just XOR with 1
                    const inverse_move: u8 = @intCast(move_idx ^ 1);

                    try self.table.states.put(next_compact, .{
                        .depth = next_depth,
                        .optimal_move = inverse_move,
                    });

                    // Update max depth
                    if (next_depth > self.table.max_depth) {
                        self.table.max_depth = next_depth;
                    }

                    // Continue BFS (no max depth limit - explore full space)
                    try queue.append(self.allocator, .{
                        .state = next_state,
                        .depth = next_depth,
                    });
                }
            }
        }

        self.table.total_states = self.table.states.count();

        const end_time = std.time.milliTimestamp();
        const elapsed_sec = @as(f64, @floatFromInt(end_time - start_time)) / 1000.0;

        std.debug.print("\nBFS complete!\n", .{});
        std.debug.print("Total states: {}\n", .{self.table.total_states});
        std.debug.print("Max depth: {}\n", .{self.table.max_depth});
        std.debug.print("Time: {:.2}s\n", .{elapsed_sec});
        std.debug.print("States/sec: {:.0}\n", .{@as(f64, @floatFromInt(self.table.total_states)) / elapsed_sec});
    }
};

/// Sample generator: produces training data from solved table
pub const SampleGenerator = struct {
    solver: *BFSSolver,
    rng: std.Random,

    /// Generate a random training sample by picking from the table
    pub fn generateSample(self: *SampleGenerator) !TrainingSample {
        // Pick a random state from the table
        const table_size = self.solver.table.states.count();
        const random_idx = self.rng.intRangeLessThan(usize, 0, table_size);

        var iter = self.solver.table.states.iterator();
        var idx: usize = 0;
        while (iter.next()) |entry| {
            if (idx == random_idx) {
                return TrainingSample{
                    .state = entry.key_ptr.toCube2x2State(),
                    .optimal_action = entry.value_ptr.optimal_move,
                    .optimal_depth = @as(f32, @floatFromInt(entry.value_ptr.depth)),
                };
            }
            idx += 1;
        }

        return error.SampleGenerationFailed;
    }

    /// Generate a batch of samples
    pub fn generateBatch(
        self: *SampleGenerator,
        batch_size: usize,
        states: []f32,        // [batch_size × 144] one-hot states
        actions: []u8,        // [batch_size] optimal actions
        depths: []f32,        // [batch_size] optimal depths
    ) !void {
        std.debug.assert(states.len == batch_size * 144);
        std.debug.assert(actions.len == batch_size);
        std.debug.assert(depths.len == batch_size);

        for (0..batch_size) |i| {
            const sample = try self.generateSample();

            // Convert state to one-hot
            const state_slice = states[i * 144 .. (i + 1) * 144];
            sample.state.toOneHot(state_slice);

            actions[i] = sample.optimal_action;
            depths[i] = sample.optimal_depth;
        }
    }
};

pub const TrainingSample = struct {
    state: Cube2x2State,
    optimal_action: u8,
    optimal_depth: f32, // Distance to solved
};

// Tests
test "CompactState conversion" {
    const testing = std.testing;

    const cube = Cube2x2State.initSolved();
    const compact = CompactState.fromCube2x2State(&cube);
    const cube2 = compact.toCube2x2State();

    try testing.expect(cube2.isSolved());
}

test "CompactState hash and equality" {
    const testing = std.testing;

    const cube1 = Cube2x2State.initSolved();
    const compact1 = CompactState.fromCube2x2State(&cube1);
    const compact2 = CompactState.fromCube2x2State(&cube1);

    try testing.expect(compact1.eql(&compact2));
    try testing.expectEqual(compact1.hash(), compact2.hash());
}

test "BFS solver basic" {
    const testing = std.testing;

    var solver = try BFSSolver.init(testing.allocator);
    defer solver.deinit();

    try solver.solve();

    // Check that solved state has depth 0
    const solved = Cube2x2State.initSolved();
    const depth = solver.table.getOptimalDepth(&solved);
    try testing.expectEqual(@as(u8, 0), depth.?);

    // Check that we found a reasonable number of states
    try testing.expect(solver.table.total_states > 1000000);
    try testing.expect(solver.table.total_states < 5000000);

    // Check max depth is reasonable (should be 11 for 2x2 cube)
    try testing.expect(solver.table.max_depth >= 10);
    try testing.expect(solver.table.max_depth <= 14);
}
