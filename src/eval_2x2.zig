const std = @import("std");
const Tensor = @import("tensor.zig").Tensor;
const NN = @import("nn.zig").NN;
const Layer = @import("nn.zig").Layer;
const Cube2x2 = @import("env/cube2x2.zig").Cube2x2;
const DQN = @import("rl/dqn.zig").DQN;

const Scramble = struct {
    depth: usize,
    index: usize,
    moves: []const u8,
};

fn loadScrambles(allocator: std.mem.Allocator, path: []const u8) ![]Scramble {
    // Read entire file into memory
    const file_content = try std.fs.cwd().readFileAlloc(allocator, path, 1024 * 1024);
    defer allocator.free(file_content);

    // First pass: count scrambles
    var count: usize = 0;
    var lines = std.mem.splitScalar(u8, file_content, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        count += 1;
    }

    // Allocate scrambles array
    const scrambles = try allocator.alloc(Scramble, count);
    errdefer allocator.free(scrambles);

    // Second pass: parse scrambles
    lines = std.mem.splitScalar(u8, file_content, '\n');
    var idx: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;

        var parts = std.mem.splitScalar(u8, line, ',');

        const depth_str = parts.next() orelse continue;
        const depth = try std.fmt.parseInt(usize, depth_str, 10);

        const index_str = parts.next() orelse continue;
        const index = try std.fmt.parseInt(usize, index_str, 10);

        // Count moves
        var move_count: usize = 0;
        var parts_copy = std.mem.splitScalar(u8, line, ',');
        _ = parts_copy.next(); // skip depth
        _ = parts_copy.next(); // skip index
        while (parts_copy.next()) |_| {
            move_count += 1;
        }

        // Allocate and parse moves
        const moves = try allocator.alloc(u8, move_count);
        var move_idx: usize = 0;
        while (parts.next()) |move_str| {
            const move = try std.fmt.parseInt(u8, move_str, 10);
            moves[move_idx] = move;
            move_idx += 1;
        }

        scrambles[idx] = .{
            .depth = depth,
            .index = index,
            .moves = moves,
        };
        idx += 1;
    }

    return scrambles;
}

fn freeScrambles(allocator: std.mem.Allocator, scrambles: []Scramble) void {
    for (scrambles) |s| {
        allocator.free(s.moves);
    }
    allocator.free(scrambles);
}

fn evaluateGreedy(
    allocator: std.mem.Allocator,
    dqn: *DQN,
    scrambles: []const Scramble,
    depth: usize,
    max_steps: usize,
    verbose: bool,
) !struct { success_rate: f32, successes: usize, total: usize } {
    var successes: usize = 0;
    var count: usize = 0;

    if (verbose) {
        std.debug.print("\nEvaluating depth-{d} scrambles (epsilon=0, greedy only)\n", .{depth});
        std.debug.print("============================================================\n", .{});
    }

    for (scrambles) |scramble| {
        if (scramble.depth != depth) continue;
        count += 1;

        // Reset environment
        var env = Cube2x2.init();

        // Apply scramble
        for (scramble.moves) |move| {
            _ = env.step(move);
        }

        // Get initial state
        var state = try env.getState(allocator);
        defer state.deinit();

        // Run greedy episode (epsilon=0)
        var solved = false;
        var steps: usize = 0;

        while (steps < max_steps) : (steps += 1) {
            // Pure greedy action selection (no exploration)
            const action = try dqn.selectAction(allocator, &state, 0.0); // epsilon=0

            // Take action
            const result = env.step(@intCast(action));

            // Update state
            state.deinit();
            state = try env.getState(allocator);

            if (result.done) {
                solved = true;
                break;
            }
        }

        if (solved) {
            successes += 1;
            if (verbose) {
                std.debug.print("Scramble {d}/{d}: SOLVED in {d} steps\n", .{
                    scramble.index + 1,
                    count,
                    steps + 1,
                });
            }
        } else {
            if (verbose) {
                std.debug.print("Scramble {d}/{d}: FAILED (max steps reached)\n", .{
                    scramble.index + 1,
                    count,
                });
            }
        }
    }

    const success_rate = if (count > 0)
        @as(f32, @floatFromInt(successes)) / @as(f32, @floatFromInt(count)) * 100.0
    else
        0.0;

    if (verbose) {
        std.debug.print("============================================================\n", .{});
        std.debug.print("Success Rate: {d:.1}% ({d}/{d})\n", .{
            success_rate,
            successes,
            count,
        });
    }

    return .{
        .success_rate = success_rate,
        .successes = successes,
        .total = count,
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Deterministic DQN Evaluation Harness (Zig)\n", .{});
    std.debug.print("============================================================\n", .{});

    // Load scrambles from file
    const scrambles_path = "python/eval_scrambles.txt";
    std.debug.print("\nLoading scrambles from {s}...\n", .{scrambles_path});

    const scrambles = loadScrambles(allocator, scrambles_path) catch |err| {
        std.debug.print("Error loading scrambles: {}\n", .{err});
        std.debug.print("Make sure you've run python/eval.py first to generate eval_scrambles.txt\n", .{});
        return err;
    };
    defer freeScrambles(allocator, scrambles);

    std.debug.print("Loaded {d} scrambles\n", .{scrambles.len});

    // Count scrambles per depth
    var depth1: usize = 0;
    var depth2: usize = 0;
    var depth3: usize = 0;
    for (scrambles) |s| {
        switch (s.depth) {
            1 => depth1 += 1,
            2 => depth2 += 1,
            3 => depth3 += 1,
            else => {},
        }
    }
    std.debug.print("  Depth-1: {d} scrambles\n", .{depth1});
    std.debug.print("  Depth-2: {d} scrambles\n", .{depth2});
    std.debug.print("  Depth-3: {d} scrambles\n", .{depth3});

    // Create DQN agent (same architecture as training)
    const state_dim = 144; // 24 positions × 6 colors
    const action_dim = 6; // U, U', R, R', F, F'

    var dqn = try DQN.init(
        allocator,
        state_dim,
        action_dim,
        &[_]usize{ 128, 64 }, // hidden layers
        0.005, // learning rate
        0.99, // gamma
        0.01, // tau (Polyak averaging)
    );
    defer dqn.deinit();

    // TODO: Load trained model weights
    // For now, we'll evaluate the untrained network to match the PyTorch setup
    std.debug.print("\nWARNING: Using untrained network (model loading not yet implemented)\n", .{});
    std.debug.print("This will show 0% success, matching PyTorch's current greedy eval.\n", .{});

    std.debug.print("\n============================================================\n", .{});
    std.debug.print("GREEDY EVALUATION (epsilon=0)\n", .{});
    std.debug.print("============================================================\n", .{});

    // Evaluate each depth
    const depths = [_]usize{ 1, 2, 3 };
    for (depths) |d| {
        std.debug.print("\n--- Depth {d} ---\n", .{d});
        const result = try evaluateGreedy(
            allocator,
            &dqn,
            scrambles,
            d,
            100, // max steps
            true, // verbose
        );
        _ = result;
    }
}
