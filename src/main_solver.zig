/// CLI tool for BFS solver
///
/// Usage:
///   zig build solver -- generate [output.bfs]    # Generate BFS table
///   zig build solver -- load [input.bfs]         # Load and query BFS table
///   zig build solver -- test-scramble [depth]    # Test solver on scrambled cubes

const std = @import("std");
const root = @import("root.zig");
const Cube2x2State = root.env.Cube2x2State;
const Move2x2 = root.env.Move2x2;
const BFSSolver = root.solver.bfs.BFSSolver;
const SolverTable = root.solver.bfs.SolverTable;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        try printUsage();
        return;
    }

    const command = args[1];

    if (std.mem.eql(u8, command, "generate")) {
        const output_path = if (args.len >= 3) args[2] else "solver_2x2.bfs";
        try generateAndSave(allocator, output_path);
    } else if (std.mem.eql(u8, command, "load")) {
        const input_path = if (args.len >= 3) args[2] else "solver_2x2.bfs";
        try loadAndQuery(allocator, input_path);
    } else if (std.mem.eql(u8, command, "test-scramble")) {
        const depth = if (args.len >= 3) try std.fmt.parseInt(usize, args[2], 10) else 7;
        try testScramble(allocator, depth);
    } else {
        std.debug.print("Unknown command: {s}\n", .{command});
        try printUsage();
    }
}

fn printUsage() !void {
    std.debug.print(
        \\BFS Solver for 2x2 Rubik's Cube
        \\
        \\Usage:
        \\  solver generate [output.bfs]    # Generate and save BFS table
        \\  solver load [input.bfs]         # Load and query BFS table
        \\  solver test-scramble [depth]    # Test solver on scrambled cubes
        \\
        \\Examples:
        \\  solver generate table.bfs
        \\  solver load table.bfs
        \\  solver test-scramble 7
        \\
    , .{});
}

fn generateAndSave(allocator: std.mem.Allocator, output_path: []const u8) !void {
    std.debug.print("=== Generating BFS Table ===\n", .{});

    var solver = try BFSSolver.init(allocator);
    defer solver.deinit();

    try solver.solve();

    std.debug.print("\nSaving to: {s}\n", .{output_path});
    try solver.table.saveToBinary(output_path);

    const file_size = blk: {
        const file = try std.fs.cwd().openFile(output_path, .{});
        defer file.close();
        const stat = try file.stat();
        break :blk stat.size;
    };

    std.debug.print("File size: {:.2} MB\n", .{@as(f64, @floatFromInt(file_size)) / 1024.0 / 1024.0});
    std.debug.print("✓ Done!\n", .{});
}

fn loadAndQuery(allocator: std.mem.Allocator, input_path: []const u8) !void {
    std.debug.print("=== Loading BFS Table ===\n", .{});
    std.debug.print("Loading from: {s}\n", .{input_path});

    const start_time = std.time.milliTimestamp();
    var table = try SolverTable.loadFromBinary(allocator, input_path);
    defer table.deinit();
    const load_time = std.time.milliTimestamp() - start_time;

    std.debug.print("Loaded {} states in {:.2}s\n", .{
        table.total_states,
        @as(f64, @floatFromInt(load_time)) / 1000.0,
    });
    std.debug.print("Max depth: {}\n", .{table.max_depth});

    // Test a few scrambled states
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    const rng = prng.random();

    std.debug.print("\n=== Testing Solver ===\n", .{});
    for (0..5) |i| {
        var cube = Cube2x2State.initSolved();
        cube.scramble(7, rng);

        const optimal_depth = table.getOptimalDepth(&cube);
        const optimal_move = table.getOptimalMove(&cube);

        std.debug.print("Test {}: depth={?}, move={?}\n", .{ i + 1, optimal_depth, optimal_move });
    }
}

fn testScramble(allocator: std.mem.Allocator, scramble_depth: usize) !void {
    std.debug.print("=== Generating BFS Table ===\n", .{});

    var solver = try BFSSolver.init(allocator);
    defer solver.deinit();

    try solver.solve();

    // Test solving scrambled cubes
    var prng = std.Random.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    const rng = prng.random();

    std.debug.print("\n=== Testing Scramble Depth {} ===\n", .{scramble_depth});

    var successes: usize = 0;
    const num_tests = 100;

    for (0..num_tests) |test_idx| {
        var cube = Cube2x2State.initSolved();
        cube.scramble(scramble_depth, rng);

        // Try to solve with greedy policy (max 50 steps)
        var steps: usize = 0;
        const max_steps = 50;

        while (!cube.isSolved() and steps < max_steps) : (steps += 1) {
            const optimal_move = solver.table.getOptimalMove(&cube);
            if (optimal_move == null) {
                std.debug.print("Test {}: state not in table!\n", .{test_idx + 1});
                break;
            }
            const move: Move2x2 = @enumFromInt(optimal_move.?);
            cube.step(move);
        }

        if (cube.isSolved()) {
            successes += 1;
        } else {
            std.debug.print("Test {}: failed to solve in {} steps\n", .{ test_idx + 1, max_steps });
        }
    }

    const success_rate = @as(f64, @floatFromInt(successes)) / @as(f64, @floatFromInt(num_tests)) * 100.0;
    std.debug.print("\n=== Results ===\n", .{});
    std.debug.print("Scramble depth: {}\n", .{scramble_depth});
    std.debug.print("Tests: {}\n", .{num_tests});
    std.debug.print("Successes: {}\n", .{successes});
    std.debug.print("Success rate: {:.1}%\n", .{success_rate});
}
