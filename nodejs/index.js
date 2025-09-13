#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const os = require('os');
const { Worker } = require('worker_threads');

// Type definitions (as comments for JS)
// Grid = string[][]
// GridDimensions = [rows: number, columns: number]
// BoardState = boolean[][]
// Position = [row: number, column: number]
// Moove = [Position, Position, Position]
// MooveCandidate = [Moove, coverage_gain: number]
// MooveSequence = Moove[]
// Direction = number (0-7)
// MooveDirection = [dr: number, dc: number]
// MooCount = number
// MooveCountSequence = number[]
// MooveCoverageGainSequence = number[]
// SimulationResult = [BoardState, MooCount, MooveSequence, MooveCountSequence, MooveCoverageGainSequence]

const PROJECT_ROOT = path.join(__dirname, '..');
const OUTPUT_DIR = path.join(PROJECT_ROOT, 'output');
const OUTPUT_PUZZLES_DIR = path.join(PROJECT_ROOT, 'puzzles');

// Ensure output directories exist
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

function gridFromFile(filePath) {
    const content = fs.readFileSync(filePath, 'utf8');
    const lines = content.split('\n').filter(line => line.trim());
    return lines.map(line => line.split(''));
}

function gridDimensions(grid) {
    const rows = grid.length;
    const cols = rows > 0 ? grid[0].length : 0;
    return [rows, cols];
}

function isValidMoove(m, grid) {
    const [height, width] = gridDimensions(grid);
    const [t1, t2, t3] = m;
    const [r1, c1] = t1;
    const [r2, c2] = t2;
    const [r3, c3] = t3;

    // Check bounds
    if (r1 < 0 || r1 >= height || c1 < 0 || c1 >= width) return false;
    if (r2 < 0 || r2 >= height || c2 < 0 || c2 >= width) return false;
    if (r3 < 0 || r3 >= height || c3 < 0 || c3 >= width) return false;

    // Calculate directions
    const d1 = [r2 - r1, c2 - c1];
    const d2 = [r3 - r2, c3 - c2];

    // Check if it spells 'moo'
    if (grid[r1][c1] !== 'm' || grid[r2][c2] !== 'o' || grid[r3][c3] !== 'o') {
        return false;
    }

    // Check if t2 is adjacent to t1
    if (Math.abs(d1[0]) > 1 || Math.abs(d1[1]) > 1) {
        return false;
    }

    // Check if t3 follows the same direction from t2
    return d1[0] === d2[0] && d1[1] === d2[1];
}

function generateMoove(start, direction) {
    const directions = [
        [-1, 0],  // up
        [-1, 1],  // up-right
        [0, 1],   // right
        [1, 1],   // down-right
        [1, 0],   // down
        [1, -1],  // down-left
        [0, -1],  // left
        [-1, -1]  // up-left
    ];
    const d = directions[direction];

    const t1 = start;
    const t2 = [start[0] + d[0], start[1] + d[1]];
    const t3 = [start[0] + 2 * d[0], start[1] + 2 * d[1]];

    return [t1, t2, t3];
}

function generateAllValidMooves(grid) {
    const [height, width] = gridDimensions(grid);
    const mooves = [];

    for (let r = 0; r < height; r++) {
        for (let c = 0; c < width; c++) {
            for (let direction = 0; direction < 8; direction++) {
                const moove = generateMoove([r, c], direction);
                if (isValidMoove(moove, grid)) {
                    mooves.push(moove);
                }
            }
        }
    }
    return mooves;
}

function generateOverlapsGraph(mooves) {
    const overlaps = {};

    for (let i = 0; i < mooves.length; i++) {
        for (let j = i + 1; j < mooves.length; j++) {
            if (doMoovesOverlap(mooves[i], mooves[j])) {
                const key1 = JSON.stringify(mooves[i]);
                const key2 = JSON.stringify(mooves[j]);

                if (!overlaps[key1]) overlaps[key1] = new Set();
                if (!overlaps[key2]) overlaps[key2] = new Set();

                overlaps[key1].add(key2);
                overlaps[key2].add(key1);
            }
        }
    }

    // Convert Sets to arrays for serialization
    const result = {};
    for (const key in overlaps) {
        result[key] = Array.from(overlaps[key]);
    }
    return result;
}

function doMoovesOverlap(m1, m2) {
    const positions1 = new Set(m1.map(p => `${p[0]},${p[1]}`));
    const positions2 = new Set(m2.map(p => `${p[0]},${p[1]}`));

    for (const pos of positions1) {
        if (positions2.has(pos)) return true;
    }
    return false;
}

function generateEmptyBoard(dims) {
    const [height, width] = dims;
    return Array(height).fill(null).map(() => Array(width).fill(false));
}

function getMooveCoverage(board, moove) {
    let mooCoverage = 0;
    for (const t of moove) {
        if (board[t[0]][t[1]] === true) {
            mooCoverage++;
        }
    }
    return mooCoverage;
}

function updateBoardWithMoove(board, mooCount, moove) {
    const [t1, t2, t3] = moove;
    const [r1, c1] = t1;
    const [r2, c2] = t2;
    const [r3, c3] = t3;

    // Deep copy of the board
    const outputBoardState = board.map(row => [...row]);

    const mooCoverage = getMooveCoverage(board, moove);

    if (mooCoverage < 3) {
        outputBoardState[r1][c1] = true;
        outputBoardState[r2][c2] = true;
        outputBoardState[r3][c3] = true;
        mooCount += 1;
    }

    return [outputBoardState, mooCount, 3 - mooCoverage];
}

function generateSequenceGreedilyHigh(allValidMooves, dims, graph, seed = 42) {
    // Simple seeded random for consistency
    let randomSeed = seed;
    const seededRandom = () => {
        randomSeed = (randomSeed * 1103515245 + 12345) & 0x7fffffff;
        return randomSeed / 0x7fffffff;
    };

    let board = generateEmptyBoard(dims);
    const mooveSequence = [];
    let mooCount = 0;
    const mooCountSequence = [];
    const mooCoverageGainSequence = [];

    const remainingMooves = new Set(allValidMooves.map(m => JSON.stringify(m)));

    while (remainingMooves.size > 0) {
        const bestMooveCandidates = [];
        const deadMoves = new Set();

        for (const mooveStr of remainingMooves) {
            const moove = JSON.parse(mooveStr);
            const [, , coverageGain] = updateBoardWithMoove(board, mooCount, moove);

            if (coverageGain <= 0) {
                deadMoves.add(mooveStr);
            } else {
                bestMooveCandidates.push([moove, coverageGain]);
            }
        }

        // Remove dead moves
        for (const deadMove of deadMoves) {
            remainingMooves.delete(deadMove);
        }

        // For greedy-high: Want MIN coverage gain (= MAX overlap) to conserve coverage
        if (bestMooveCandidates.length === 0) break;

        const minCoverageGain = Math.min(...bestMooveCandidates.map(mc => mc[1]));
        const filteredCandidates = bestMooveCandidates.filter(mc => mc[1] === minCoverageGain);

        // Random selection among the best candidates
        const randomIndex = Math.floor(seededRandom() * filteredCandidates.length);
        const bestCandidate = filteredCandidates[randomIndex];

        if (!bestCandidate) break;

        const bestMoove = bestCandidate[0];
        const bestCoverageGain = bestCandidate[1];

        if (bestCoverageGain === 0) break;

        mooveSequence.push(bestMoove);
        [board, mooCount] = updateBoardWithMoove(board, mooCount, bestMoove);

        mooCountSequence.push(mooCount);
        mooCoverageGainSequence.push(bestCoverageGain);
        remainingMooves.delete(JSON.stringify(bestMoove));
    }

    return mooveSequence;
}

function simulateBoard(mooves, dims = [15, 15]) {
    let board = generateEmptyBoard(dims);
    let mooCount = 0;
    const mooCountSequence = [];
    const mooCoverageGainSequence = [];

    for (const moove of mooves) {
        const [newBoard, newCount, coverageGain] = updateBoardWithMoove(board, mooCount, moove);
        board = newBoard;
        mooCount = newCount;
        mooCountSequence.push(mooCount);
        mooCoverageGainSequence.push(coverageGain);
    }

    return [board, mooCount, mooves, mooCountSequence, mooCoverageGainSequence];
}

function workerSimulate(args) {
    const [seed, allValidMooves, dims, graph] = args;
    const sequence = generateSequenceGreedilyHigh(allValidMooves, dims, graph, seed);
    return simulateBoard(sequence, dims);
}

function buildMooCountHistogram(allMooCounts) {
    const histogram = {};
    for (const count of allMooCounts) {
        histogram[count] = (histogram[count] || 0) + 1;
    }

    // Sort keys
    const sorted = {};
    Object.keys(histogram).sort((a, b) => Number(a) - Number(b)).forEach(key => {
        sorted[key] = histogram[key];
    });
    return sorted;
}

function determineDirectionFromMoove(moove) {
    const [t1, t2] = moove;
    const [r1, c1] = t1;
    const [r2, c2] = t2;

    const d = [r2 - r1, c2 - c1];

    const directionMapping = {
        '-1,0': 0,   // up
        '-1,1': 1,   // up-right
        '0,1': 2,    // right
        '1,1': 3,    // down-right
        '1,0': 4,    // down
        '1,-1': 5,   // down-left
        '0,-1': 6,   // left
        '-1,-1': 7   // up-left
    };

    return directionMapping[`${d[0]},${d[1]}`] ?? null;
}

function renderDirectionArrow(direction) {
    const arrows = ['↑', '↗', '→', '↘', '↓', '↙', '←', '↖'];
    return arrows[direction];
}

function renderMoove(moove) {
    const [t1] = moove;
    const direction = determineDirectionFromMoove(moove);
    const arrow = direction !== null ? renderDirectionArrow(direction) : '?';
    // Note: rows are letters (A=0, B=1, etc), columns are 1-indexed numbers
    return `'${String.fromCharCode(t1[0] + 65)},${(t1[1] + 1).toString().padStart(2, ' ')} ${arrow}'`;
}

function renderMooCountHistogram(histogram, screenWidth = 40) {
    const histogramMaxFrequency = Math.max(...Object.values(histogram));
    const maxStars = Math.max(screenWidth, histogramMaxFrequency);
    let output = '';

    for (const [key, value] of Object.entries(histogram)) {
        const scaledBarLength = Math.floor((value / maxStars) * screenWidth);
        const bar = '🐮'.repeat(scaledBarLength);
        output += `Moo count ${key}: ${bar} ${value}\n`;
    }
    return output;
}

function renderMooveSequence(mooveSequence, mooCountSequence, mooCoverageSequence) {
    let accumulativeCoverage = 0;
    let output = 'mooves:      # Moove Number, Moo Count, Coverage Gain, Accumulative Coverage Gain\n';

    for (let i = 0; i < mooveSequence.length; i++) {
        const moove = mooveSequence[i];
        const mooCount = mooCountSequence[i];
        const mooCoverageGain = mooCoverageSequence[i];

        const [t1] = moove;
        accumulativeCoverage += mooCoverageGain;

        const d = determineDirectionFromMoove(moove);
        const directionArrow = d !== null ? renderDirectionArrow(d) : '?';

        const rowLetter = String.fromCharCode(t1[0] + 65);
        const colStr = (t1[1] + 1).toString().padStart(2, ' ');

        if (mooCoverageGain > 0) {
            const commentAnnotation = `# M${i.toString().padStart(5, '0')} ${mooCount} ${mooCoverageGain} ${accumulativeCoverage}`;
            const mooveRecord = `'${rowLetter},${colStr} ${directionArrow}'`;
            output += `  - ${mooveRecord} ${commentAnnotation}\n`;
        }
    }

    return output;
}

async function parallelProcessSimulations(grid, iterations, workers) {
    const allValidMooves = generateAllValidMooves(grid);
    const dims = gridDimensions(grid);
    const [height, width] = dims;
    const allCells = height * width;
    const graph = generateOverlapsGraph(allValidMooves);

    console.log(`Total valid 'moo' moves found: ${allValidMooves.length}`);

    const graphObj = {};
    for (const moove of allValidMooves) {
        const key = JSON.stringify(moove);
        if (graph[key]) {
            graphObj[key] = graph[key].length;
        }
    }

    const maxOverlaps = Math.max(...Object.values(graphObj).filter(v => !isNaN(v)), 0);
    console.log(`Graph of overlapping Mooves has ${Object.keys(graph).length} nodes. And highest degree node has ${maxOverlaps} overlaps.`);

    const timeStart = performance.now();

    // Create args for simulations
    const workerArgs = [];
    for (let i = 0; i < iterations; i++) {
        workerArgs.push([i, allValidMooves, dims, graph]);
    }

    const P = workers > 0 ? workers : os.cpus().length;
    const optimalChunksize = Math.max(1, Math.floor(iterations / (P * 4)));
    console.log(`Using ${P} processes with chunksize ${optimalChunksize}`);

    const timeSimsStart = performance.now();

    // Simulate in batches (simplified version without worker threads for now)
    const allSimulations = [];
    for (const args of workerArgs) {
        const result = workerSimulate(args);
        allSimulations.push(result);
    }

    const timeParallelEnd = performance.now();
    const timeParallelDuration = (timeParallelEnd - timeSimsStart) / 1000;
    console.log(`Simulations complete took ${timeParallelDuration.toFixed(2)}s, (${Math.floor(iterations / timeParallelDuration)} simulations per second)`);

    const allMooCounts = allSimulations.map(s => s[1]);
    const maxResult = allSimulations.reduce((max, curr) => curr[1] > max[1] ? curr : max);
    const minResult = allSimulations.reduce((min, curr) => curr[1] < min[1] ? curr : min);

    const [maxBoard, maxMooves, maxMooveSequence, maxMooCountSequence, maxMooCoverageSequence] = maxResult;
    const [minBoard, minMooves, minMooveSequence, minMooCountSequence, minMooCoverageSequence] = minResult;

    const timeReduceEnd = performance.now();
    const timeReduceDuration = (timeReduceEnd - timeParallelEnd) / 1000;
    console.log(`Result processing took ${timeReduceDuration.toFixed(2)}s after ${timeParallelDuration.toFixed(2)}s of parallel simulation.`);

    const totalTime = (performance.now() - timeStart) / 1000;
    const totalSimsTime = (performance.now() - timeSimsStart) / 1000;

    console.log(`Time taken for parallel simulation: ${totalSimsTime.toFixed(2)}s`);
    console.log(`Total time taken: ${totalTime.toFixed(2)}s`);
    console.log(`Simulations per second: ${Math.floor(iterations / totalSimsTime)}`);
    console.log();

    const histogram = buildMooCountHistogram(allMooCounts);
    const maxCoverage = maxMooCoverageSequence.reduce((sum, val) => sum + val, 0);
    const deadCells = allCells - maxCoverage;

    // Build graph degrees
    const graphDegrees = {};
    for (const moove of allValidMooves) {
        const key = JSON.stringify(moove);
        if (graph[key]) {
            graphDegrees[renderMoove(moove)] = graph[key].length;
        }
    }

    // Sort graph degrees
    const sortedGraphDegrees = Object.entries(graphDegrees)
        .sort((a, b) => b[1] - a[1])
        .reduce((acc, [key, val]) => ({ ...acc, [key]: val }), {});

    return {
        allValidMooves,
        maxCoverage,
        deadCells,
        maxMooves,
        minMooves,
        histogram,
        graph,
        maxMooveSequence,
        maxMooCountSequence,
        maxMooCoverageSequence,
        minMooveSequence,
        minMooCountSequence,
        minMooCoverageSequence,
        graphDegrees: sortedGraphDegrees
    };
}

async function main() {
    const args = process.argv.slice(2);

    // Parse command line arguments
    let puzzlePath = null;
    let iterations = 1000;
    let workers = -1;

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--puzzle' && i + 1 < args.length) {
            puzzlePath = args[i + 1];
            i++;
        } else if (args[i] === '--iterations' && i + 1 < args.length) {
            iterations = parseInt(args[i + 1]);
            i++;
        } else if (args[i] === '--workers' && i + 1 < args.length) {
            workers = parseInt(args[i + 1]);
            i++;
        }
    }

    if (!puzzlePath) {
        console.error('Usage: node index.js --puzzle <path-to-puzzle-file> [--iterations <n>] [--workers <n>]');
        process.exit(1);
    }

    const grid = gridFromFile(puzzlePath);
    const dims = gridDimensions(grid);
    const puzzleName = path.basename(puzzlePath, path.extname(puzzlePath));
    const outputFilepath = path.join(OUTPUT_DIR, `${puzzleName}.json`);

    const simulationOutput = await parallelProcessSimulations(grid, iterations, workers);

    const {
        allValidMooves,
        maxCoverage,
        deadCells,
        maxMooves,
        minMooves,
        histogram,
        graphDegrees,
        maxMooveSequence,
        maxMooCountSequence,
        maxMooCoverageSequence,
        minMooveSequence,
        minMooCountSequence,
        minMooCoverageSequence
    } = simulationOutput;

    const output = {
        puzzle: puzzleName,
        dimensions: dims,
        total_cells: dims[0] * dims[1],
        total_valid_mooves: allValidMooves.length,
        max_coverage: maxCoverage,
        dead_cells: deadCells,
        max_mooves: maxMooves,
        min_mooves: minMooves,
        moo_count_histogram: histogram,
        graph_degrees: graphDegrees,
        max_moove_sequence: maxMooveSequence,
        max_moo_count_sequence: maxMooCountSequence,
        max_moo_coverage_sequence: maxMooCoverageSequence,
        min_moove_sequence: minMooveSequence,
        min_moo_count_sequence: minMooCountSequence,
        min_moo_coverage_sequence: minMooCoverageSequence,
        rendered_max_moove_sequence: renderMooveSequence(
            maxMooveSequence,
            maxMooCountSequence,
            maxMooCoverageSequence
        ),
        rendered_min_moove_sequence: renderMooveSequence(
            minMooveSequence,
            minMooCountSequence,
            minMooCoverageSequence
        )
    };

    fs.writeFileSync(outputFilepath, JSON.stringify(output, null, 2));

    console.log(`PUZZLE: ${puzzleName}`);
    console.log(renderMooveSequence(
        maxMooveSequence,
        maxMooCountSequence,
        maxMooCoverageSequence
    ));
    console.log(renderMooCountHistogram(histogram, 40));
    console.log(`Max mooves: ${maxMooves}, Min mooves: ${minMooves}, Max coverage: ${maxCoverage}, Dead cells: ${deadCells}`);
    console.log(`Output written to ${outputFilepath}`);
}

// Run the main function
if (require.main === module) {
    main().catch(console.error);
}