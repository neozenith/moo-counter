// Load and display statistics for all puzzle sizes
async function loadStatistics() {
    const puzzleSizes = ['micro', 'mini', 'maxi'];

    for (const size of puzzleSizes) {
        try {
            const response = await fetch(`${size}.json`);
            const data = await response.json();

            // Update statistics
            document.getElementById(`${size}-moves`).textContent = data.total_valid_mooves;
            document.getElementById(`${size}-coverage`).textContent = `${data.max_coverage}/${data.total_cells} (${Math.round(data.max_coverage / data.total_cells * 100)}%)`;
            document.getElementById(`${size}-dead`).textContent = data.dead_cells;
            document.getElementById(`${size}-max`).textContent = data.max_mooves;

            // Display solutions
            const solutionsDiv = document.getElementById(`${size}-solutions`);

            // Create max solution section
            const maxSolutionHTML = `
                <div class="solution-section">
                    <h3>Maximum Moo Score Solution (${data.max_mooves} moves)</h3>
                    <div class="move-sequence">
                        ${formatMoveSequence(data.rendered_max_moove_sequence, data.max_coverage)}
                    </div>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-label">Final Coverage:</span>
                            <span class="stat-value">${data.max_coverage} cells (${Math.round(data.max_coverage / data.total_cells * 100)}%)</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Move Count:</span>
                            <span class="stat-value">${data.max_mooves}</span>
                        </div>
                    </div>
                </div>
            `;

            // Calculate min solution coverage
            const minCoverage = data.min_moo_coverage_sequence.reduce((a, b) => a + b, 0);

            // Create min solution section
            const minSolutionHTML = `
                <div class="solution-section">
                    <h3>Minimum Moves Solution (${data.min_mooves} moves)</h3>
                    <div class="move-sequence">
                        ${formatMoveSequence(data.rendered_min_moove_sequence, minCoverage)}
                    </div>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <span class="stat-label">Final Coverage:</span>
                            <span class="stat-value">${minCoverage} cells (${Math.round(minCoverage / data.total_cells * 100)}%)</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Move Count:</span>
                            <span class="stat-value">${data.min_mooves}</span>
                        </div>
                    </div>
                </div>
            `;

            // Create histogram section
            const histogramHTML = `
                <div class="solution-section">
                    <h3>Move Count Distribution</h3>
                    <div class="histogram">
                        ${Object.entries(data.moo_count_histogram)
                            .map(([moves, count]) => `
                                <div class="histogram-bar">
                                    <div class="bar-label">${moves} moves</div>
                                    <div class="bar-count">${count} solutions</div>
                                </div>
                            `).join('')}
                    </div>
                </div>
            `;

            solutionsDiv.innerHTML = maxSolutionHTML + minSolutionHTML + histogramHTML;

        } catch (error) {
            console.error(`Error loading ${size}.json:`, error);
            document.getElementById(`${size}-moves`).textContent = 'Error loading data';
        }
    }
}

// Format move sequence for display
function formatMoveSequence(sequence, finalCoverage) {
    if (!sequence) return 'No sequence data available';

    // Parse the YAML-like format
    const lines = sequence.split('\n').filter(line => line.trim().startsWith("- '"));

    // Build table HTML
    let tableHTML = `
        <table class="moves-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Move</th>
                    <th>Points</th>
                    <th>Coverage Gain</th>
                    <th>Total Coverage</th>
                    <th>Coverage %</th>
                </tr>
            </thead>
            <tbody>
    `;

    lines.forEach((line, index) => {
        const match = line.match(/- '([^']+)'/);
        if (match) {
            const moveData = match[1];
            const comment = line.split('#')[1] || '';

            // Parse the comment data: M00000 1 3 3
            const commentParts = comment.trim().split(/\s+/);
            const movePoints = commentParts[1] || '-';
            const coverageGain = commentParts[2] || '-';
            const totalCoverage = commentParts[3] || '-';
            const coveragePercent = finalCoverage && totalCoverage !== '-'
                ? Math.round((parseInt(totalCoverage) / finalCoverage) * 100) + '%'
                : '-';

            tableHTML += `
                <tr>
                    <td class="move-number">${index + 1}</td>
                    <td class="move-action">${moveData}</td>
                    <td class="move-points">${movePoints}</td>
                    <td class="move-coverage-gain">${coverageGain}</td>
                    <td class="move-total-coverage">${totalCoverage}</td>
                    <td class="move-coverage-percent">${coveragePercent}</td>
                </tr>
            `;
        }
    });

    tableHTML += `
            </tbody>
        </table>
    `;

    return tableHTML;
}

// Load statistics when page loads
document.addEventListener('DOMContentLoaded', loadStatistics);