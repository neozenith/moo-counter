"""Display and rendering functions for Moo Counter."""

from .moo_types import (
    DIRECTION_ARROWS,
    DIRECTION_MAPPING,
    BoardState,
    Direction,
    Grid,
    MooCountHistogram,
    Moove,
    MooveCountSequence,
    MooveCoverageGainSequence,
    MooveSequence,
)


def render_board(board: BoardState, grid: Grid) -> str:
    """Render the board state with upper/lowercase letters."""
    output = ""
    for r, row in enumerate(board):
        row_str = ""
        for c, cell in enumerate(row):
            if cell is True:
                row_str += grid[r][c].upper() + " "
            elif cell is False:
                row_str += grid[r][c].lower() + " "
            else:
                row_str += f"{cell} "
        output += row_str.strip() + "\n"
    return output


def determine_direction_from_moove(moove: Moove) -> Direction | None:
    """Determine the direction of a moove."""
    t1, t2, _ = moove
    r1, c1 = t1
    r2, c2 = t2
    d = (r2 - r1, c2 - c1)
    return DIRECTION_MAPPING.get(d, None)


def render_direction_arrow(direction: Direction) -> str:
    """Get the arrow character for a direction."""
    return DIRECTION_ARROWS[direction]


def render_moove(moove: Moove) -> str:
    """Render a single moove in the format 'A, 1 →' or 'H,12 ↖'."""
    t1, _, _ = moove
    direction = determine_direction_from_moove(moove)
    arrow = render_direction_arrow(direction) if direction is not None else "?"
    return f"'{chr(t1[0] + 65)},{t1[1]+1:>2} {arrow}'"


def render_moove_sequence(
    moove_sequence: MooveSequence,
    moo_count_sequence: MooveCountSequence,
    moo_coverage_sequence: MooveCoverageGainSequence,
) -> str:
    """Render a sequence of mooves with statistics."""
    accumulative_coverage = 0
    output = "mooves:      # Moove Number, Moo Count, Coverage Gain, Accumulative Coverage Gain\n"

    for i, (moove, moo_count, moo_coverage_gain) in enumerate(
        zip(moove_sequence, moo_count_sequence, moo_coverage_sequence, strict=True)
    ):
        t1, _, _ = moove
        accumulative_coverage += moo_coverage_gain
        d = determine_direction_from_moove(moove)
        direction_arrow = render_direction_arrow(d) if d is not None else "?"

        row_letter = chr(t1[0] + 65)  # Map 0->A, 1->B, 2->C, ...
        col_str = f"{t1[1]+1:>2}"

        if moo_coverage_gain > 0:
            comment_annotation = f"# M{i:05d} {moo_count} {moo_coverage_gain} {accumulative_coverage}"
            moove_record = f"'{row_letter},{col_str} {direction_arrow}'"
            output += f"  - {moove_record} {comment_annotation}\n"

    return output


def render_moo_count_histogram(histogram: MooCountHistogram, screen_width: int = 40) -> str:
    """Render a histogram of moo counts with cow emojis."""
    if not histogram:
        return "No histogram data\n"

    histogram_max_frequency = float(max(histogram.values()))
    max_stars = max(screen_width, histogram_max_frequency)
    output = ""

    for key, value in histogram.items():
        scaled_bar_length = int((value / max_stars) * screen_width)
        bar = "🐮" * scaled_bar_length
        output += f"Moo count {key}: {bar} {value}\n"

    return output


def generate_cytoscape_graph(
    all_valid_mooves: MooveSequence,
    graph: dict[Moove, set[Moove]],
    max_moove_sequence: MooveSequence,
    max_moo_count_sequence: MooveCountSequence,
    max_moo_coverage_sequence: MooveCoverageGainSequence,
    dims: tuple[int, int],
) -> dict:
    """Generate Cytoscape.js compatible graph data for visualization.

    Creates a graph with:
    - START and END terminal nodes
    - All valid mooves as nodes
    - Path edges showing the max_moove_sequence
    - Possible edges showing other connections
    """
    nodes = []
    edges = []

    def moove_to_id(moove: Moove) -> str:
        """Convert a moove to a unique ID."""
        t1, _, _ = moove
        direction = determine_direction_from_moove(moove)
        arrow = render_direction_arrow(direction) if direction is not None else "?"
        return f"{chr(t1[0] + 65)}_{t1[1]+1}_{arrow}"

    # Add START node
    nodes.append({"data": {"id": "start", "label": "START", "type": "terminal"}, "classes": "start-node"})

    # Add all moove nodes
    moove_to_node_map = {}
    for moove in all_valid_mooves:
        node_id = moove_to_id(moove)
        moove_to_node_map[moove] = node_id

        in_path = moove in max_moove_sequence

        nodes.append(
            {
                "data": {
                    "id": node_id,
                    "label": render_moove(moove).strip("'"),
                    "degree": len(graph.get(moove, [])),
                    "position": {"row": moove[0][0], "col": moove[0][1]},
                    "in_path": in_path,
                },
                "classes": "path-node" if in_path else "regular-node",
            }
        )

    # Add END node
    total_coverage = sum(max_moo_coverage_sequence) if max_moo_coverage_sequence else 0
    nodes.append(
        {
            "data": {
                "id": "end",
                "label": "END",
                "type": "terminal",
                "coverage": total_coverage,
                "score": len(max_moove_sequence),
            },
            "classes": "end-node",
        }
    )

    # Add path edges for max_moove_sequence
    if max_moove_sequence:
        # Edge from START to first moove
        first_moove_id = moove_to_node_map[max_moove_sequence[0]]
        edges.append(
            {
                "data": {
                    "id": f"start_to_{first_moove_id}",
                    "source": "start",
                    "target": first_moove_id,
                    "sequence": 0,
                    "coverage_gain": max_moo_coverage_sequence[0] if max_moo_coverage_sequence else 0,
                    "total_coverage": max_moo_coverage_sequence[0] if max_moo_coverage_sequence else 0,
                    "moo_count": 1,
                },
                "classes": "path-edge",
            }
        )

        # Edges between consecutive mooves
        total_coverage = max_moo_coverage_sequence[0] if max_moo_coverage_sequence else 0
        for i in range(len(max_moove_sequence) - 1):
            source_id = moove_to_node_map[max_moove_sequence[i]]
            target_id = moove_to_node_map[max_moove_sequence[i + 1]]
            coverage_gain = max_moo_coverage_sequence[i + 1] if i + 1 < len(max_moo_coverage_sequence) else 0
            total_coverage += coverage_gain

            edges.append(
                {
                    "data": {
                        "id": f"{source_id}_to_{target_id}",
                        "source": source_id,
                        "target": target_id,
                        "sequence": i + 1,
                        "coverage_gain": coverage_gain,
                        "total_coverage": total_coverage,
                        "moo_count": max_moo_count_sequence[i + 1] if i + 1 < len(max_moo_count_sequence) else 0,
                    },
                    "classes": "path-edge",
                }
            )

        # Edge from last moove to END
        last_moove_id = moove_to_node_map[max_moove_sequence[-1]]
        edges.append(
            {
                "data": {
                    "id": f"{last_moove_id}_to_end",
                    "source": last_moove_id,
                    "target": "end",
                    "sequence": len(max_moove_sequence),
                    "coverage_gain": 0,
                    "total_coverage": total_coverage,
                    "moo_count": max_moo_count_sequence[-1] if max_moo_count_sequence else 0,
                },
                "classes": "path-edge",
            }
        )

    # Add possible edges based on overlap graph
    max_moove_set = set(max_moove_sequence)
    for moove1, overlapping_mooves in graph.items():
        source_id = moove_to_node_map[moove1]

        for moove2 in overlapping_mooves:
            # Skip if this edge is part of the path
            if moove1 in max_moove_set and moove2 in max_moove_set:
                idx1 = max_moove_sequence.index(moove1) if moove1 in max_moove_sequence else -1
                idx2 = max_moove_sequence.index(moove2) if moove2 in max_moove_sequence else -1
                if abs(idx1 - idx2) == 1:
                    continue

            target_id = moove_to_node_map[moove2]
            edge_id = f"{source_id}_overlap_{target_id}"

            if not any(e["data"]["id"] == edge_id for e in edges):
                edges.append(
                    {"data": {"id": edge_id, "source": source_id, "target": target_id}, "classes": "possible-edge"}
                )

    # Create complete Cytoscape data structure with styling
    return {
        "elements": {"nodes": nodes, "edges": edges},
        "style": _get_cytoscape_styles(),
        "layout": {
            "name": "cose",
            "animate": False,
            "nodeDimensionsIncludeLabels": True,
            "nodeRepulsion": 400000,
            "idealEdgeLength": 100,
            "edgeElasticity": 100,
            "nestingFactor": 5,
            "gravity": 80,
            "numIter": 1000,
            "componentSpacing": 100,
        },
    }


def _get_cytoscape_styles() -> list[dict]:
    """Get the Cytoscape styling configuration."""
    return [
        {
            "selector": "node",
            "style": {
                "label": "data(label)",
                "background-color": "#666",
                "text-valign": "center",
                "text-halign": "center",
                "width": 30,
                "height": 30,
                "font-size": 10,
            },
        },
        {
            "selector": ".start-node",
            "style": {
                "background-color": "#27ae60",
                "shape": "diamond",
                "width": 50,
                "height": 50,
                "font-weight": "bold",
            },
        },
        {
            "selector": ".end-node",
            "style": {
                "background-color": "#e74c3c",
                "shape": "diamond",
                "width": 50,
                "height": 50,
                "font-weight": "bold",
            },
        },
        {
            "selector": ".path-node",
            "style": {"background-color": "#3498db", "border-width": 3, "border-color": "#2980b9"},
        },
        {"selector": ".regular-node", "style": {"background-color": "#95a5a6", "opacity": 0.7}},
        {
            "selector": ".path-edge",
            "style": {
                "line-color": "#3498db",
                "target-arrow-color": "#3498db",
                "width": 4,
                "curve-style": "bezier",
                "target-arrow-shape": "triangle",
                "label": "data(sequence)",
                "text-rotation": "autorotate",
                "text-margin-y": -10,
                "font-size": 12,
                "font-weight": "bold",
            },
        },
        {
            "selector": ".possible-edge",
            "style": {"line-color": "#333", "width": 1, "line-style": "dashed", "opacity": 0.3},
        },
    ]
