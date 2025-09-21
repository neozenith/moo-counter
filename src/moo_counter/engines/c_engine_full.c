#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>

// Direction vectors for movement (matches Python order)
static int DIRECTIONS[8][2] = {
    {-1, 0},   // up
    {-1, 1},   // up-right
    {0, 1},    // right
    {1, 1},    // down-right
    {1, 0},    // down
    {1, -1},   // down-left
    {0, -1},   // left
    {-1, -1},  // up-left
};

// Direction symbols (matches direction order)
static const char* DIRECTION_SYMBOLS[8] = {
    "↑", "↗", "→", "↘", "↓", "↙", "←", "↖"
};

// CEngine structure
typedef struct {
    PyObject_HEAD
    char* name;
} CEngine;

// Helper function to check if a position is valid
static bool is_valid_position(int row, int col, int rows, int cols) {
    return row >= 0 && row < rows && col >= 0 && col < cols;
}

// Helper function to check if a moove spells 'moo' on the grid
static bool check_moo_letters(PyObject* grid, int start_row, int start_col, int dir_idx) {
    if (!PyList_Check(grid)) return false;

    int dr = DIRECTIONS[dir_idx][0];
    int dc = DIRECTIONS[dir_idx][1];

    // Check first position for 'm'
    PyObject* row1 = PyList_GetItem(grid, start_row);
    if (!row1) return false;
    PyObject* cell1 = PyList_GetItem(row1, start_col);
    if (!cell1) return false;
    const char* str1 = PyUnicode_AsUTF8(cell1);
    if (!str1 || str1[0] != 'm') return false;

    // Check second position for 'o'
    int r2 = start_row + dr;
    int c2 = start_col + dc;
    PyObject* row2 = PyList_GetItem(grid, r2);
    if (!row2) return false;
    PyObject* cell2 = PyList_GetItem(row2, c2);
    if (!cell2) return false;
    const char* str2 = PyUnicode_AsUTF8(cell2);
    if (!str2 || str2[0] != 'o') return false;

    // Check third position for 'o'
    int r3 = start_row + 2 * dr;
    int c3 = start_col + 2 * dc;
    PyObject* row3 = PyList_GetItem(grid, r3);
    if (!row3) return false;
    PyObject* cell3 = PyList_GetItem(row3, c3);
    if (!cell3) return false;
    const char* str3 = PyUnicode_AsUTF8(cell3);
    if (!str3 || str3[0] != 'o') return false;

    return true;
}

// Method: get_name
static PyObject* CEngine_get_name(CEngine* self, PyObject* Py_UNUSED(ignored)) {
    return PyUnicode_FromString(self->name);
}

// Method: generate_all_valid_mooves - now checks grid letters
static PyObject* CEngine_generate_all_valid_mooves(CEngine* self, PyObject* args) {
    PyObject* grid;
    if (!PyArg_ParseTuple(args, "O", &grid)) {
        return NULL;
    }

    if (!PyList_Check(grid)) {
        PyErr_SetString(PyExc_TypeError, "grid must be a list");
        return NULL;
    }

    Py_ssize_t rows = PyList_Size(grid);
    if (rows == 0) {
        return PyList_New(0);
    }

    PyObject* first_row = PyList_GetItem(grid, 0);
    if (!PyList_Check(first_row)) {
        PyErr_SetString(PyExc_TypeError, "grid must be a list of lists");
        return NULL;
    }
    Py_ssize_t cols = PyList_Size(first_row);

    PyObject* mooves = PyList_New(0);
    if (!mooves) return NULL;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            for (int dir_idx = 0; dir_idx < 8; dir_idx++) {
                // Check if all positions are in bounds
                bool valid = true;
                int dr = DIRECTIONS[dir_idx][0];
                int dc = DIRECTIONS[dir_idx][1];

                for (int i = 0; i < 3; i++) {
                    int r = row + i * dr;
                    int c = col + i * dc;
                    if (!is_valid_position(r, c, rows, cols)) {
                        valid = false;
                        break;
                    }
                }

                if (valid && check_moo_letters(grid, row, col, dir_idx)) {
                    PyObject* moove_tuple = Py_BuildValue("(iii)", row, col, dir_idx);
                    if (!moove_tuple) {
                        Py_DECREF(mooves);
                        return NULL;
                    }
                    PyList_Append(mooves, moove_tuple);
                    Py_DECREF(moove_tuple);
                }
            }
        }
    }

    return mooves;
}

// Method: do_mooves_overlap - checks if two mooves share any cells
static PyObject* CEngine_do_mooves_overlap(CEngine* self, PyObject* args) {
    PyObject *m1, *m2;
    if (!PyArg_ParseTuple(args, "OO", &m1, &m2)) {
        return NULL;
    }

    // Extract positions from both mooves
    PyObject *t1_1, *t2_1, *t3_1;
    PyObject *t1_2, *t2_2, *t3_2;

    if (!PyArg_ParseTuple(m1, "OOO", &t1_1, &t2_1, &t3_1)) {
        return NULL;
    }

    if (!PyArg_ParseTuple(m2, "OOO", &t1_2, &t2_2, &t3_2)) {
        return NULL;
    }

    // Check if any positions match
    if (PyObject_RichCompareBool(t1_1, t1_2, Py_EQ) == 1 ||
        PyObject_RichCompareBool(t1_1, t2_2, Py_EQ) == 1 ||
        PyObject_RichCompareBool(t1_1, t3_2, Py_EQ) == 1 ||
        PyObject_RichCompareBool(t2_1, t1_2, Py_EQ) == 1 ||
        PyObject_RichCompareBool(t2_1, t2_2, Py_EQ) == 1 ||
        PyObject_RichCompareBool(t2_1, t3_2, Py_EQ) == 1 ||
        PyObject_RichCompareBool(t3_1, t1_2, Py_EQ) == 1 ||
        PyObject_RichCompareBool(t3_1, t2_2, Py_EQ) == 1 ||
        PyObject_RichCompareBool(t3_1, t3_2, Py_EQ) == 1) {
        Py_RETURN_TRUE;
    }

    Py_RETURN_FALSE;
}

// Method: generate_overlaps_graph
static PyObject* CEngine_generate_overlaps_graph(CEngine* self, PyObject* args) {
    PyObject* mooves;
    if (!PyArg_ParseTuple(args, "O", &mooves)) {
        return NULL;
    }

    if (!PyList_Check(mooves)) {
        PyErr_SetString(PyExc_TypeError, "mooves must be a list");
        return NULL;
    }

    Py_ssize_t num_mooves = PyList_Size(mooves);
    PyObject* graph = PyDict_New();
    if (!graph) return NULL;

    for (Py_ssize_t i = 0; i < num_mooves; i++) {
        PyObject* moove_i = PyList_GetItem(mooves, i);
        PyObject* overlaps_set = PySet_New(NULL);
        if (!overlaps_set) {
            Py_DECREF(graph);
            return NULL;
        }

        for (Py_ssize_t j = 0; j < num_mooves; j++) {
            if (i == j) continue;

            PyObject* moove_j = PyList_GetItem(mooves, j);

            // Check overlap
            PyObject* overlap_args = Py_BuildValue("(OO)", moove_i, moove_j);
            PyObject* overlaps = CEngine_do_mooves_overlap(self, overlap_args);
            Py_DECREF(overlap_args);

            if (overlaps == Py_True) {
                PySet_Add(overlaps_set, moove_j);
            }
            Py_DECREF(overlaps);
        }

        PyDict_SetItem(graph, moove_i, overlaps_set);
        Py_DECREF(overlaps_set);
    }

    return graph;
}

// Method: generate_empty_board
static PyObject* CEngine_generate_empty_board(CEngine* self, PyObject* args) {
    int rows, cols;
    if (!PyArg_ParseTuple(args, "(ii)", &rows, &cols)) {
        return NULL;
    }

    PyObject* board = PyList_New(rows);
    if (!board) return NULL;

    for (int r = 0; r < rows; r++) {
        PyObject* row = PyList_New(cols);
        if (!row) {
            Py_DECREF(board);
            return NULL;
        }

        for (int c = 0; c < cols; c++) {
            Py_INCREF(Py_False);
            PyList_SET_ITEM(row, c, Py_False);
        }

        PyList_SET_ITEM(board, r, row);
    }

    return board;
}

// Method: update_board_with_moove
static PyObject* CEngine_update_board_with_moove(CEngine* self, PyObject* args) {
    PyObject *board, *moove;
    int moo_count;

    if (!PyArg_ParseTuple(args, "OiO", &board, &moo_count, &moove)) {
        return NULL;
    }

    // Deep copy board
    PyObject* copy_module = PyImport_ImportModule("copy");
    PyObject* deepcopy = PyObject_GetAttrString(copy_module, "deepcopy");
    PyObject* new_board = PyObject_CallFunction(deepcopy, "O", board);
    Py_DECREF(copy_module);
    Py_DECREF(deepcopy);

    if (!new_board) return NULL;

    // Extract positions from moove
    PyObject *t1, *t2, *t3;
    if (!PyArg_ParseTuple(moove, "OOO", &t1, &t2, &t3)) {
        Py_DECREF(new_board);
        return NULL;
    }

    int coverage_gain = 0;
    PyObject* positions[] = {t1, t2, t3};

    for (int i = 0; i < 3; i++) {
        int r, c;
        if (!PyArg_ParseTuple(positions[i], "ii", &r, &c)) {
            Py_DECREF(new_board);
            return NULL;
        }

        PyObject* row = PyList_GetItem(new_board, r);
        PyObject* cell = PyList_GetItem(row, c);

        if (cell == Py_False) {
            coverage_gain++;
        }
    }

    // Only update board and increment count if we're adding new cells
    if (coverage_gain > 0) {
        for (int i = 0; i < 3; i++) {
            int r, c;
            if (!PyArg_ParseTuple(positions[i], "ii", &r, &c)) {
                Py_DECREF(new_board);
                return NULL;
            }

            PyObject* row = PyList_GetItem(new_board, r);
            PyObject* new_val = PyLong_FromLong(moo_count);
            PyList_SetItem(row, c, new_val);
        }
        return Py_BuildValue("(Oii)", new_board, moo_count + 1, coverage_gain);
    } else {
        // No new cells, don't increment moo_count
        return Py_BuildValue("(Oii)", new_board, moo_count, coverage_gain);
    }
}

// Method: simulate_board
static PyObject* CEngine_simulate_board(CEngine* self, PyObject* args) {
    PyObject *mooves, *dims;

    if (!PyArg_ParseTuple(args, "OO", &mooves, &dims)) {
        return NULL;
    }

    // Generate empty board
    PyObject* board = CEngine_generate_empty_board(self, Py_BuildValue("(O)", dims));
    if (!board) return NULL;

    int moo_count = 1;
    PyObject* moove_sequence = PyList_New(0);
    PyObject* moo_count_sequence = PyList_New(0);
    PyObject* moo_coverage_gain_sequence = PyList_New(0);

    Py_ssize_t num_mooves = PyList_Size(mooves);

    for (Py_ssize_t i = 0; i < num_mooves; i++) {
        PyObject* moove = PyList_GetItem(mooves, i);

        // Update board
        PyObject* update_args = Py_BuildValue("(OiO)", board, moo_count, moove);
        PyObject* result = CEngine_update_board_with_moove(self, update_args);
        Py_DECREF(update_args);

        if (!result) {
            Py_DECREF(board);
            Py_DECREF(moove_sequence);
            Py_DECREF(moo_count_sequence);
            Py_DECREF(moo_coverage_gain_sequence);
            return NULL;
        }

        PyObject *new_board;
        int new_moo_count, coverage_gain;
        if (!PyArg_ParseTuple(result, "Oii", &new_board, &new_moo_count, &coverage_gain)) {
            Py_DECREF(result);
            Py_DECREF(board);
            Py_DECREF(moove_sequence);
            Py_DECREF(moo_count_sequence);
            Py_DECREF(moo_coverage_gain_sequence);
            return NULL;
        }

        Py_DECREF(board);
        board = new_board;
        Py_INCREF(board);
        moo_count = new_moo_count;

        PyList_Append(moove_sequence, moove);
        PyList_Append(moo_count_sequence, PyLong_FromLong(moo_count - 1));
        PyList_Append(moo_coverage_gain_sequence, PyLong_FromLong(coverage_gain));

        Py_DECREF(result);
    }

    // Build result dictionary matching SimulationResult
    PyObject* result_dict = PyDict_New();
    PyDict_SetItemString(result_dict, "board", board);
    PyDict_SetItemString(result_dict, "moo_count", PyLong_FromLong(moo_count - 1));
    PyDict_SetItemString(result_dict, "moove_sequence", moove_sequence);
    PyDict_SetItemString(result_dict, "moo_count_sequence", moo_count_sequence);
    PyDict_SetItemString(result_dict, "moo_coverage_gain_sequence", moo_coverage_gain_sequence);

    Py_DECREF(board);
    Py_DECREF(moove_sequence);
    Py_DECREF(moo_count_sequence);
    Py_DECREF(moo_coverage_gain_sequence);

    return result_dict;
}

// Constructor
static int CEngine_init(CEngine* self, PyObject* args, PyObject* kwds) {
    self->name = strdup("C");
    return 0;
}

// Destructor
static void CEngine_dealloc(CEngine* self) {
    if (self->name) {
        free(self->name);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// Method definitions
static PyMethodDef CEngine_methods[] = {
    {"get_name", (PyCFunction)CEngine_get_name, METH_NOARGS, "Get engine name"},
    {"generate_all_valid_mooves", (PyCFunction)CEngine_generate_all_valid_mooves, METH_VARARGS, "Generate all valid mooves"},
    {"do_mooves_overlap", (PyCFunction)CEngine_do_mooves_overlap, METH_VARARGS, "Check if two mooves overlap"},
    {"generate_overlaps_graph", (PyCFunction)CEngine_generate_overlaps_graph, METH_VARARGS, "Generate overlap graph"},
    {"generate_empty_board", (PyCFunction)CEngine_generate_empty_board, METH_VARARGS, "Generate empty board"},
    {"update_board_with_moove", (PyCFunction)CEngine_update_board_with_moove, METH_VARARGS, "Update board with moove"},
    {"simulate_board", (PyCFunction)CEngine_simulate_board, METH_VARARGS, "Simulate board"},
    {NULL}
};

// Type definition
static PyTypeObject CEngineType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "c_engine_full.CEngine",
    .tp_doc = "Complete C implementation of game engine",
    .tp_basicsize = sizeof(CEngine),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)CEngine_init,
    .tp_dealloc = (destructor)CEngine_dealloc,
    .tp_methods = CEngine_methods,
};

// Module definition
static struct PyModuleDef c_engine_full_module = {
    PyModuleDef_HEAD_INIT,
    "c_engine_full",
    "Complete C implementation of the game engine",
    -1,
    NULL
};

// Module initialization
PyMODINIT_FUNC PyInit_c_engine_full(void) {
    PyObject* m;
    if (PyType_Ready(&CEngineType) < 0)
        return NULL;

    m = PyModule_Create(&c_engine_full_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&CEngineType);
    if (PyModule_AddObject(m, "CEngine", (PyObject*)&CEngineType) < 0) {
        Py_DECREF(&CEngineType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}