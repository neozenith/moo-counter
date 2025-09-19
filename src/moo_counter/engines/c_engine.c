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

// Helper function to get cell positions for a moove
static PyObject* get_moove_cells(int start_row, int start_col, int dir_idx, int rows, int cols) {
    PyObject* cells = PyList_New(0);
    if (!cells) return NULL;

    int dr = DIRECTIONS[dir_idx][0];
    int dc = DIRECTIONS[dir_idx][1];

    // Calculate all three cells
    for (int i = 0; i < 3; i++) {
        int r = start_row + i * dr;
        int c = start_col + i * dc;

        if (!is_valid_position(r, c, rows, cols)) {
            Py_DECREF(cells);
            Py_RETURN_NONE;
        }

        PyObject* pos = Py_BuildValue("(ii)", r, c);
        if (!pos) {
            Py_DECREF(cells);
            return NULL;
        }
        PyList_Append(cells, pos);
        Py_DECREF(pos);
    }

    return cells;
}

// Method: get_name
static PyObject* CEngine_get_name(CEngine* self, PyObject* Py_UNUSED(ignored)) {
    return PyUnicode_FromString(self->name);
}

// Method: generate_moove_str
static PyObject* CEngine_generate_moove_str(CEngine* self, PyObject* args) {
    int row, col, dir_idx;
    if (!PyArg_ParseTuple(args, "iii", &row, &col, &dir_idx)) {
        return NULL;
    }

    char col_letter = 'A' + col;
    int row_num = row + 1;
    const char* dir_symbol = DIRECTION_SYMBOLS[dir_idx];

    return PyUnicode_FromFormat("'%c, %d %s'", col_letter, row_num, dir_symbol);
}

// Method: is_valid_moove
static PyObject* CEngine_is_valid_moove(CEngine* self, PyObject* args) {
    int start_row, start_col, dir_idx, rows, cols;
    if (!PyArg_ParseTuple(args, "iiiii", &start_row, &start_col, &dir_idx, &rows, &cols)) {
        return NULL;
    }

    int dr = DIRECTIONS[dir_idx][0];
    int dc = DIRECTIONS[dir_idx][1];

    // Check all three cells
    for (int i = 0; i < 3; i++) {
        int r = start_row + i * dr;
        int c = start_col + i * dc;
        if (!is_valid_position(r, c, rows, cols)) {
            Py_RETURN_FALSE;
        }
    }

    Py_RETURN_TRUE;
}

// Method: get_moove_cells
static PyObject* CEngine_get_moove_cells_method(CEngine* self, PyObject* args) {
    int start_row, start_col, dir_idx, rows, cols;
    if (!PyArg_ParseTuple(args, "iiiii", &start_row, &start_col, &dir_idx, &rows, &cols)) {
        return NULL;
    }

    return get_moove_cells(start_row, start_col, dir_idx, rows, cols);
}

// Method: generate_all_valid_mooves
static PyObject* CEngine_generate_all_valid_mooves(CEngine* self, PyObject* args) {
    int rows, cols;
    if (!PyArg_ParseTuple(args, "ii", &rows, &cols)) {
        return NULL;
    }

    PyObject* mooves = PyList_New(0);
    if (!mooves) return NULL;

    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            for (int dir_idx = 0; dir_idx < 8; dir_idx++) {
                PyObject* is_valid = CEngine_is_valid_moove(self,
                    Py_BuildValue("iiiii", row, col, dir_idx, rows, cols));

                if (is_valid == Py_True) {
                    PyObject* moove_tuple = Py_BuildValue("(iii)", row, col, dir_idx);
                    if (!moove_tuple) {
                        Py_DECREF(mooves);
                        return NULL;
                    }
                    PyList_Append(mooves, moove_tuple);
                    Py_DECREF(moove_tuple);
                }
                Py_XDECREF(is_valid);
            }
        }
    }

    return mooves;
}

// Method: generate_all_valid_mooves_parallel (same as non-parallel for C)
static PyObject* CEngine_generate_all_valid_mooves_parallel(CEngine* self, PyObject* args) {
    return CEngine_generate_all_valid_mooves(self, args);
}

// Method: get_cells_for_mooves
static PyObject* CEngine_get_cells_for_mooves(CEngine* self, PyObject* args) {
    PyObject* mooves_list;
    int rows, cols;
    if (!PyArg_ParseTuple(args, "Oii", &mooves_list, &rows, &cols)) {
        return NULL;
    }

    if (!PyList_Check(mooves_list)) {
        PyErr_SetString(PyExc_TypeError, "mooves must be a list");
        return NULL;
    }

    Py_ssize_t mooves_len = PyList_Size(mooves_list);
    PyObject* result = PyList_New(mooves_len);
    if (!result) return NULL;

    for (Py_ssize_t i = 0; i < mooves_len; i++) {
        PyObject* moove = PyList_GetItem(mooves_list, i);
        int row, col, dir_idx;

        if (!PyArg_ParseTuple(moove, "iii", &row, &col, &dir_idx)) {
            Py_DECREF(result);
            return NULL;
        }

        PyObject* cells = get_moove_cells(row, col, dir_idx, rows, cols);
        if (!cells) {
            Py_DECREF(result);
            return NULL;
        }

        PyList_SetItem(result, i, cells);
    }

    return result;
}

// Method: build_moove_network
static PyObject* CEngine_build_moove_network(CEngine* self, PyObject* args) {
    PyObject* mooves_list, *cells_list;
    if (!PyArg_ParseTuple(args, "OO", &mooves_list, &cells_list)) {
        return NULL;
    }

    if (!PyList_Check(mooves_list) || !PyList_Check(cells_list)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be lists");
        return NULL;
    }

    Py_ssize_t num_mooves = PyList_Size(mooves_list);
    PyObject* network = PyDict_New();
    if (!network) return NULL;

    for (Py_ssize_t i = 0; i < num_mooves; i++) {
        PyObject* moove_i = PyList_GetItem(mooves_list, i);
        PyObject* cells_i = PyList_GetItem(cells_list, i);
        PyObject* neighbors = PyList_New(0);

        if (!neighbors) {
            Py_DECREF(network);
            return NULL;
        }

        // Convert cells_i to a set for faster lookup
        PyObject* cells_i_set = PySet_New(cells_i);
        if (!cells_i_set) {
            Py_DECREF(neighbors);
            Py_DECREF(network);
            return NULL;
        }

        for (Py_ssize_t j = 0; j < num_mooves; j++) {
            if (i == j) continue;

            PyObject* cells_j = PyList_GetItem(cells_list, j);
            Py_ssize_t cells_j_len = PyList_Size(cells_j);

            // Check for overlap
            bool has_overlap = false;
            for (Py_ssize_t k = 0; k < cells_j_len; k++) {
                PyObject* cell = PyList_GetItem(cells_j, k);
                if (PySet_Contains(cells_i_set, cell) == 1) {
                    has_overlap = true;
                    break;
                }
            }

            if (has_overlap) {
                PyObject* j_obj = PyLong_FromLongLong(j);
                if (!j_obj) {
                    Py_DECREF(cells_i_set);
                    Py_DECREF(neighbors);
                    Py_DECREF(network);
                    return NULL;
                }
                PyList_Append(neighbors, j_obj);
                Py_DECREF(j_obj);
            }
        }

        Py_DECREF(cells_i_set);

        PyObject* i_obj = PyLong_FromLongLong(i);
        if (!i_obj) {
            Py_DECREF(neighbors);
            Py_DECREF(network);
            return NULL;
        }
        PyDict_SetItem(network, i_obj, neighbors);
        Py_DECREF(i_obj);
        Py_DECREF(neighbors);
    }

    return network;
}

// Method: calculate_coverage
static PyObject* CEngine_calculate_coverage(CEngine* self, PyObject* args) {
    PyObject* mooves_list, *cells_list;
    if (!PyArg_ParseTuple(args, "OO", &mooves_list, &cells_list)) {
        return NULL;
    }

    if (!PyList_Check(mooves_list) || !PyList_Check(cells_list)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be lists");
        return NULL;
    }

    PyObject* covered_cells = PySet_New(NULL);
    if (!covered_cells) return NULL;

    Py_ssize_t num_cells = PyList_Size(cells_list);
    for (Py_ssize_t i = 0; i < num_cells; i++) {
        PyObject* cells = PyList_GetItem(cells_list, i);
        if (!cells || cells == Py_None) continue;

        Py_ssize_t cells_len = PyList_Size(cells);
        for (Py_ssize_t j = 0; j < cells_len; j++) {
            PyObject* cell = PyList_GetItem(cells, j);
            PySet_Add(covered_cells, cell);
        }
    }

    Py_ssize_t coverage = PySet_Size(covered_cells);
    Py_DECREF(covered_cells);

    return PyLong_FromLongLong(coverage);
}

// Deallocation
static void CEngine_dealloc(CEngine* self) {
    if (self->name) {
        free(self->name);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

// Initialization
static int CEngine_init(CEngine* self, PyObject* args, PyObject* kwds) {
    self->name = strdup("C");
    return 0;
}

// Method definitions
static PyMethodDef CEngine_methods[] = {
    {"get_name", (PyCFunction)CEngine_get_name, METH_NOARGS, "Get engine name"},
    {"generate_moove_str", (PyCFunction)CEngine_generate_moove_str, METH_VARARGS, "Generate moove string"},
    {"is_valid_moove", (PyCFunction)CEngine_is_valid_moove, METH_VARARGS, "Check if moove is valid"},
    {"get_moove_cells", (PyCFunction)CEngine_get_moove_cells_method, METH_VARARGS, "Get cells for a moove"},
    {"generate_all_valid_mooves", (PyCFunction)CEngine_generate_all_valid_mooves, METH_VARARGS, "Generate all valid mooves"},
    {"generate_all_valid_mooves_parallel", (PyCFunction)CEngine_generate_all_valid_mooves_parallel, METH_VARARGS, "Generate all valid mooves (parallel)"},
    {"get_cells_for_mooves", (PyCFunction)CEngine_get_cells_for_mooves, METH_VARARGS, "Get cells for multiple mooves"},
    {"build_moove_network", (PyCFunction)CEngine_build_moove_network, METH_VARARGS, "Build moove network"},
    {"calculate_coverage", (PyCFunction)CEngine_calculate_coverage, METH_VARARGS, "Calculate coverage"},
    {NULL}
};

// Type definition
static PyTypeObject CEngineType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "c_engine.CEngine",
    .tp_doc = "C implementation of game engine",
    .tp_basicsize = sizeof(CEngine),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)CEngine_init,
    .tp_dealloc = (destructor)CEngine_dealloc,
    .tp_methods = CEngine_methods,
};

// Module definition
static struct PyModuleDef c_engine_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "c_engine",
    .m_doc = "C implementation of moo counter engine",
    .m_size = -1,
};

// Module initialization
PyMODINIT_FUNC PyInit_c_engine(void) {
    PyObject* m;

    if (PyType_Ready(&CEngineType) < 0) {
        return NULL;
    }

    m = PyModule_Create(&c_engine_module);
    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&CEngineType);
    if (PyModule_AddObject(m, "CEngine", (PyObject*)&CEngineType) < 0) {
        Py_DECREF(&CEngineType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}