#ifndef TYPES_COMMON
#define TYPES_COMMON

typedef double real_type;

/*
 * store a matrix is a bipartite graph. One side is the rows and the other is the column.
 */
struct bipartite_graph_descriptor {
  int rows, cols;
  size_t nonzeros;
  bool force_non_square; //do not optimize, so each row and column will get its own graph node, even if the matrix is square

  bipartite_graph_descriptor() : rows(0), cols(0), nonzeros(0), force_non_square(false) { }

   // is the matrix square?
  bool is_square() const { return rows == cols && !force_non_square; }
  // get the position of the starting row/col node
  int get_start_node(bool _rows) const { if (is_square()) return 0; else return (_rows?0:rows); }
  // get the position of the ending row/col node 
  int get_end_node(bool _rows) const { if (is_square()) return rows; else return (_rows?rows:(rows+cols)); }
  // get howmany row/column nodes
  int num_nodes(bool _rows) const { if (_rows) return rows; else return cols; }
  // how many total nodes
  int total() const { if (is_square()) return rows; else return rows+cols; }
  //is this a row node
  bool is_row_node(int id) const { return id < rows; }
  //debug print?
  bool toprint(int id) const { return (id == 0) || (id == rows - 1) || (id == rows) || (id == rows+cols-1); }
  
}; // end of bipartite graph descriptor



#endif
