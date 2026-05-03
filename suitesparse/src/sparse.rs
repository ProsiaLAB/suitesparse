#[derive(Debug, Default, Clone)]
pub struct CscMatrix {
    pub col_ptr: Vec<i64>,
    pub row_ind: Vec<i64>,
    pub values: Vec<f64>,
}

impl CscMatrix {
    #[must_use]
    pub fn new(n: usize, nnz: usize) -> Self {
        Self {
            col_ptr: vec![0; n + 1],
            row_ind: vec![0; nnz],
            values: vec![0.0; nnz],
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.col_ptr.clear();
        self.row_ind.clear();
        self.values.clear();
    }
}

#[derive(Debug, Default, Clone)]
pub struct SparseTriplet {
    pub rows: Vec<i64>,
    pub cols: Vec<i64>,
    pub vals: Vec<f64>,
}

impl SparseTriplet {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.rows.clear();
        self.cols.clear();
        self.vals.clear();
    }

    #[inline(always)]
    pub fn add(&mut self, row: usize, col: usize, val: f64) {
        self.rows.push(row as i64);
        self.cols.push(col as i64);
        self.vals.push(val);
    }
}
