use std::ffi::c_void;

use crate::sparse::{CscMatrix, SparseTriplet};
use suitesparse_sys::umfpack_dl_defaults;
use suitesparse_sys::{
    UMFPACK_A, UMFPACK_CONTROL, UMFPACK_INFO, UMFPACK_OK, umfpack_dl_free_numeric,
    umfpack_dl_free_symbolic, umfpack_dl_numeric, umfpack_dl_report_info, umfpack_dl_report_status,
    umfpack_dl_solve, umfpack_dl_symbolic, umfpack_dl_triplet_to_col,
};
use suitesparse_sys::{
    UMFPACK_ALLOC_INIT, UMFPACK_FIXQ, UMFPACK_IRSTEP, UMFPACK_ORDERING, UMFPACK_ORDERING_CHOLMOD,
    UMFPACK_PIVOT_TOLERANCE, UMFPACK_SCALE, UMFPACK_SCALE_NONE, UMFPACK_STRATEGY,
    UMFPACK_STRATEGY_SYMMETRIC,
};

/// UMFPACK symbolic and numeric factorization data
#[derive(Debug, Default)]
pub struct UmfpackLU {
    pub symbolic: *mut c_void,
    pub numeric: *mut c_void,
    pub numeric_valid: bool,
}

impl UmfpackLU {
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            symbolic: std::ptr::null_mut(),
            numeric: std::ptr::null_mut(),
            numeric_valid: false,
        }
    }

    #[inline(always)]
    pub fn factorize(
        &mut self,
        n: usize,
        csc: &CscMatrix,
        control: &UmfpackControl,
        info: &mut UmfpackInfo,
    ) -> i32 {
        unsafe {
            if !self.symbolic.is_null() {
                umfpack_dl_free_symbolic(&raw mut self.symbolic);
            }

            let status = umfpack_dl_symbolic(
                n as i64,
                n as i64,
                csc.col_ptr.as_ptr(),
                csc.row_ind.as_ptr(),
                csc.values.as_ptr(),
                &raw mut self.symbolic,
                control.inner.as_ptr(),  // Control (use defaults)
                info.inner.as_mut_ptr(), // Info
            );

            if status != UMFPACK_OK as i32 {
                eprintln!("UMFPACK symbolic factorization failed: {status}");
                umfpack_dl_report_info(control.inner.as_ptr(), info.inner.as_ptr());
                umfpack_dl_report_status(control.inner.as_ptr(), status);
                return 1; // Recoverable error
            }
        }

        // Perform sparse LU factorization
        // Numeric factorization
        unsafe {
            if !self.numeric.is_null() {
                umfpack_dl_free_numeric(&raw mut self.numeric);
            }

            self.numeric_valid = false;

            let status = umfpack_dl_numeric(
                csc.col_ptr.as_ptr(),
                csc.row_ind.as_ptr(),
                csc.values.as_ptr(),
                self.symbolic,
                &raw mut self.numeric,
                control.inner.as_ptr(),  // Control
                info.inner.as_mut_ptr(), // Info
            );

            if status != UMFPACK_OK as i32 {
                eprintln!("UMFPACK numeric factorization failed: {status}");
                umfpack_dl_report_info(control.inner.as_ptr(), info.inner.as_ptr());
                umfpack_dl_report_status(control.inner.as_ptr(), status);
                self.numeric = std::ptr::null_mut();
                self.numeric_valid = false;
                return 1; // Recoverable error
            }

            self.numeric_valid = true;
            0
        }
    }

    #[inline(always)]
    pub fn solve(
        &self,
        r: &mut [f64],
        z: &mut [f64],
        csc: &CscMatrix,
        control: &UmfpackControl,
        info: &mut UmfpackInfo,
    ) -> i32 {
        if !self.numeric_valid || self.numeric.is_null() {
            return -1;
        }

        let rptr = r.as_mut_ptr();
        let zptr = z.as_mut_ptr();
        // Solve the system using UMFPACK
        unsafe {
            let status = umfpack_dl_solve(
                UMFPACK_A as i32, // Solve Ax=b (not transposed)
                csc.col_ptr.as_ptr(),
                csc.row_ind.as_ptr(),
                csc.values.as_ptr(),
                zptr, // Solution vector (output)
                rptr, // RHS vector (input)
                self.numeric,
                control.inner.as_ptr(),  // Control
                info.inner.as_mut_ptr(), // Info
            );

            if status != UMFPACK_OK as i32 {
                eprintln!("UMFPACK solve failed: {status}");
                umfpack_dl_report_info(control.inner.as_ptr(), info.inner.as_ptr());
                umfpack_dl_report_status(control.inner.as_ptr(), status);
                return -1;
            }
        }

        0
    }

    #[inline(always)]
    pub fn free_symbolic(&mut self) {
        if !self.symbolic.is_null() {
            unsafe { umfpack_dl_free_symbolic(&raw mut self.symbolic) };
        }
    }

    #[inline(always)]
    pub fn free_numeric(&mut self) {
        if !self.numeric.is_null() {
            unsafe { umfpack_dl_free_numeric(&raw mut self.numeric) };
        }
    }
}

impl Drop for UmfpackLU {
    #[inline(always)]
    fn drop(&mut self) {
        self.free_symbolic();
        self.free_numeric();
    }
}

impl CscMatrix {
    #[inline(always)]
    pub fn assemble(&mut self, n: usize, nnz: usize, triplet: &SparseTriplet) -> i32 {
        let status = unsafe {
            umfpack_dl_triplet_to_col(
                n as i64,
                n as i64,
                nnz as i64,
                triplet.rows.as_ptr(),
                triplet.cols.as_ptr(),
                triplet.vals.as_ptr(),
                self.col_ptr.as_mut_ptr(),
                self.row_ind.as_mut_ptr(),
                self.values.as_mut_ptr(),
                std::ptr::null_mut(), // Map (not needed)
            )
        };

        if status != UMFPACK_OK as i32 {
            eprintln!("UMFPACK triplet to CSC conversion failed: {status}");
            return -1;
        }

        unsafe {
            self.row_ind.set_len(nnz);
            self.values.set_len(nnz);
        }
        0
    }
}

#[derive(Debug)]
pub struct UmfpackControl {
    pub inner: [f64; UMFPACK_CONTROL as usize],
}

impl Default for UmfpackControl {
    fn default() -> Self {
        Self {
            inner: [0.0; UMFPACK_CONTROL as usize],
        }
    }
}

impl UmfpackControl {
    #[must_use]
    #[inline(always)]
    pub fn new() -> Self {
        let mut control = Self::default();
        unsafe {
            umfpack_dl_defaults(control.inner.as_mut_ptr());
        }

        // Robustness tuning for stiff kinetics
        control.inner[UMFPACK_SCALE as usize] = f64::from(UMFPACK_SCALE_NONE);
        control.inner[UMFPACK_PIVOT_TOLERANCE as usize] = 1.0;
        control.inner[UMFPACK_ORDERING as usize] = f64::from(UMFPACK_ORDERING_CHOLMOD);
        control.inner[UMFPACK_IRSTEP as usize] = 5.0;
        control.inner[UMFPACK_ALLOC_INIT as usize] = 0.9;

        // Semenov (2010) used an unsymmetric solver for their factorization
        // needs through a HLSL library. But for us, symmetric seems to work better.
        control.inner[UMFPACK_STRATEGY as usize] = f64::from(UMFPACK_STRATEGY_SYMMETRIC);
        control.inner[UMFPACK_FIXQ as usize] = 1.0;
        control
    }
}

#[derive(Debug)]
pub struct UmfpackInfo {
    pub inner: [f64; UMFPACK_INFO as usize],
}

impl Default for UmfpackInfo {
    fn default() -> Self {
        Self {
            inner: [0.0; UMFPACK_INFO as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_umfpack_lu() {
        let n = 5;
        let mut triplet = SparseTriplet::default();
        // A simple 5x5 matrix
        let entries = [
            (0, 0, 2.),
            (0, 1, 3.),
            (1, 0, 3.),
            (1, 2, 4.),
            (1, 4, 6.),
            (2, 1, -1.),
            (2, 2, -3.),
            (2, 3, 2.),
            (3, 2, 1.),
            (4, 1, 4.),
            (4, 2, 2.),
            (4, 4, 1.),
        ];
        for (r, c, v) in entries {
            triplet.add(r, c, v);
        }

        let mut csc = CscMatrix {
            col_ptr: vec![0; n + 1],
            row_ind: vec![0; entries.len()],
            values: vec![0.0; entries.len()],
        };

        let status = csc.assemble(n, entries.len(), &triplet);
        assert_eq!(status, 0);

        let mut lu = UmfpackLU::new();
        let control = UmfpackControl::new();
        let mut info = UmfpackInfo::default();

        let status = lu.factorize(n, &csc, &control, &mut info);
        assert_eq!(status, 0);

        let b = vec![8., 45., -3., 3., 19.];
        let mut x = vec![0.0; n];
        let mut r = b.clone();

        let status = lu.solve(&mut r, &mut x, &csc, &control, &mut info);
        assert_eq!(status, 0);

        let expected = [1., 2., 3., 4., 5.];
        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-9, "Expected {b}, got {a}");
        }
    }
}
