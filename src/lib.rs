use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::PyArray2;
use std::f64::consts::PI;
use ndarray::Array2;
use physical_constants;

/// Solves the RCSJ overdamped model using the Runge-Kutta 4th order method.
#[pyfunction]
fn rcsj_solver_rk4_grid(
    py: Python<'_>,
    Ic: f64,
    Idc: Vec<f64>,
    Iac: Vec<f64>,
    freq: f64,
    R: f64,
    dt: f64,
    t_max: f64,
) -> (Py<PyArray2<f64>>, Py<PyArray2<f64>>) {
    let n_idc = Idc.len();
    let n_iac = Iac.len();
    let Iac: Vec<f64> = Iac.iter().map(|&x| x / Ic).collect();
    let Idc: Vec<f64> = Idc.iter().map(|&x| x / Ic).collect();
    let mut voltages = Array2::<f64>::zeros((n_idc, n_iac));
    let mut powers = Array2::<f64>::zeros((n_idc, n_iac));
    let hbar = physical_constants::REDUCED_PLANCK_CONSTANT;
    let e = physical_constants::ELEMENTARY_CHARGE;
    let norm = 2.0 * e * R * Ic / hbar;
    let freq = freq * 2.0 * PI / norm;
    let t_max = norm * t_max;
    let dt = norm * dt;

    for (j, iac) in Iac.iter().enumerate() {
        for (i, idc) in Idc.iter().enumerate() {
            let (V, P) = rcsj_solver_rk4(idc, iac, &freq, &dt, &t_max);
            voltages[[i, j]] = V * R * Ic;
            powers[[i, j]] = P * R * Ic * Ic;
        }
    }
    (PyArray2::from_owned_array(py, voltages).to_owned().into(), PyArray2::from_owned_array(py, powers).to_owned().into())
}

fn rcsj_solver_rk4(Idc: &f64, Iac: &f64, freq: &f64, dt: &f64, t_max: &f64) -> (f64, f64) {
    let mut t = 0.0;
    let mut n: u64 = 0;
    let mut phi = 0.0;
    let mut V = 0.0;
    let mut P = 0.0;

    // Differential equation that we want to solve.
    let dphi_dt = |phi: f64, t: f64| -> f64 {
        Idc + Iac * (freq * t).sin() - phi.sin()
    };

    while t < *t_max {
        let k1 = dphi_dt(phi, t) * dt;
        let k2 = dphi_dt(phi + 0.5 * k1, t + 0.5 * dt) * dt;
        let k3 = dphi_dt(phi + 0.5 * k2, t + 0.5 * dt) * dt;
        let k4 = dphi_dt(phi + k3, t + dt) * dt;
        let dphi = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        phi += dphi ;
        t += dt;
        if phi > 2.0 * PI {
            phi -= 2.0 * PI;
        }
        if phi < 0.0 {
            phi += 2.0 * PI;
        }
        if t * freq / 2.0 / PI > 15.0 { // Skip the first 15 cycles for phi to stabilize
            n += 1;
            V += dphi / dt; // Voltage is the time derivative of phase
            P += dphi / dt * (Idc + Iac * (freq * (t - dt/2.0)).sin());
        }
    }
    (V / (n as f64), P / (n as f64))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rcsj_solver(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rcsj_solver_rk4_grid, m)?)?;
    Ok(())
}