extern crate core;

mod variance_estimator;

use crate::variance_estimator::VarianceEstimator;
use rand::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use thread_local::ThreadLocal;

// Estimate integral from a to b of f(x) dx
fn monte_carlo_integration(
    f: impl Fn(f64) -> f64 + Sync,
    a: f64,
    b: f64,
    sample_count: usize,
) -> f64 {
    let sum: f64 = (0..sample_count)
        .into_par_iter()
        .map_init(thread_rng, |rng, _| f(rng.gen_range(a..=b)))
        .sum();

    sum * (b - a) / (sample_count as f64)
}

fn test_monte_carlo_integration(
    f: impl Fn(f64) -> f64 + Sync + Copy,
    f_desc: &str,
    a: f64,
    b: f64,
    expected: f64,
) {
    println!("Estimate {f_desc}. Expected result: {expected}");
    for i in 0..8 {
        let sample_count = 2_usize.pow(i);
        let tl = ThreadLocal::new();

        (0..128).into_iter().for_each(|_| {
            let result = monte_carlo_integration(f, a, b, sample_count);

            let cell = tl.get_or(|| RefCell::new(VarianceEstimator::new()));
            cell.borrow_mut().add(result);
        });

        let ve = tl.into_iter().fold(VarianceEstimator::new(), |a, b| {
            VarianceEstimator::merge(&a, &b.borrow())
        });

        println!(
            "sample count: {}, mean of means: {:.2}, variance: {:.1e}",
            sample_count,
            ve.mean,
            ve.variance()
        );
    }
    println!("==========");
}

fn main() {
    use std::f64::consts::{E, PI};
    test_monte_carlo_integration(|x| x * x, "∫ from 0 to 1 of x^2 dx", 0.0, 1.0, 0.33);
    test_monte_carlo_integration(|x| x.sin(), "∫ from 0 to PI of sin(x) dx", 0.0, PI, 2.0);
    test_monte_carlo_integration(|x| x.cos(), "∫ from 0 to PI of cos(x) dx", 0.0, PI, 0.0);

    test_monte_carlo_integration(
        |x| 2.0 / PI.sqrt() * E.powf(-x * x),
        "Error Function erf(1)",
        0.0,
        1.0,
        0.84,
    );
}
