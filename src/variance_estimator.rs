// Variance estimator that uses the Welford’s algorithm
// Code adapted from https://pbr-book.org/4ed/Utilities/Mathematical_Infrastructure#RobustVarianceEstimation
#[derive(Copy, Clone, Debug, Default)]
pub struct VarianceEstimator {
    pub mean: f64,
    sum_square_differences: f64,
    sample_count: i64,
}

impl VarianceEstimator {
    pub fn new() -> Self {
        Self {
            mean: 0.0,
            sum_square_differences: 0.0,
            sample_count: 0,
        }
    }

    pub fn add_sample(&mut self, x: f64) {
        self.sample_count += 1;
        let delta = x - self.mean;
        self.mean += delta / (self.sample_count as f64);
        let delta2 = x - self.mean;
        self.sum_square_differences += delta * delta2;
    }

    pub fn variance(&self) -> f64 {
        if self.sample_count > 1 {
            self.sum_square_differences / (self.sample_count - 1) as f64
        } else {
            0.0
        }
    }

    #[allow(dead_code)]
    pub fn relative_variance(&self) -> f64 {
        if self.sample_count < 1 || self.mean == 0.0 {
            0.0
        } else {
            self.variance() / self.mean
        }
    }

    pub fn merge(lhs: Self, rhs: Self) -> Self {
        if rhs.sample_count == 0 {
            return lhs;
        }

        let left_sample_count_f64 = lhs.sample_count as f64;
        let right_sample_count_f64 = rhs.sample_count as f64;
        let sample_count = lhs.sample_count + rhs.sample_count;

        let sqr_mean_diff = (rhs.mean - lhs.mean) * (rhs.mean - lhs.mean);
        let sum_square_differences = lhs.sum_square_differences
            + rhs.sum_square_differences
            + sqr_mean_diff * left_sample_count_f64 * right_sample_count_f64
                / (sample_count as f64);
        let mean = (left_sample_count_f64 * lhs.mean + right_sample_count_f64 * rhs.mean)
            / (sample_count as f64);

        Self {
            mean,
            sum_square_differences,
            sample_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx_eq::assert_approx_eq;
    use rayon::prelude::*;

    #[test]
    fn test_all_same() {
        const CONSTANT: f64 = 42.0;
        let mut ve = VarianceEstimator::new();
        (0..100).for_each(|_| ve.add_sample(CONSTANT));
        assert_eq!(ve.mean, CONSTANT);
        assert_eq!(ve.variance(), 0.0);
        assert_eq!(ve.relative_variance(), 0.0);
    }

    #[test]
    fn test_range() {
        let mut ve = VarianceEstimator::new();
        // An integer sequence from 0 to 100 has an variance around 841.67
        (0..100).for_each(|i| ve.add_sample(i as f64));

        assert_eq!(ve.mean, 49.5);
        assert_approx_eq!(ve.variance(), 841.67, 0.01);
        assert_approx_eq!(ve.relative_variance(), 841.67 / ve.mean, 0.01);
    }

    #[test]
    fn test_merge() {
        let mut ve1 = VarianceEstimator::new();
        let mut ve2 = VarianceEstimator::new();

        // An integer sequence from 0 to 200 has an variance around 3350
        (0..100).for_each(|i| ve1.add_sample(i as f64));
        (100..200).for_each(|i| ve2.add_sample(i as f64));

        let ve = VarianceEstimator::merge(ve1, ve2);

        assert_eq!(ve.mean, 99.5);
        assert_approx_eq!(ve.variance(), 3350.0, 0.01);
        assert_approx_eq!(ve.relative_variance(), 3350.0 / ve.mean, 0.01);
    }

    #[test]
    fn test_concurrent_accumulate() {
        // An integer sequence from 0 to 10000 has an variance around 8334166.67
        let ve = (0..10000)
            .into_par_iter()
            .fold(VarianceEstimator::new, |mut ve, i| {
                ve.add_sample(i as f64);
                ve
            })
            .reduce(VarianceEstimator::new, VarianceEstimator::merge);

        assert_approx_eq!(ve.variance(), 8334166.67, 0.01);
    }
}
