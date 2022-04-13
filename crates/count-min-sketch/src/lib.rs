//! Count-Min Sketch implementation in Rust
//!
//! Based on the paper:
//! <http://dimacs.rutgers.edu/~graham/pubs/papers/cmencyc.pdf

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

// TODO this is probably not what we want to use as the hash function;
// should look at the PgAnyElementHashMap for ideas on how to do
// hashing that cooperates better with Postgres
#[derive(Clone)]
struct HashFunction(DefaultHasher);

impl HashFunction {
    pub fn new() -> Self {
        Self(DefaultHasher::new())
    }

    pub fn hash_into_buckets(&self, item: &str, nbuckets: usize) -> usize {
        let mut hasher = self.0.clone();
        item.hash(&mut hasher);
        let hash_val = hasher.finish();
        (hash_val % (nbuckets as u64)) as usize
    }
}

/// The Count-Min Sketch is a compact summary data structure capable of
/// representing a high-dimensional vector and answering queries on this vector,
/// in particular point queries and dot product queries, with strong accuracy
/// guarantees. Such queries are at the core of many computations, so the
/// structure can be used in order to answer a variety of other queries, such as
/// frequent items (heavy hitters), quantile finding, join size estimation, and
/// more. Since the data structure can easily process updates in the form of
/// additions or subtractions to dimensions of the vector (which may correspond
/// to insertions or deletions, or other transactions), it is capable of working
/// over streams of updates, at high rates.[1]
///
/// [1] <http://dimacs.rutgers.edu/~graham/pubs/papers/cmencyc.pdf>
pub struct CountMinSketch {
    counters: Vec<Vec<isize>>,
    hashfuncs: Vec<HashFunction>,
    width: usize,
    depth: usize,
}

impl CountMinSketch {
    pub fn new(epsilon: f64, delta: f64) -> Self {
        assert!(0.0 < epsilon && epsilon < 1.0);
        assert!(0.0 < delta && delta < 1.0);

        let w = (1f64.exp() / epsilon).ln().ceil() as usize;
        let d = (1f64 / delta).ln().ceil() as usize;
        Self {
            counters: vec![vec![0; w]; d],
            hashfuncs: vec![HashFunction::new(); d],
            width: w,
            depth: d,
        }
    }

    pub fn estimate(&self, item: &str) -> isize {
        let buckets = self
            .hashfuncs
            .iter()
            .map(|h| h.hash_into_buckets(item, self.width));

        self.counters
            .iter()
            .zip(buckets)
            .map(|(counter, bucket)| counter[bucket])
            .min()
            .unwrap()
    }

    pub fn add_value(&mut self, item: &str) {
        for i in 0..self.depth {
            let bucket = self.hashfuncs[i].hash_into_buckets(item, self.width);
            self.counters[i][bucket] += 1;
        }
    }

    pub fn subtract_value(&mut self, item: &str) {
        for i in 0..self.depth {
            let bucket = self.hashfuncs[i].hash_into_buckets(item, self.width);
            self.counters[i][bucket] -= 1;
        }
    }

    // TODO complete once the `HashFunction` struct is settled and
    // we can generate identical sets of hash functions where
    // desirable.

    // pub fn combine(&mut self, sketch: CountMinSketch) {
    //     assert_eq!(self.width, sketch.width);
    //     assert_eq!(self.depth, sketch.depth);
    //     assert_eq!(self.hashfuncs, sketch.hashfuncs);
    //     for (counter1, counter2) in self.counters.iter_mut().zip(sketch.counters) {
    //         for (val1, val2) in counter1.iter_mut().zip(counter2) {
    //             *val1 += val2;
    //         }
    //     }
    // }
}

#[cfg(test)]
mod tests {
    use crate::CountMinSketch;

    #[test]
    fn empty_sketch() {
        let cms = CountMinSketch::new(0.01, 0.01);
        assert_eq!(cms.estimate("foo"), 0);
    }

    #[test]
    fn add_once() {
        let mut cms = CountMinSketch::new(0.01, 0.01);
        cms.add_value("foo");
        assert_eq!(cms.estimate("foo"), 1);
    }

    #[test]
    fn subtract_once() {
        let mut cms = CountMinSketch::new(0.01, 0.01);
        cms.subtract_value("foo");
        assert_eq!(cms.estimate("foo"), -1);
    }

    #[test]
    fn add_repeated() {
        let mut cms = CountMinSketch::new(0.01, 0.01);
        for _ in 0..100_000 {
            cms.add_value("foo")
        }
        assert_eq!(cms.estimate("foo"), 100_000);
    }

    #[test]
    fn add_many_once() {
        let mut cms = CountMinSketch::new(0.01, 0.01);
        cms.add_value("foo");
        cms.add_value("bar");
        cms.add_value("baz");
        assert_eq!(cms.estimate("foo"), 1);
        assert_eq!(cms.estimate("bar"), 1);
        assert_eq!(cms.estimate("baz"), 1);
    }

    #[test]
    fn add_many_repeated() {
        let mut cms = CountMinSketch::new(0.01, 0.01);

        for _ in 0..100_000 {
            cms.add_value("foo")
        }

        for _ in 0..1_000 {
            cms.add_value("bar")
        }

        for _ in 0..1_000_000 {
            cms.add_value("baz")
        }

        let foo_est = cms.estimate("foo");
        let bar_est = cms.estimate("bar");
        let baz_est = cms.estimate("baz");

        // TODO Theoretical guarantee is that these tests pass with only
        // probability 0.99; what to do for the other 1% of the time?
        let err_margin = (0.01 * (100_000f64 + 1_000f64 + 1_000_000f64)) as isize;
        assert!(foo_est.ge(&100_000) && foo_est.le(&(err_margin + 100_000)));
        assert!(bar_est.ge(&1_000) && bar_est.le(&(err_margin + 1_000)));
        assert!(baz_est.ge(&1_000_000) && baz_est.le(&(err_margin + 1_000_000)));
    }

    #[test]
    fn add_and_subtract_many_repeated() {
        use rand::Rng;

        let mut cms = CountMinSketch::new(0.01, 0.01);
        let mut rng = rand::thread_rng();

        let mut foo_count = 0;
        let mut bar_count = 0;
        let mut baz_count = 0;

        for _ in 0..100_000 {
            match rng.gen::<f64>().lt(&0.75) {
                true => {
                    cms.add_value("foo");
                    foo_count += 1;
                }
                false => {
                    cms.subtract_value("foo");
                    foo_count -= 1;
                }
            }
        }

        for _ in 0..1_000 {
            match rng.gen::<f64>().lt(&0.25) {
                true => {
                    cms.add_value("bar");
                    bar_count += 1;
                }
                false => {
                    cms.subtract_value("bar");
                    bar_count -= 1;
                }
            }
        }

        for _ in 0..1_000_000 {
            match rng.gen::<f64>().lt(&0.9) {
                true => {
                    cms.add_value("baz");
                    baz_count += 1;
                }
                false => {
                    cms.subtract_value("baz");
                    baz_count -= 1;
                }
            }
        }

        let foo_est = cms.estimate("foo");
        let bar_est = cms.estimate("bar");
        let baz_est = cms.estimate("baz");

        // TODO Theoretical guarantee is that these tests pass with only
        // probability 0.99; what to do for the other 1% of the time?
        let err_margin = (0.01 * (100_000f64 + 1_000f64 + 1_000_000f64)) as isize;
        assert!(foo_est.ge(&foo_count) && foo_est.le(&(err_margin + foo_count)));
        assert!(bar_est.ge(&bar_count) && bar_est.le(&(err_margin + bar_count)));
        assert!(baz_est.ge(&baz_count) && baz_est.le(&(err_margin + baz_count)));
    }
}
