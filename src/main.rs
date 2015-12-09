#![feature(slice_patterns)]
#![feature(advanced_slice_patterns)]
#![feature(convert)]
extern crate rand;
extern crate time;
extern crate ml_math;

use ml_math::VarianceIncrementor;

use std::fs::File;
use std::io::prelude::*;
use std::str::FromStr;
use std::num::ParseFloatError;
use rand::{Rng, SeedableRng, StdRng};
use std::f64::consts::PI;

const DATA_FILE : &'static str = "pima-indians-diabetes.data";

// Number of comma seperated values contained in each line of the data file
const TOTAL_FIELD_COUNT : usize = 9; 

// Takes a deterministially generated value between 1 and zero and and compares it to this number
// to determine whether the current row should be used for testing or learning
const PERCENT_TEST_ROWS : f64 = 0.30;

/// See the calculateProbability function in [this blog post] for 
///
fn gaussian_probability_density(value: f64, mean: f64, variance: f64) -> f64 {
	let exponent = (-(value - mean).powi(2) / (2f64 * variance)).exp();
	1f64 / ((2f64 * PI).sqrt() * variance.sqrt()) * exponent
}

fn main() {
	
	let start = time::now();
	
	let mut variance_incr = VarianceIncrementor::new();
	
	let mut file = File::open(DATA_FILE).unwrap();
	let mut file_text : String = String::new();
	
	file.read_to_string(&mut file_text).unwrap();

	for i in 0..20 {
		let random_seed = &[i,1];
		let accuracy = predict(random_seed, &file_text);
		variance_incr.add(accuracy);
	}
	
	let duration = time::now() - start;

	println!("Time taken: {}", (duration.num_milliseconds() as f64) / 1000f64);
	println!("Test accurracy summary - mean: {}%, standard_dev: {}", (variance_incr.mean() * 100.0) as u64, variance_incr.variance().sqrt());
}

fn predict(random_seed: &[usize], file_text: &String) -> f64 {
        let mut patients_diseased = [VarianceIncrementor::new(); 8];
        let mut patients_healthy = [VarianceIncrementor::new(); 8];
    
        let mut rng : StdRng = SeedableRng::from_seed(random_seed);

        for line in file_text.lines() {
            let is_test_row = rng.gen_range(0.0, 1.0) <= PERCENT_TEST_ROWS;
            
            if !is_test_row {
                if let Ok(patient_data) = parse_patient_data(&line) {
                    if patient_data.len() != TOTAL_FIELD_COUNT {
//                        println!("Line {} does not have {} columns; skipping", index, TOTAL_FIELD_COUNT);
                    } else {
                        match patient_data.as_slice() {
                            [properties.., has_disease] => {
                                    for (i, property) in properties.iter().enumerate() {
                                            if has_disease == 1f64 {
                                             patients_diseased[i].add(*property);       					
                                            } else {
                                     patients_healthy[i].add(*property);       					
                                            }
                                    }
                            },
                            _ => {} // println!("Line {} is not in the right format", index)
                        }
                    }
                } else {
//                    println!("Line {} contains values that could not be parsed", index);
                }
            }
        } 

        rng.reseed(random_seed);
        
        let mut correct_guesses = 0;
        let mut rows_tested = 0;

        // TODO remove additional File::open
        for line in file_text.lines() {
            let is_test_row = rng.gen_range(0.0, 1.0) <= PERCENT_TEST_ROWS;
            
            if is_test_row {
                let patient_data = parse_patient_data(&line).unwrap();

                if patient_data.len() != TOTAL_FIELD_COUNT {
//                    println!("Line {} does not have {} columns; skipping", index, TOTAL_FIELD_COUNT);
                } else {
                    match patient_data.as_slice() {
                        [properties.., has_disease] => {
                            let mut probability_healthy = 1.0;
                            let mut probability_diseased = 1.0;
                            for (i, property) in properties.iter().enumerate() {
                            	probability_healthy *= gaussian_probability_density(*property, patients_healthy[i].mean(), patients_healthy[i].variance());
                            	probability_diseased *= gaussian_probability_density(*property, patients_diseased[i].mean(), patients_diseased[i].variance());
                            }
                            
                            let healthy_prediction = probability_healthy > probability_diseased;
                            let healthy_actual = has_disease == 0.0;

                            if (healthy_actual && healthy_prediction) || (!healthy_actual && !healthy_prediction) {
                            	correct_guesses += 1;
                            }
                        },
                        _ => {}//println!("Line {} is not in the right format", index)
                    }
                }
                rows_tested += 1;
            }
        } 
		
		let percent_correct_guesses = correct_guesses as f64 / rows_tested as f64;

//		println!("Rows read: {}", rows_read);

//		println!("Rows correctly predicted: {} of {} ({} %)", correct_guesses, rows_tested, percent_correct_guesses);
		
		percent_correct_guesses
}

fn parse_patient_data(patient_data: &str) -> Result<Vec<f64>, ParseFloatError>  {
	let unparsed_properties : Vec<&str> = patient_data.split(",").collect(); 
	unparsed_properties.iter().map(|&patient_data| f64::from_str(patient_data)).collect()
}

#[test]
fn test_parse_patient_data() {
	assert_eq!(vec![6.0, 148.0, 72.0], parse_patient_data("6,148,72").unwrap());
}

#[test]
// Test function using values taken from this [blog post](http://machinelearningmastery.com/naive-bayes-classifier-scratch-python)
fn test_gaussian_probability_density() {
	let sd : f64 = 6.2;
	assert_eq!(0.06248965759370005, gaussian_probability_density(71.5, 73.0, sd.powi(2)));
}