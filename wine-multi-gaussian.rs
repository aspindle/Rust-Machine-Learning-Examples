use plotters::prelude::*;
use nalgebra::{SMatrix, SVector};

#[derive(Clone)]
struct Wine {
    class: usize,
    alcohol: f64,
    malic_acid: f64,
    ash: f64,
    alcalinity_of_ash: f64,
    magnesium: f64,
    total_phenols: f64,
    flavanoids: f64,
    nonflavanoid_phenols: f64,
    proanthocyanins: f64,
    color_intensity: f64,
    hue: f64,
    od280_od315: f64,
    proline: f64,
}

const NUM_FEATURES: usize = 13;

struct MultivariateGaussian {
    mean: [f64; NUM_FEATURES],
    covariance: [[f64; NUM_FEATURES]; NUM_FEATURES],
}

fn get_features(wine: &Wine) -> [f64; NUM_FEATURES] {
    [
        wine.alcohol,
        wine.malic_acid,
        wine.ash,
        wine.alcalinity_of_ash,
        wine.magnesium,
        wine.total_phenols,
        wine.flavanoids,
        wine.nonflavanoid_phenols,
        wine.proanthocyanins,
        wine.color_intensity,
        wine.hue,
        wine.od280_od315,
        wine.proline,
    ]
}


fn main() {
    let wines = load_wines().unwrap();

    let (training_wines, testing_wines) = split_dataset(wines);

    // Calculate Gaussian distributions for each winery.
    let gaussian1 = calculate_multivariate_gaussian(&training_wines, 1);
    let gaussian2 = calculate_multivariate_gaussian(&training_wines, 2);
    let gaussian3 = calculate_multivariate_gaussian(&training_wines, 3);

    /*
    // Print the means and covariance matrices.
    println!("Winery 1 Mean: {:?}", gaussian1.mean);
    println!("Winery 2 Mean: {:?}", gaussian2.mean);
    println!("Winery 3 Mean: {:?}", gaussian3.mean);

    println!("\nWinery 1 Covariance Matrix:");
    for row in gaussian1.covariance {
        println!("{:?}", row);
    }

    println!("\nWinery 2 Covariance Matrix:");
    for row in gaussian2.covariance {
        println!("{:?}", row);
    }

    println!("\nWinery 3 Covariance Matrix:");
    for row in gaussian3.covariance {
        println!("{:?}", row);
    }*/
    let mut incorrect_predictions = 0;

    // Test every wine.
    for i in 0..testing_wines.len() {

        let actual_class = testing_wines[i].class;

        let predicted_class = predict_wine(
            &testing_wines[i],
            &gaussian1,
            &gaussian2,
            &gaussian3,
            &training_wines,
        );

        if predicted_class != actual_class {
            incorrect_predictions += 1;

            println!(
                "Wine {}: Predicted {}, Actual {}",
                i,
                predicted_class,
                actual_class
            );
        }
    }

    let total = testing_wines.len();

    let error_rate =
        incorrect_predictions as f64 / total as f64 * 100.0;

    let accuracy = 100.0 - error_rate;

    println!("\nClassification Results");
    println!("----------------------");
    println!("Total wines: {}", total);
    println!("Incorrect predictions: {}", incorrect_predictions);
    println!("Error rate: {:.2}%", error_rate);
    println!("Accuracy: {:.2}%", accuracy);


    //EXPECTED OUTPUT
    /*
    Total wines loaded: 178
    Training wines: 130
    Testing wines: 48
    Wine 27: Predicted 1, Actual 2

    Classification Results
    ----------------------
    Total wines: 48
    Incorrect predictions: 1
    Error rate: 2.08%
    Accuracy: 97.92%
    */

}

fn load_wines() -> Result<Vec<Wine>, Box<dyn std::error::Error>> {
    
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("../../wine/wine.data")?;


    let mut wines = Vec::new();

    // Read each row in the CSV
    for result in reader.records() {

        let record = result?;

        let wine = Wine {
            class: record[0].parse()?,
            alcohol: record[1].parse()?,
            malic_acid: record[2].parse()?,
            ash: record[3].parse()?,
            alcalinity_of_ash: record[4].parse()?,
            magnesium: record[5].parse()?,
            total_phenols: record[6].parse()?,
            flavanoids: record[7].parse()?,
            nonflavanoid_phenols: record[8].parse()?,
            proanthocyanins: record[9].parse()?,
            color_intensity: record[10].parse()?,
            hue: record[11].parse()?,
            od280_od315: record[12].parse()?,
            proline: record[13].parse()?,
        };

        wines.push(wine);
    }

    println!("Total wines loaded: {}", wines.len());

    Ok(wines)
}

fn split_dataset(wines: Vec<Wine>) -> (Vec<Wine>, Vec<Wine>) {

    let mut training_wines = Vec::new();
    let mut testing_wines = Vec::new();

    //Since we don't know the exact split used by Professor Dasgupta, 
    //we will just take the non-randomized first n wines for the training
    //and test sets. We know the training set should add up to 130 and 
    //the test set should add up to 48. 
    let training_counts = [43, 51, 36];
    let testing_counts = [16, 20, 12];

    // Separate each winery.
    for class in 1..=3 {

        let mut class_wines = Vec::new();

        for wine in wines.iter() {
            if wine.class == class {
                class_wines.push(wine);
            }
        }

        // Add training wines.
        for i in 0..training_counts[class - 1] {
            training_wines.push(class_wines[i].clone());
        }

        // Add testing wines.
        for i in training_counts[class - 1]..
            training_counts[class - 1] + testing_counts[class - 1] {

            testing_wines.push(class_wines[i].clone());
        }
    }

    println!("Training wines: {}", training_wines.len());
    println!("Testing wines: {}", testing_wines.len());

    (training_wines, testing_wines)
}

fn calculate_multivariate_gaussian(
    wines: &[Wine],
    winery: usize,
) -> MultivariateGaussian {

    let mut count = 0;
    let mut mean = [0.0; NUM_FEATURES];

    // Calculate the mean.
    for wine in wines {
        if wine.class == winery {
            let features = get_features(wine);

            for i in 0..NUM_FEATURES {
                mean[i] += features[i];
            }

            count += 1;
        }
    }

    if count == 0 {
        panic!("No wines found for this winery!");
    }

    for i in 0..NUM_FEATURES {
        mean[i] /= count as f64;
    }

    // Calculate the covariance matrix.
    let mut covariance = [[0.0; NUM_FEATURES]; NUM_FEATURES];

    for wine in wines {
        if wine.class == winery {
            let features = get_features(wine);

            for i in 0..NUM_FEATURES {
                for j in 0..NUM_FEATURES {
                    covariance[i][j] +=
                        (features[i] - mean[i]) *
                        (features[j] - mean[j]);
                }
            }
        }
    }

    for i in 0..NUM_FEATURES {
        for j in 0..NUM_FEATURES {
            covariance[i][j] /= count as f64;
        }
    }

    MultivariateGaussian {
        mean,
        covariance,
    }
}

//apperently we need the log probability, looks too complicated though
fn calculate_log_probability(
    wine: &Wine,
    gaussian: &MultivariateGaussian,
) -> f64 {

    // Convert the wine's features into a vector.
    let features = get_features(wine);
    let x = SVector::<f64, NUM_FEATURES>::from_row_slice(&features);

    // Convert the covariance array into a nalgebra matrix.
    let mut covariance =
        SMatrix::<f64, NUM_FEATURES, NUM_FEATURES>::from_fn(
            |i, j| gaussian.covariance[i][j]
        );

    // Add a small regularization value to the diagonal.
    // This helps if the covariance matrix is nearly singular.
    let mut max_variance: f64 = 0.0;

    for i in 0..NUM_FEATURES {
        if covariance[(i, i)] > max_variance {
            max_variance = covariance[(i, i)];
        }
    }

    let regularization = max_variance * 1e-6;

    for i in 0..NUM_FEATURES {
        covariance[(i, i)] += regularization;
    }

    // Calculate the difference between the wine and the mean.
    let mean = SVector::<f64, NUM_FEATURES>::from_row_slice(
        &gaussian.mean
    );

    let difference = x - mean;

    // Cholesky decomposition.
    let cholesky = covariance.cholesky()
        .expect("Covariance matrix is not positive definite");

    // Solve covariance * y = difference.
    let solved = cholesky.solve(&difference);

    // Calculate the log determinant.
    let lower = cholesky.l();
    let mut log_determinant = 0.0;

    for i in 0..NUM_FEATURES {
        log_determinant += 2.0 * lower[(i, i)].ln();
    }

    // Calculate the quadratic term.
    let quadratic = difference.dot(&solved);

    // Calculate the log of the Gaussian probability density.
    let log_probability = -0.5 * (
        NUM_FEATURES as f64 * (2.0 * std::f64::consts::PI).ln()
        + log_determinant
        + quadratic
    );

    log_probability
}

fn predict_wine(
    wine: &Wine,
    gaussian1: &MultivariateGaussian,
    gaussian2: &MultivariateGaussian,
    gaussian3: &MultivariateGaussian,
    training_wines: &[Wine],
) -> usize {

    let mut class_counts = [0; 3];

    for training_wine in training_wines {
        class_counts[training_wine.class - 1] += 1;
    }

    let total = training_wines.len() as f64;

    let score1 = calculate_log_probability(wine, gaussian1)
        + (class_counts[0] as f64 / total).ln();

    let score2 = calculate_log_probability(wine, gaussian2)
        + (class_counts[1] as f64 / total).ln();

    let score3 = calculate_log_probability(wine, gaussian3)
        + (class_counts[2] as f64 / total).ln();

    if score1 > score2 && score1 > score3 {
        1
    } else if score2 > score1 && score2 > score3 {
        2
    } else {
        3
    }
}