use plotters::prelude::*;

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

struct BivariateGaussian {
    mean_alcohol: f64,
    mean_flavanoids: f64,

    covariance: [[f64; 2]; 2],
}


fn main() {
    let wines = load_wines().unwrap();

    let (training_wines, _testing_wines) = split_dataset(wines);

    plot_wines(&training_wines).unwrap();


    let gaussian = calculate_bivariate_gaussian(&training_wines, 1);

    println!("Winery 1:");
    println!("Mean Alcohol: {}", gaussian.mean_alcohol);
    println!("Mean Flavanoids: {}", gaussian.mean_flavanoids);

    println!("Covariance Matrix:");

    for row in gaussian.covariance {
        println!("{:?}", row);
    }

    let gaussian = calculate_bivariate_gaussian(&training_wines, 2);

    println!("Winery 2:");
    println!("Mean Alcohol: {}", gaussian.mean_alcohol);
    println!("Mean Flavanoids: {}", gaussian.mean_flavanoids);

    println!("Covariance Matrix:");

    for row in gaussian.covariance {
        println!("{:?}", row);
    }

    let gaussian = calculate_bivariate_gaussian(&training_wines, 3);

    println!("Winery 3:");
    println!("Mean Alcohol: {}", gaussian.mean_alcohol);
    println!("Mean Flavanoids: {}", gaussian.mean_flavanoids);

    println!("Covariance Matrix:");

    for row in gaussian.covariance {
        println!("{:?}", row);
    }

   

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


fn plot_wines(wines: &[Wine]) -> Result<(), Box<dyn std::error::Error>> {

    // Find the minimum and maximum values.
    let mut min_alcohol = f64::INFINITY;
    let mut max_alcohol = f64::NEG_INFINITY;

    let mut min_flavanoids = f64::INFINITY;
    let mut max_flavanoids = f64::NEG_INFINITY;

    for wine in wines {
        if wine.alcohol < min_alcohol {
            min_alcohol = wine.alcohol;
        }

        if wine.alcohol > max_alcohol {
            max_alcohol = wine.alcohol;
        }

        if wine.flavanoids < min_flavanoids {
            min_flavanoids = wine.flavanoids;
        }

        if wine.flavanoids > max_flavanoids {
            max_flavanoids = wine.flavanoids;
        }
    }

    // Create drawing area.
    let root = BitMapBackend::new("wine_scatter.png", (900, 700))
        .into_drawing_area();

    root.fill(&WHITE)?;

    // Create chart.
    let mut chart = ChartBuilder::on(&root)
        .caption("Wine Training Dataset - Alcohol vs Flavanoids", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(
            (min_alcohol - 0.5)..(max_alcohol + 0.5),
            (min_flavanoids - 0.5)..(max_flavanoids + 0.5),
        )?;

    chart.configure_mesh()
        .x_desc("Alcohol")
        .y_desc("Flavanoids")
        .draw()?;

    // Plot each winery using a different color.
    for wine in wines {

        let color = match wine.class {
            1 => RED,
            2 => BLACK,
            3 => GREEN,
            _ => BLUE,
        };

        chart.draw_series(std::iter::once(
            Circle::new(
                (wine.alcohol, wine.flavanoids),
                5,
                color.filled(),
            )
        ))?;
    }

     // Calculate the Gaussian distributions.
    let gaussian1 = calculate_bivariate_gaussian(wines, 1);
    let gaussian2 = calculate_bivariate_gaussian(wines, 2);
    let gaussian3 = calculate_bivariate_gaussian(wines, 3);

    // Generate ellipse coordinates.
    let ellipse1 = gaussian_ellipse_points(&gaussian1, 2.447);
    let ellipse2 = gaussian_ellipse_points(&gaussian2, 2.447);
    let ellipse3 = gaussian_ellipse_points(&gaussian3, 2.447);

    // Draw the ellipses.
    chart.draw_series(LineSeries::new(ellipse1, &RED))?;
    chart.draw_series(LineSeries::new(ellipse2, &BLACK))?;
    chart.draw_series(LineSeries::new(ellipse3, &GREEN))?;

    root.present()?;

    println!("Scatter plot saved to wine_scatter.png");

    Ok(())
}

fn calculate_bivariate_gaussian(
    wines: &[Wine],
    winery: usize
) -> BivariateGaussian {

    let mut alcohol_values = Vec::new();
    let mut flavanoid_values = Vec::new();

    // Collect measurements for the selected winery.
    for wine in wines {
        if wine.class == winery {
            alcohol_values.push(wine.alcohol);
            flavanoid_values.push(wine.flavanoids);
        }
    }

    let count = alcohol_values.len() as f64;

    if count == 0.0 {
        panic!("No wines found for this winery!");
    }

    // Calculate means.
    let mut sum_alcohol = 0.0;
    let mut sum_flavanoids = 0.0;

    for i in 0..alcohol_values.len() {
        sum_alcohol += alcohol_values[i];
        sum_flavanoids += flavanoid_values[i];
    }

    let mean_alcohol = sum_alcohol / count;
    let mean_flavanoids = sum_flavanoids / count;

    // Calculate variances and covariance.
    let mut variance_alcohol = 0.0;
    let mut variance_flavanoids = 0.0;
    let mut covariance = 0.0;

    for i in 0..alcohol_values.len() {

        let diff_alcohol = alcohol_values[i] - mean_alcohol;
        let diff_flavanoids = flavanoid_values[i] - mean_flavanoids;

        variance_alcohol += diff_alcohol.powi(2);
        variance_flavanoids += diff_flavanoids.powi(2);

        covariance += diff_alcohol * diff_flavanoids;
    }

    variance_alcohol /= count;
    variance_flavanoids /= count;
    covariance /= count;

    // Construct covariance matrix.
    let covariance_matrix = [
        [variance_alcohol, covariance],
        [covariance, variance_flavanoids],
    ];

    BivariateGaussian {
        mean_alcohol,
        mean_flavanoids,
        covariance: covariance_matrix,
    }
}

fn gaussian_ellipse_points(
    gaussian: &BivariateGaussian,
    radius: f64,
) -> Vec<(f64, f64)> {

    let a = gaussian.covariance[0][0];
    let b = gaussian.covariance[0][1];
    let d = gaussian.covariance[1][1];

    // Calculate the two eigenvalues.
    let difference = ((a - d).powi(2) + 4.0 * b.powi(2)).sqrt();

    let lambda1 = (a + d + difference) / 2.0;
    let lambda2 = (a + d - difference) / 2.0;

    // Angle of the ellipse.
    let angle = 0.5 * (2.0 * b).atan2(a - d);

    let mut points = Vec::new();

    // Generate 101 points around the ellipse.
    for i in 0..=100 {
        let theta = 2.0 * std::f64::consts::PI * i as f64 / 100.0;

        let x = radius * lambda1.sqrt() * theta.cos();
        let y = radius * lambda2.sqrt() * theta.sin();

        // Rotate and move the point to the Gaussian's center.
        let rotated_x = x * angle.cos() - y * angle.sin();
        let rotated_y = x * angle.sin() + y * angle.cos();

        let final_x = gaussian.mean_alcohol + rotated_x;
        let final_y = gaussian.mean_flavanoids + rotated_y;

        points.push((final_x, final_y));
    }

    points
}