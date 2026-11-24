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



fn main() -> Result<(), Box<dyn std::error::Error>> {
    let wines = load_wines().unwrap();

    let (training_wines, testing_wines) = split_dataset(wines);

    //plot_wines(&training_wines).unwrap();

    
    let gaussian1 = calculate_bivariate_gaussian(&training_wines, 1);
    let gaussian2 = calculate_bivariate_gaussian(&training_wines, 2);
    let gaussian3 = calculate_bivariate_gaussian(&training_wines, 3);

    let (prior1, prior2, prior3) = calculate_priors(&training_wines);

    plot_wines(
        &training_wines,
        &gaussian1,
        &gaussian2,
        &gaussian3,
        prior1,
        prior2,
        prior3,
    )?;

    Ok(())

}

fn calculate_priors(wines: &[Wine]) -> (f64, f64, f64) {
    let total = wines.len() as f64;

    let mut count1 = 0;
    let mut count2 = 0;
    let mut count3 = 0;

    for wine in wines {
        if wine.class == 1 {
            count1 += 1;
        } else if wine.class == 2 {
            count2 += 1;
        } else if wine.class == 3 {
            count3 += 1;
        }
    }

    let prior1 = count1 as f64 / total;
    let prior2 = count2 as f64 / total;
    let prior3 = count3 as f64 / total;

    (prior1, prior2, prior3)
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


fn plot_wines(
    wines: &[Wine],
    gaussian1: &BivariateGaussian,
    gaussian2: &BivariateGaussian,
    gaussian3: &BivariateGaussian,
    prior1: f64,
    prior2: f64,
    prior3: f64,
) -> Result<(), Box<dyn std::error::Error>> {

    let root = BitMapBackend::new("wine.png", (1000, 800))
        .into_drawing_area();

    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Wine Classification - Decision Boundary", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(11.0..15.0, 0.0..6.0)?;

    chart.configure_mesh()
        .x_desc("Alcohol")
        .y_desc("Flavanoids")
        .draw()?;

    // 1. Draw wine data points
    for wine in wines {
        let color = match wine.class {
            1 => &RED,
            2 => &BLACK,
            3 => &GREEN,
            _ => &BLUE,
        };

        chart.draw_series(
            std::iter::once(Circle::new(
                (wine.alcohol, wine.flavanoids),
                4,
                color.filled(),
            ))
        )?;
    }

    // 2. Draw Gaussian ellipses
    let ellipse1 = gaussian_ellipse_points(&gaussian1, 2.447);
    let ellipse2 = gaussian_ellipse_points(&gaussian2, 2.447);
    let ellipse3 = gaussian_ellipse_points(&gaussian3, 2.447);

    //chart.draw_series(LineSeries::new(ellipse1, &RED))?;
    //chart.draw_series(LineSeries::new(ellipse2, &BLACK))?;
    //chart.draw_series(LineSeries::new(ellipse3, &GREEN))?;

     /*Get decision boundary is the meat of this example 
       
     */
    // Class 1 vs Class 2, checking against Class 3
    let boundary12 = get_decision_boundary(
        &gaussian1, &gaussian2, &gaussian3,
        prior1, prior2, prior3,
        11.0, 15.0, 0.0, 6.0,
    );

    // Class 1 vs Class 3, checking against Class 2
    let boundary13 = get_decision_boundary(
        &gaussian1, &gaussian3, &gaussian2,
        prior1, prior3, prior2,
        11.0, 15.0, 0.0, 6.0,
    );

    // Class 2 vs Class 3, checking against Class 1
    let boundary23 = get_decision_boundary(
        &gaussian2, &gaussian3, &gaussian1,
        prior2, prior3, prior1,
        11.0, 15.0, 0.0, 6.0,
    );


    chart.draw_series(boundary12.iter().map(|s| {
        PathElement::new(vec![s[0], s[1]], MAGENTA)
    }))?;

    chart.draw_series(boundary13.iter().map(|s| {
        PathElement::new(vec![s[0], s[1]], MAGENTA)
    }))?;

    chart.draw_series(boundary23.iter().map(|s| {
        PathElement::new(vec![s[0], s[1]], MAGENTA)
    }))?;

    


    root.present()?;

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

fn calculate_bivariate_probability(
    alcohol: f64,
    flavanoids: f64,
    gaussian: &BivariateGaussian,
) -> f64 {

    let dx = alcohol - gaussian.mean_alcohol;
    let dy = flavanoids - gaussian.mean_flavanoids;

    let a = gaussian.covariance[0][0];
    let b = gaussian.covariance[0][1];
    let d = gaussian.covariance[1][1];

    let determinant = a * d - b * b;

    let quadratic = (
        d * dx * dx
        - 2.0 * b * dx * dy
        + a * dy * dy
    ) / determinant;

    let probability =
        (1.0 / (2.0 * std::f64::consts::PI * determinant.sqrt()))
        * (-0.5 * quadratic).exp();

    probability
}

fn decision_boundary_value(
    alcohol: f64,
    flavanoids: f64,
    gaussian1: &BivariateGaussian,
    gaussian2: &BivariateGaussian,
    prior1: f64,
    prior2: f64,
) -> f64 {

    let probability1 =
        calculate_bivariate_probability(alcohol, flavanoids, gaussian1);

    let probability2 =
        calculate_bivariate_probability(alcohol, flavanoids, gaussian2);

    probability1 * prior1 - probability2 * prior2
}

fn interpolate_zero(
    p1: (f64, f64),
    p2: (f64, f64),
    v1: f64,
    v2: f64,
) -> (f64, f64) {

    let t = v1 / (v1 - v2);

    (
        p1.0 + t * (p2.0 - p1.0),
        p1.1 + t * (p2.1 - p1.1),
    )
}

//Vec<[(f64, f64); 2]> represents many line segments

fn get_decision_boundary(
    gaussian1: &BivariateGaussian,
    gaussian2: &BivariateGaussian,
    gaussian3: &BivariateGaussian,
    prior1: f64,
    prior2: f64,
    prior3: f64,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Vec<[(f64, f64); 2]> {

    let mut boundary = Vec::new();
    //let mut segments = Vec::new();

    let steps = 150;

    let dx = (x_max - x_min) / steps as f64;
    let dy = (y_max - y_min) / steps as f64;

    for i in 0..steps {
        for j in 0..steps {

            let x = x_min + i as f64 * dx;
            let y = y_min + j as f64 * dy;

            // Four corners of the square
            let p = [
                (x, y),
                (x + dx, y),
                (x + dx, y + dy),
                (x, y + dy),
            ];

            // Calculate function at each corner
            let mut v = [0.0; 4];

            for k in 0..4 {
                v[k] = decision_boundary_value(
                    p[k].0,
                    p[k].1,
                    gaussian1,
                    gaussian2,
                    prior1,
                    prior2,
                );
            }

            // Check the four edges
            let edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
            ];

            let mut intersections = Vec::new();

            for (a, b) in edges {

                if (v[a] < 0.0 && v[b] >= 0.0)
                    || (v[a] >= 0.0 && v[b] < 0.0)
                {
                    let point = interpolate_zero(
                        p[a],
                        p[b],
                        v[a],
                        v[b],
                    );

                    intersections.push(point);
                }
            }

            // Connect the two intersection points
            if intersections.len() == 2 {
                let p1 = intersections[0];
                let p2 = intersections[1];

                // Find the midpoint of the boundary segment
                let midpoint_x = (p1.0 + p2.0) / 2.0;
                let midpoint_y = (p1.1 + p2.1) / 2.0;

                // Calculate the scores of all three classes
                let score1 = calculate_score(
                    midpoint_x, midpoint_y, gaussian1, prior1
                );

                let score2 = calculate_score(
                    midpoint_x, midpoint_y, gaussian2, prior2
                );

                let score3 = calculate_score(
                    midpoint_x, midpoint_y, gaussian3, prior3
                );

                // Keep the segment only if the two competing
                // classes outperform the third class.
                let competing_score = (score1 + score2) / 2.0;

                if competing_score >= score3 {
                    boundary.push([p1, p2]);
                }
            }
        }
    }

    boundary
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

fn calculate_score(
    alcohol: f64,
    flavanoids: f64,
    gaussian: &BivariateGaussian,
    prior: f64,
) -> f64 {
    calculate_bivariate_probability(alcohol, flavanoids, gaussian) * prior
}