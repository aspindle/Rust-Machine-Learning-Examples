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

struct Gaussian {
    mean: f64,
    variance: f64,
}



fn main() {
    let wines = load_wines().unwrap();

    let (training_wines, _testing_wines) = split_dataset(wines);

    plot_alcohol_histogram(&training_wines).unwrap();

    let (mut mean, mut variance) = calculate_alcohol_stats(&training_wines, 1);

    println!("Winery 1 Alcohol Statistics:");
    println!("Mean: {}", mean);
    println!("Variance: {}", variance);

    (mean, variance) = calculate_alcohol_stats(&training_wines, 2);

    println!("Winery 2 Alcohol Statistics:");
    println!("Mean: {}", mean);
    println!("Variance: {}", variance);

    (mean, variance) = calculate_alcohol_stats(&training_wines, 3);

    println!("Winery 3 Alcohol Statistics:");
    println!("Mean: {}", mean);
    println!("Variance: {}", variance);
}

fn load_wines() -> Result<Vec<Wine>, Box<dyn std::error::Error>> {
    
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("../../wine/wine.data")?;

    /*let mut reader = csv::Reader::from_path(
        "../../wine/wine.data"
    )?;*/

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

fn plot_alcohol_histogram(wines: &[Wine]) -> Result<(), Box<dyn std::error::Error>> {

    // Collect alcohol measurements for Winery 1.
    let mut alcohol_values = Vec::new();

    for wine in wines {
        if wine.class == 1 {
            alcohol_values.push(wine.alcohol);
        }
    }

    if alcohol_values.is_empty() {
        println!("No wines found for Winery 1.");
        return Ok(());
    }

    // Find minimum and maximum alcohol values.
    let min = alcohol_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = alcohol_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Configure histogram.
    let num_bins = 10;
    let bin_width = (max - min) / num_bins as f64;

    let mut bins = vec![0.0f64; num_bins];

    // Count wines in each bin.
    for alcohol in &alcohol_values {
        let mut index = ((*alcohol - min) / bin_width) as usize;

        if index >= num_bins {
            index = num_bins - 1;
        }

        bins[index] += 1.0;
    }

    // NORMALIZE BINS INTO DENSITY
    let total_wines = alcohol_values.len() as f64;

    for i in 0..num_bins {
        bins[i] = bins[i] / (total_wines * bin_width);
    }

    // Create drawing area.
    let root = BitMapBackend::new("winery1_alcohol.png", (900, 600))
        .into_drawing_area();

    root.fill(&WHITE)?;

    // Find maximum bin height.
    let max_density = bins.iter().cloned().fold(0.0, f64::max);

    // Create chart.
    let mut chart = ChartBuilder::on(&root)
        .caption("Winery 1 - Alcohol Distribution", ("sans-serif", 30))
        .margin(30)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(min..max, 0.0..(max_density * 1.2))?;

    chart.configure_mesh()
        .x_desc("Alcohol")
        .y_desc("Density")
        .draw()?;

    // Draw histogram bars.
    for i in 0..num_bins {

        let x0 = min + i as f64 * bin_width;
        let x1 = x0 + bin_width;

        chart.draw_series(std::iter::once(
            Rectangle::new(
                [(x0, 0.0), (x1, bins[i])],
                BLUE.mix(0.6).filled(),
            )
        ))?;
    }

    root.present()?;

    println!("Histogram saved to winery1_alcohol.png");

    Ok(())
}

fn calculate_alcohol_stats(wines: &[Wine], winery: usize) -> (f64, f64) {

    let mut alcohol_values = Vec::new();

    // Collect alcohol measurements for the selected winery.
    for wine in wines {
        if wine.class == winery {
            alcohol_values.push(wine.alcohol);
        }
    }

    let count = alcohol_values.len() as f64;

    if count == 0.0 {
        panic!("No wines found for this winery!");
    }

    // Calculate mean.
    let mut sum = 0.0;

    for alcohol in &alcohol_values {
        sum += alcohol;
    }

    let mean = sum / count;

    // Calculate variance.
    let mut squared_differences = 0.0;

    for alcohol in &alcohol_values {
        squared_differences += (alcohol - mean).powi(2);
    }

    let variance = squared_differences / count;

    (mean, variance)
}