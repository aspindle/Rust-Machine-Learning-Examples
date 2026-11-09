use flate2::read::GzDecoder;
use minifb::{Key, Window, WindowOptions};
use std::fs::File;
use std::io::Read;

fn main() {

    // k-NN classification using Euclidean distance on MNIST dataset

    // Load MNIST dataset
    let train_images = load_train_images();
    let train_labels = load_train_labels();

    let test_images = load_test_images();
    let test_labels = load_test_labels();

    println!("Training images: {}", train_images.len());
    println!("Training labels: {}", train_labels.len());

    println!("Test images: {}", test_images.len());
    println!("Test labels: {}", test_labels.len());
    
    /* Compare images */
    let image_index_a = 4;
    let image_index_b = 5;

    let distance = find_euclidean_dist(
        &train_images[image_index_a],
        &train_images[image_index_b]
    );

    println!("Euclidean distance between image 4 and image 5: {}", distance);

    // Display image 3 at a scale of 20
    // display_image(&train_images[3], 20);
    
    let test_image_index = 100;
    println!("Displaying test image {} at a scale of 20", test_image_index);
    display_image(&test_images[test_image_index], 20);
    

    let (distance, nearest_index) = find_nearest_neighbor(
        &train_images,
        &test_images[test_image_index]
    );

    println!("Shortest distance: {}", distance);
    println!("Nearest image index: {}", nearest_index);
    
    println!("Displaying nearest training image {} at a scale of 20", nearest_index);
    display_image(&train_images[nearest_index], 20);
    println!("Predicted digit: {}", train_labels[nearest_index]);
    println!("Actual digit: {}", test_labels[test_image_index]);

    /* now lets evaluate the model on the entire test set */
    
    let mut incorrect_predictions = 0;

    let k = 3;
    
    for i in 0..test_images.len() {
    //for i in 0..100 {

        // Find the K nearest training images
        let neighbors = find_k_nearest_neighbors(
            &train_images,
            &test_images[i],
            k
        );

        // Predict using majority voting
        let predicted_digit = predict_digit(
            &neighbors,
            &train_labels
        );

        // Get actual label
        let actual_digit = test_labels[i];

        // Check prediction
        if predicted_digit != actual_digit {
            incorrect_predictions += 1;

            println!(
                "Test image {}: predicted {}, actual {}",
                i, predicted_digit, actual_digit
            );
        }
    }

    // Calculate error rate and accuracy
    let total_test_images = test_images.len();

    let error_rate =
        (incorrect_predictions as f64 / total_test_images as f64) * 100.0;

    let accuracy = 100.0 - error_rate;

    println!("\n--- k-nn MNIST Results ---");
    println!("Total test images: {}", total_test_images);
    println!("Incorrect predictions: {}", incorrect_predictions);
    println!("Correct predictions: {}", total_test_images - incorrect_predictions);
    println!("Error rate: {:.2}%", error_rate);
    println!("Accuracy: {:.2}%", accuracy);


    /* Expected output:
    --- k-nn MNIST Results (k=3) ---
    Total test images: 10000
    Incorrect predictions: 295
    Correct predictions: 9705
    Error rate: 2.95%
    Accuracy: 97.05%
    */

    /* In his presentation, Prof. Sanjoy Dasgupta used k=3 and got 294 incorrect predictions,
       which is very close to our result of 295. We could experiment with other values of k, 
       but I will leave that as an exercise for the reader. */
    
}


fn find_euclidean_dist(image_a: &[u8], image_b: &[u8]) -> f64 {
    let mut distance = 0.0;

    for i in 0..image_a.len() {
        let difference = image_a[i] as f64 - image_b[i] as f64;
        distance += difference.powi(2);
    }

    distance.sqrt()
}

fn find_nearest_neighbor(
    training_images: &[Vec<u8>],
    single_test_image: &[u8]
) -> (f64, usize) {

    let mut shortest_distance_value = f64::INFINITY;
    let mut shortest_distance_index = 0;

    for i in 0..training_images.len() {

        let train_img = &training_images[i];

        let new_distance = find_euclidean_dist(
            train_img,
            single_test_image
        );

        if new_distance < shortest_distance_value {
            shortest_distance_value = new_distance;
            shortest_distance_index = i;
        }
    }

    (shortest_distance_value, shortest_distance_index)
}

fn find_k_nearest_neighbors(
    train_images: &Vec<Vec<u8>>,
    test_image: &Vec<u8>,
    k: usize,
) -> Vec<(f64, usize)> {

    let mut nearest_neighbors: Vec<(f64, usize)> = Vec::new();

    for i in 0..train_images.len() {

        // Calculate distance to training image
        let distance = find_euclidean_dist(
            &train_images[i],
            test_image
        );

        // If we have fewer than K neighbors, add this one
        if nearest_neighbors.len() < k {

            nearest_neighbors.push((distance, i));

        } else {

            // Find the neighbor with the largest distance
            let mut largest_index = 0;

            for j in 1..nearest_neighbors.len() {
                if nearest_neighbors[j].0 > nearest_neighbors[largest_index].0 {
                    largest_index = j;
                }
            }

            // Replace it if the new distance is smaller
            if distance < nearest_neighbors[largest_index].0 {
                nearest_neighbors[largest_index] = (distance, i);
            }
        }
    }

    // Sort the K nearest neighbors
    nearest_neighbors.sort_by(|a, b| a.0.total_cmp(&b.0));

    nearest_neighbors
}


fn predict_digit(
    neighbors: &Vec<(f64, usize)>,
    train_labels: &Vec<u8>,
) -> u8 {

    let mut votes = [0; 10];
    // count votes for each digit
    for &(_distance, index) in neighbors {
        let digit = train_labels[index] as usize;
        votes[digit] += 1;
    }

    // Find the digit with the most votes
    let mut predicted_digit = 0;

    for i in 1..10 {
        if votes[i] > votes[predicted_digit] {
            predicted_digit = i;
        }
    }

    predicted_digit as u8
}

fn load_train_images() -> Vec<Vec<u8>> {
    let file = File::open("../../../mnist/train-images-idx3-ubyte.gz").unwrap();
    let mut decoder = GzDecoder::new(file);

    let mut header = [0u8; 16];
    decoder.read_exact(&mut header).unwrap();

    let num_images = u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize;

    let mut buf = vec![0u8; num_images * 28 * 28];
    decoder.read_exact(&mut buf).unwrap();

    let mut images = Vec::new();

    for image in buf.chunks_exact(28 * 28) {
        images.push(image.to_vec());
    }

    images
}

fn load_train_labels() -> Vec<u8> {
    let file = File::open("../../../mnist/train-labels-idx1-ubyte.gz").unwrap();
    let mut decoder = GzDecoder::new(file);

    let mut header = [0u8; 8];
    decoder.read_exact(&mut header).unwrap();

    let num_labels = u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize;

    let mut labels = vec![0u8; num_labels];
    decoder.read_exact(&mut labels).unwrap();

    labels
}

fn load_test_images() -> Vec<Vec<u8>> {
    let file = File::open("../../../mnist/t10k-images-idx3-ubyte.gz").unwrap();
    let mut decoder = GzDecoder::new(file);

    let mut header = [0u8; 16];
    decoder.read_exact(&mut header).unwrap();

    let num_images = u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize;

    let mut buf = vec![0u8; num_images * 28 * 28];
    decoder.read_exact(&mut buf).unwrap();

    let mut images = Vec::new();

    for image in buf.chunks_exact(28 * 28) {
        images.push(image.to_vec());
    }

    images
}


fn load_test_labels() -> Vec<u8> {
    let file = File::open("../../../mnist/t10k-labels-idx1-ubyte.gz").unwrap();
    let mut decoder = GzDecoder::new(file);

    let mut header = [0u8; 8];
    decoder.read_exact(&mut header).unwrap();

    let num_labels = u32::from_be_bytes(header[4..8].try_into().unwrap()) as usize;

    let mut labels = vec![0u8; num_labels];
    decoder.read_exact(&mut labels).unwrap();

    labels
}

fn display_image(image: &[u8], scale: usize) {

    let image_size = 28;

    let width = image_size * scale;
    let height = image_size * scale;

    // Create pixel buffer
    let mut buffer = vec![0u32; width * height];

    for y in 0..image_size {
        for x in 0..image_size {

            let pixel = image[y * image_size + x];

            // Convert grayscale to RGB
            let gray = pixel as u32;
            let color = (gray << 16) | (gray << 8) | gray;

            // Scale individual pixels
            for sy in 0..scale {
                for sx in 0..scale {

                    let px = x * scale + sx;
                    let py = y * scale + sy;

                    buffer[py * width + px] = color;
                }
            }
        }
    }

    // Create window
    let mut window = Window::new(
        "MNIST Image",
        width,
        height,
        WindowOptions::default(),
    ).unwrap();

    // Display image
    while window.is_open() && !window.is_key_down(Key::Escape) {
        window.update_with_buffer(&buffer, width, height).unwrap();
    }
}
