use std::error::Error;
use std::collections::HashMap;

struct Email {
    subject: String,
    message: String,
    label: String,
}

struct Perceptron {
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
}


fn main() -> Result<(), Box<dyn Error>> {

    let emails = load_emails()?;

    println!("Total emails loaded: {}", emails.len());

    // --------------------------------------------------
    // STEP 2: BUILD VOCABULARY
    // --------------------------------------------------

    let vocabulary = build_vocabulary(&emails);

    println!("Vocabulary size: {}", vocabulary.len());


    // --------------------------------------------------
    // STEP 3: CONVERT EMAILS INTO FEATURES
    // --------------------------------------------------

    let features = encode_email(&emails[0], &vocabulary);

    println!("Features in first email: {}", features.len());


    // --------------------------------------------------
    // STEP 4: INITIALIZE PERCEPTRON
    // --------------------------------------------------

    let mut model = initialize_perceptron(
        vocabulary.len(),
        0.01
    );


    // --------------------------------------------------
    // STEP 5: TRAIN PERCEPTRON
    // --------------------------------------------------

    train_perceptron(
        &emails,
        &vocabulary,
        &mut model,
        10
    );


    // --------------------------------------------------
    // TEST WITH A NEW EMAIL
    // --------------------------------------------------

    let test_email = Email {
        subject: String::from("Free money"),
        message: String::from(
            "Congratulations! You have won free money. Claim your offer today!"
        ),
        label: String::new(),
    };

    let test_features = encode_email(&test_email, &vocabulary);

    let spam_probability = predict_probability(&test_features, &model);
    let ham_probability = 1.0 - spam_probability;

    println!("Spam probability: {:.2}%", spam_probability * 100.0);
    println!("Ham probability:  {:.2}%", ham_probability * 100.0);

    //let prediction = predict(&test_features, &model);

    if spam_probability >= 0.5 {
        println!("Prediction: SPAM");
    }
    else {
        println!("Prediction: HAM");
    }

    Ok(())
}


fn load_emails() -> Result<Vec<Email>, Box<dyn Error>> {

    let mut reader = csv::Reader::from_path(
        "../../enron/enron_spam_data.csv"
    )?;

    let headers = reader.headers()?.clone();

    // Find the column positions
    let subject_index = headers.iter().position(|h| h == "Subject").unwrap();
    let message_index = headers.iter().position(|h| h == "Message").unwrap();
    let label_index = headers.iter().position(|h| h == "Spam/Ham").unwrap();

    let mut emails: Vec<Email> = Vec::new();

    // Read each row in the CSV
    for result in reader.records() {

        let record = result?;

        let subject = record.get(subject_index).unwrap_or("").to_string();
        let message = record.get(message_index).unwrap_or("").to_string();
        let label = record.get(label_index).unwrap_or("").to_string();

        let email = Email {
            subject,
            message,
            label,
        };

        emails.push(email);
    }

    Ok(emails)
}


fn build_vocabulary(emails: &Vec<Email>) -> HashMap<String, usize> {

    let mut vocabulary: HashMap<String, usize> = HashMap::new();

    // Go through every email.
    for email in emails {

        // Combine subject and message.
        let text = format!("{} {}", email.subject, email.message);

        // Split the email into individual words.
        let words = text.split_whitespace();

        for word in words {

            // Convert to lowercase and remove punctuation.
            let word = word.to_lowercase();

            let word = word.trim_matches(|c: char| !c.is_alphanumeric());

            // Ignore empty words.
            if word.is_empty() {
                continue;
            }

            // Only add words that aren't already in the vocabulary.
            if !vocabulary.contains_key(word) {

                let index = vocabulary.len();

                vocabulary.insert(word.to_string(), index);
            }
        }
    }

    vocabulary
}


fn encode_email(
    email: &Email,
    vocabulary: &HashMap<String, usize>
) -> Vec<(usize, f64)> {

    let mut features: Vec<(usize, f64)> = Vec::new();

    let text = format!("{} {}", email.subject, email.message);

    let words = text.split_whitespace();

    for word in words {

        let word = word.to_lowercase();

        let word = word.trim_matches(|c: char| !c.is_alphanumeric());

        if word.is_empty() {
            continue;
        }

        // Find the word's index in our vocabulary.
        if let Some(&index) = vocabulary.get(word) {

            // Check whether this word is already in our feature vector.
            let mut found = false;

            for feature in features.iter_mut() {

                if feature.0 == index {

                    // Increment the word frequency.
                    feature.1 += 1.0;

                    found = true;

                    break;
                }
            }

            // If this is the first occurrence, add it.
            if !found {
                features.push((index, 1.0));
            }
        }
    }

    features
}


fn initialize_perceptron(
    vocabulary_size: usize,
    learning_rate: f64
) -> Perceptron {

    let mut weights: Vec<f64> = Vec::new();

    // Initialize every weight to zero.
    for _ in 0..vocabulary_size {
        weights.push(0.0);
    }

    Perceptron {
        weights,
        bias: 0.0,
        learning_rate,
    }
}

fn predict(
    features: &Vec<(usize, f64)>,
    model: &Perceptron
) -> f64 {

    let mut sum = model.bias;

    // Calculate the weighted sum.
    for feature in features {

        let index = feature.0;
        let value = feature.1;

        sum += model.weights[index] * value;
    }

    // Apply the step function.
    if sum >= 0.0 {
        return 1.0;
    }
    else {
        return 0.0;
    }
}

fn train_perceptron(
    emails: &Vec<Email>,
    vocabulary: &HashMap<String, usize>,
    model: &mut Perceptron,
    epochs: usize
) {

    for epoch in 0..epochs {

        let mut incorrect_predictions = 0;

        // Go through every training email.
        for email in emails {

            // Convert the email into numerical features.
            let features = encode_email(email, vocabulary);

            // Convert the actual label into a number.
            let actual: f64;

            if email.label.trim().eq_ignore_ascii_case("spam") {
                actual = 1.0;
            }
            else {
                actual = 0.0;
            }

            // Make a prediction.
            let predicted = predict(&features, model);

            // Calculate the prediction error.
            let error = actual - predicted;

            // Count incorrect predictions.
            if error != 0.0 {
                incorrect_predictions += 1;
            }

            // Update the weights if the prediction was incorrect.
            if error != 0.0 {

                for feature in &features {

                    let index = feature.0;
                    let value = feature.1;

                    model.weights[index] +=
                        model.learning_rate * error * value;
                }

                // Update the bias.
                model.bias += model.learning_rate * error;
            }
        }

        let accuracy =
            (emails.len() - incorrect_predictions) as f64
            / emails.len() as f64 * 100.0;

        println!(
            "Epoch {}: Incorrect = {}, Accuracy = {:.2}%",
            epoch + 1,
            incorrect_predictions,
            accuracy
        );
    }
}

fn predict_probability(
    features: &Vec<(usize, f64)>,
    model: &Perceptron
) -> f64 {

    let mut sum = model.bias;

    // Calculate weighted sum.
    for feature in features {

        let index = feature.0;
        let value = feature.1;

        sum += model.weights[index] * value;
    }

    // Apply sigmoid function.
    let probability = 1.0 / (1.0 + (-sum).exp());

    probability
}