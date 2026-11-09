use std::error::Error;
use std::collections::{HashMap, HashSet};

struct ProbabilityModel {
    spam_prior: f64,
    ham_prior: f64,

    spam_word_probabilities: HashMap<String, f64>,
    ham_word_probabilities: HashMap<String, f64>,
}

struct Email {
    subject: String,
    message: String,
    label: String,
}

struct WordCounts {
    spam_words: HashMap<String, usize>,
    ham_words: HashMap<String, usize>,

    spam_email_count: usize,
    ham_email_count: usize,

    spam_total_words: usize,
    ham_total_words: usize,
}

/* There's just four functions in main, one that loads the emails, one
   that trains on the emails, one that calculates the probabilities, and 
   lastly one that classifies on a new email.
   
   Bayes' theorem is used to predict whether an email is spam or legitimate 
   (ham) based on the words it contains. We do this by first calculating the 
   prior probabilities then the word probabilities, apply Bayes's thoerem by 
   combining the prior and word probabilities, and finally making a 
   prediction based on the result. 
   */
fn main() -> Result<(), Box<dyn Error>> {

    let emails = load_emails()?;

    println!("Total emails loaded: {}", emails.len());

     // Stage 1: Count word frequencies
    let counts = train_classifier(&emails);

    // Stage 2: Calculate probabilities
    let model = calculate_probabilities(&counts);

    println!("Spam prior: {}", model.spam_prior);
    println!("Ham prior: {}", model.ham_prior);

    /* //testing a specific word's probabilities
    // Inspect a word
    let word = "free";

    let spam_probability =
        model.spam_word_probabilities.get(word).unwrap_or(&0.0);

    let ham_probability =
        model.ham_word_probabilities.get(word).unwrap_or(&0.0);

    println!("\nWord: {}", word);

    println!("P(free | spam): {}", spam_probability);
    println!("P(free | ham): {}", ham_probability);
    */

    // Change this to any new email you want to classify:
    let test_email = "Congratulations! You have won free money. Click here to claim your prize!";

    let prediction = classify_email(test_email, &model);

    println!("Email: {}", test_email);
    println!("Prediction: {}", prediction);

    /* // Train the classifier
    let classifier = train_classifier(&emails);

    println!("Spam emails: {}", classifier.spam_email_count);
    println!("Ham emails: {}", classifier.ham_email_count);

    println!("Total spam words: {}", classifier.spam_total_words);
    println!("Total ham words: {}", classifier.ham_total_words);

    println!("Unique spam words: {}", classifier.spam_words.len());
    println!("Unique ham words: {}", classifier.ham_words.len());

    // Test looking up a word
    let word = "free";

    let spam_count = classifier.spam_words.get(word).unwrap_or(&0);
    let ham_count = classifier.ham_words.get(word).unwrap_or(&0);

    println!("\nWord: {}", word);
    println!("Spam occurrences: {}", spam_count);
    println!("Ham occurrences: {}", ham_count);
    */

    /* // Count the number of spam and ham emails
    let mut spam_count = 0;
    let mut ham_count = 0;

    for email in &emails {

        if email.label == "spam" {
            spam_count += 1;
        }
        else if email.label == "ham" {
            ham_count += 1;
        }
    }

    println!("Spam emails: {}", spam_count);
    println!("Ham emails: {}", ham_count);

    // Print the first email
    if emails.len() > 0 {
        println!("\nFirst email:");
        println!("Subject: {}", emails[294].subject);
        println!("Message: {}", emails[294].message);
        println!("Label: {}", emails[294].label);
    }
    */

    Ok(())
}

fn load_emails() -> Result<Vec<Email>, Box<dyn Error>> {

    let mut reader = csv::Reader::from_path(
        "../../../enron/enron_spam_data.csv"
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

fn train_classifier(emails: &Vec<Email>) -> WordCounts {

    // Create our two HashMaps
    let mut spam_words: HashMap<String, usize> = HashMap::new();
    let mut ham_words: HashMap<String, usize> = HashMap::new();

    // Keep track of email and word counts
    let mut spam_email_count = 0;
    let mut ham_email_count = 0;

    let mut spam_total_words = 0;
    let mut ham_total_words = 0;

    // Process every email
    for email in emails {

        // Combine subject and message
        let text = format!("{} {}", email.subject, email.message);

        // Determine which HashMap to use
        let is_spam = email.label.trim().to_lowercase() == "spam";

        if is_spam {
            spam_email_count += 1;
        } else if email.label.trim().to_lowercase() == "ham" {
            ham_email_count += 1;
        } else {
            continue;
        }

        // Split the email into individual words
        let words = text.split_whitespace();

        for word in words {

            // Convert word to lowercase
            let word = word.to_lowercase();

            // Remove basic punctuation
            let word = word.trim_matches(|c: char| !c.is_alphanumeric());

            if word.is_empty() {
                continue;
            }

            if is_spam {
                let count = spam_words.entry(word.to_string()).or_insert(0);
                *count += 1;

                spam_total_words += 1;
            } else {
                let count = ham_words.entry(word.to_string()).or_insert(0);
                *count += 1;

                ham_total_words += 1;
            }
        }
    }

    // Return all the learned information
    WordCounts {
        spam_words,
        ham_words,
        spam_email_count,
        ham_email_count,
        spam_total_words,
        ham_total_words,
    }
}


fn calculate_probabilities(counts: &WordCounts) -> ProbabilityModel {

    // Calculate total number of emails
    let total_emails =
        counts.spam_email_count + counts.ham_email_count;

    if total_emails == 0 {
        panic!("Cannot train classifier with zero emails");
    }

    // Calculate prior probabilities
    let spam_prior =
        counts.spam_email_count as f64 / total_emails as f64;

    let ham_prior =
        counts.ham_email_count as f64 / total_emails as f64;


    // Create a collection of all unique words
    let mut vocabulary: HashSet<String> = HashSet::new();

    for word in counts.spam_words.keys() {
        vocabulary.insert(word.clone());
    }

    for word in counts.ham_words.keys() {
        vocabulary.insert(word.clone());
    }

    // Get vocabulary size
    let vocabulary_size = vocabulary.len();


    // Create HashMaps to store probabilities
    let mut spam_word_probabilities: HashMap<String, f64> =
        HashMap::new();

    let mut ham_word_probabilities: HashMap<String, f64> =
        HashMap::new();


    // Calculate probability for every word
    for word in vocabulary {

        // Get word count in spam
        let spam_count =
            counts.spam_words.get(&word).unwrap_or(&0);

        // Get word count in ham
        let ham_count =
            counts.ham_words.get(&word).unwrap_or(&0);


        // Calculate spam probability with Laplace smoothing
        let spam_probability =
            (*spam_count as f64 + 1.0) /
            (counts.spam_total_words as f64 + vocabulary_size as f64);

        // Calculate ham probability with Laplace smoothing
        let ham_probability =
            (*ham_count as f64 + 1.0) /
            (counts.ham_total_words as f64 + vocabulary_size as f64);


        // Store probabilities
        spam_word_probabilities.insert(
            word.clone(),
            spam_probability
        );

        ham_word_probabilities.insert(
            word,
            ham_probability
        );
    }


    // Return our probability model
    ProbabilityModel {
        spam_prior,
        ham_prior,
        spam_word_probabilities,
        ham_word_probabilities,
    }
}


fn classify_email(
    email_text: &str,
    model: &ProbabilityModel
) -> String {

    // Start with the prior probabilities
    let mut spam_score = model.spam_prior.ln();
    let mut ham_score = model.ham_prior.ln();

    // Split email into words
    let words = email_text.split_whitespace();

    for word in words {

        // Convert to lowercase
        let word = word.to_lowercase();

        // Remove punctuation
        let word = word.trim_matches(|c: char| !c.is_alphanumeric());

        if word.is_empty() {
            continue;
        }

        // Look up spam probability
        let spam_probability =
            model.spam_word_probabilities.get(word).unwrap_or(&0.0);

        // Look up ham probability
        let ham_probability =
            model.ham_word_probabilities.get(word).unwrap_or(&0.0);

        // Add log probabilities to our scores
        if *spam_probability > 0.0 {
            spam_score += spam_probability.ln();
        }

        if *ham_probability > 0.0 {
            ham_score += ham_probability.ln();
        }
    }

    // Compare the scores
    if spam_score > ham_score {
        return "spam".to_string();
    } else {
        return "ham".to_string();
    }
}