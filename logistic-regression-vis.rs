
use macroquad::prelude::*;

const SCREEN_WIDTH: i32 = 1100;
const SCREEN_HEIGHT: i32 = 750;

const PLOT_LEFT: f32 = 40.0;
const PLOT_TOP: f32 = 40.0;
const PLOT_WIDTH: f32 = 760.0;
const PLOT_HEIGHT: f32 = 650.0;

const PANEL_X: f32 = 830.0;

// --------------------------------------------------
// DATA
// --------------------------------------------------

#[derive(Clone, Copy)]
struct Point {
    x1: f64,
    x2: f64,
    label: f64,
}

struct Dataset {
    points: Vec<Point>,
}

impl Dataset {
    fn new() -> Dataset {
        Dataset {
            points: Vec::new(),
        }
    }

    fn generate(&mut self) {

        self.points.clear();

        // Class 0
        self.points.push(Point {
            x1: -0.5,
            x2: -0.5,
            label: 0.0,
        });

        // Class 1
        self.points.push(Point {
            x1: 0.5,
            x2: 0.5,
            label: 1.0,
        });
    }

    fn add_point(&mut self, x1: f64, x2: f64, label: f64) {
        self.points.push(Point {
            x1,
            x2,
            label,
        });
    }
}

struct LogisticRegression {
    weights: Vec<f64>,
    bias: f64,
    degree: usize,
}

impl LogisticRegression {

    fn new(degree: usize) -> LogisticRegression {

        let number_of_weights = degree * (degree + 3) / 2;

        LogisticRegression {
            weights: vec![0.0; number_of_weights],
            bias: 0.0,
            degree,
        }
    }

    // Generate polynomial features
    fn generate_features(&self, x1: f64, x2: f64) -> Vec<f64> {

        let mut features = Vec::new();

        // Generate features for each polynomial degree
        for total_degree in 1..=self.degree {

            // Generate combinations of x1 and x2
            for x1_power in (0..=total_degree).rev() {

                let x2_power = total_degree - x1_power;

                let feature =
                    x1.powi(x1_power as i32)
                    * x2.powi(x2_power as i32);

                features.push(feature);
            }
        }

        features
    }

    // Sigmoid activation function
    fn sigmoid(z: f64) -> f64 {

        1.0 / (1.0 + (-z.clamp(-30.0, 30.0)).exp())
    }

    // Calculate probability
    fn predict_probability(&self, x1: f64, x2: f64) -> f64 {

        let features = self.generate_features(x1, x2);

        let mut z = self.bias;

        for i in 0..self.weights.len() {

            z += self.weights[i] * features[i];
        }

        Self::sigmoid(z)
    }

    // Predict class
    fn predict(&self, x1: f64, x2: f64) -> u8 {

        if self.predict_probability(x1, x2) >= 0.5 {
            1
        } else {
            0
        }
    }

    // Calculate Binary Cross-Entropy
    fn calculate_loss(&self, dataset: &Dataset) -> f64 {

        if dataset.points.is_empty() {
            return 0.0;
        }

        let mut total_loss = 0.0;

        for point in &dataset.points {

            let prediction =
                self.predict_probability(point.x1, point.x2);

            let prediction =
                prediction.clamp(1e-7, 1.0 - 1e-7);

            let loss =
                -point.label * prediction.ln()
                -(1.0 - point.label) * (1.0 - prediction).ln();

            total_loss += loss;
        }

        total_loss / dataset.points.len() as f64
    }

    // Gradient descent
    fn train(&mut self, dataset: &Dataset, learning_rate: f64) {

        let m = dataset.points.len();

        if m == 0 {
            return;
        }

        let mut gradients = vec![0.0; self.weights.len()];
        let mut bias_gradient = 0.0;

        for point in &dataset.points {

            let features =
                self.generate_features(point.x1, point.x2);

            let prediction =
                self.predict_probability(point.x1, point.x2);

            let error = prediction - point.label;

            for i in 0..self.weights.len() {

                gradients[i] += error * features[i];
            }

            bias_gradient += error;
        }

        // Average gradients
        for i in 0..self.weights.len() {

            gradients[i] /= m as f64;
        }

        bias_gradient /= m as f64;

        // Update weights
        for i in 0..self.weights.len() {

            self.weights[i] -= learning_rate * gradients[i];
        }

        self.bias -= learning_rate * bias_gradient;
    }

    // Calculate accuracy
    fn accuracy(&self, dataset: &Dataset) -> f64 {

        if dataset.points.is_empty() {
            return 0.0;
        }

        let mut correct = 0;

        for point in &dataset.points {

            let prediction = self.predict(point.x1, point.x2);

            if prediction as f64 == point.label {
                correct += 1;
            }
        }

        correct as f64 / dataset.points.len() as f64 * 100.0
    }

    // Change polynomial order
    fn set_degree(&mut self, degree: usize) {

        self.degree = degree;

        let number_of_weights = degree * (degree + 3) / 2;

        // Preserve existing weights and initialize new ones to zero
        self.weights.resize(number_of_weights, 0.0);
    }
}


// --------------------------------------------------
// COORDINATE CONVERSION
// --------------------------------------------------

fn world_to_screen(x: f64, y: f64) -> (f32, f32) {

    let screen_x =
        PLOT_LEFT + ((x + 1.0) / 2.0) as f32 * PLOT_WIDTH;

    let screen_y =
        PLOT_TOP + ((1.0 - y) / 2.0) as f32 * PLOT_HEIGHT;

    (screen_x, screen_y)
}

fn screen_to_world(x: f32, y: f32) -> (f64, f64) {

    let world_x =
        ((x - PLOT_LEFT) / PLOT_WIDTH * 2.0 - 1.0) as f64;

    let world_y =
        (1.0 - (y - PLOT_TOP) / PLOT_HEIGHT * 2.0) as f64;

    (world_x, world_y)
}

// --------------------------------------------------
// DRAW COORDINATE PLANE
// --------------------------------------------------

fn draw_plane() {

    draw_rectangle(
        PLOT_LEFT,
        PLOT_TOP,
        PLOT_WIDTH,
        PLOT_HEIGHT,
        Color::from_rgba(245, 245, 250, 255),
    );

    // Grid lines
    for i in 0..=10 {

        let x = PLOT_LEFT + i as f32 * PLOT_WIDTH / 10.0;
        let y = PLOT_TOP + i as f32 * PLOT_HEIGHT / 10.0;

        draw_line(
            x, PLOT_TOP,
            x, PLOT_TOP + PLOT_HEIGHT,
            1.0,
            LIGHTGRAY,
        );

        draw_line(
            PLOT_LEFT, y,
            PLOT_LEFT + PLOT_WIDTH, y,
            1.0,
            LIGHTGRAY,
        );
    }

    // Coordinate axes
    let (origin_x, origin_y) = world_to_screen(0.0, 0.0);

    draw_line(
        origin_x, PLOT_TOP,
        origin_x, PLOT_TOP + PLOT_HEIGHT,
        2.0,
        GRAY,
    );

    draw_line(
        PLOT_LEFT, origin_y,
        PLOT_LEFT + PLOT_WIDTH, origin_y,
        2.0,
        GRAY,
    );

    draw_rectangle_lines(
        PLOT_LEFT,
        PLOT_TOP,
        PLOT_WIDTH,
        PLOT_HEIGHT,
        2.0,
        DARKGRAY,
    );

    draw_text("X1", PLOT_LEFT + PLOT_WIDTH - 20.0,
        PLOT_TOP + PLOT_HEIGHT + 25.0, 18.0, BLACK);

    draw_text("X2", PLOT_LEFT - 28.0,
        PLOT_TOP + 15.0, 18.0, BLACK);
}

// --------------------------------------------------
// DRAW PROBABILITY HEATMAP
// --------------------------------------------------

fn draw_heatmap(model: &LogisticRegression) {

    let cell_size = 5.0;

    let mut x = PLOT_LEFT;

    while x < PLOT_LEFT + PLOT_WIDTH {

        let mut y = PLOT_TOP;

        while y < PLOT_TOP + PLOT_HEIGHT {

            let (wx, wy) = screen_to_world(
                x + cell_size / 2.0,
                y + cell_size / 2.0,
            );

            let probability =
                model.predict_probability(wx, wy) as f32;

            // Blue = Class 0
            // Red = Class 1
            let color = Color::new(
                probability,
                0.0,
                1.0 - probability,
                0.65,
            );

            draw_rectangle(
                x,
                y,
                cell_size + 1.0,
                cell_size + 1.0,
                color,
            );

            y += cell_size;
        }

        x += cell_size;
    }
}

// --------------------------------------------------
// DRAW DECISION BOUNDARY
// --------------------------------------------------

fn draw_decision_boundary(model: &LogisticRegression) {

    let step = 5.0;

    let right = PLOT_LEFT + PLOT_WIDTH;
    let bottom = PLOT_TOP + PLOT_HEIGHT;

    let mut x = PLOT_LEFT;

    while x < right {

        let mut y = PLOT_TOP;

        while y < bottom {

            let (wx, wy) = screen_to_world(x, y);

            let probability =
                model.predict_probability(wx, wy) - 0.5;

            // Check the neighboring point to the right
            let (right_wx, right_wy) =
                screen_to_world(x + step, y);

            let right_probability =
                model.predict_probability(right_wx, right_wy) - 0.5;

            // Check the neighboring point below
            let (down_wx, down_wy) =
                screen_to_world(x, y + step);

            let down_probability =
                model.predict_probability(down_wx, down_wy) - 0.5;

            // Detect where probability crosses 0.5
            if probability * right_probability < 0.0 {

                draw_circle(x + step / 2.0, y, 2.0, BLACK);
            }

            if probability * down_probability < 0.0 {

                draw_circle(x, y + step / 2.0, 2.0, BLACK);
            }

            y += step;
        }

        x += step;
    }
}


// --------------------------------------------------
// DRAW DATA POINTS
// --------------------------------------------------

fn draw_points(dataset: &Dataset) {

    for point in &dataset.points {

        let (x, y) = world_to_screen(point.x1, point.x2);

        let color = if point.label == 0.0 {
            BLUE
        } else {
            RED
        };

        draw_circle(x, y, 7.0, color);

        draw_circle_lines(x, y, 7.0, 1.5, BLACK);
    }
}

// --------------------------------------------------
// DRAW INFORMATION PANEL
// --------------------------------------------------

fn draw_panel(
    model: &LogisticRegression,
    dataset: &Dataset,
    learning_rate: f64,
    training: bool,
    selected_class: f64,
    iterations: usize,
) {

    draw_text("LOGISTIC REGRESSION", PANEL_X, 55.0, 23.0, BLACK);

    draw_line(PANEL_X, 70.0, 1070.0, 70.0, 1.0, GRAY);

    draw_text("MODEL PARAMETERS", PANEL_X, 110.0, 19.0, DARKGRAY);

    draw_text(
        &format!("Polynomial order: {}", model.degree),
        PANEL_X, 145.0, 19.0, BLACK,
    );

    draw_text(
        &format!("Number of weights: {}", model.weights.len()),
        PANEL_X, 175.0, 19.0, BLACK,
    );

    draw_text(
        &format!("Bias: {:.4}", model.bias),
        PANEL_X, 205.0, 19.0, BLACK,
    );

    draw_line(PANEL_X, 225.0, 1070.0, 225.0, 1.0, GRAY);

    draw_text("TRAINING METRICS", PANEL_X, 265.0, 19.0, DARKGRAY);

    draw_text(
        &format!("BCE Loss: {:.5}", model.calculate_loss(dataset)),
        PANEL_X, 300.0, 19.0, BLACK,
    );

    draw_text(
        &format!("Accuracy: {:.2}%", model.accuracy(dataset)),
        PANEL_X, 330.0, 19.0, BLACK,
    );

    draw_text(
        &format!("Points: {}", dataset.points.len()),
        PANEL_X, 360.0, 19.0, BLACK,
    );

    draw_text(
        &format!("Iterations: {}", iterations),
        PANEL_X, 390.0, 19.0, BLACK,
    );

    draw_line(PANEL_X, 410.0, 1070.0, 410.0, 1.0, GRAY);

    draw_text("SETTINGS", PANEL_X, 450.0, 19.0, DARKGRAY);

    draw_text(
        &format!("Learning rate: {:.4}", learning_rate),
        PANEL_X, 485.0, 19.0, BLACK,
    );

    draw_text(
        &format!("Selected class: {}", selected_class as u8),
        PANEL_X, 515.0, 19.0, BLACK,
    );

    draw_text(
        if training { "Status: TRAINING" } else { "Status: PAUSED" },
        PANEL_X, 550.0, 19.0, BLACK,
    );

    draw_line(PANEL_X, 575.0, 1070.0, 575.0, 1.0, GRAY);

    draw_text("CONTROLS", PANEL_X, 610.0, 19.0, DARKGRAY);

    draw_text("1 / 2: Select class", PANEL_X, 640.0, 16.0, BLACK);
    draw_text("Click: Add / drag point", PANEL_X, 665.0, 16.0, BLACK);
    draw_text("Space: Train / Pause", PANEL_X, 690.0, 16.0, BLACK);
    draw_text("N: Single iteration", PANEL_X, 715.0, 16.0, BLACK);
    draw_text("Up / Down: Polynomial order", PANEL_X, 740.0, 15.0, BLACK);
}

// --------------------------------------------------
// MAIN
// --------------------------------------------------

fn window_conf() -> Conf {

    Conf {
        window_title: "Logistic Regression".to_owned(),
        window_width: SCREEN_WIDTH,
        window_height: SCREEN_HEIGHT,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {


    let mut dragged_point: Option<usize> = None;


    let mut dataset = Dataset::new();
    dataset.generate();

    let mut polynomial_degree = 1;

    let mut model = LogisticRegression::new(polynomial_degree);

    let mut learning_rate = 0.1;
    let mut training = false;
    let mut selected_class = 0.0;
    let mut iterations = 0usize;

    loop {

        clear_background(WHITE);

        // --------------------------
        // KEYBOARD INPUT
        // --------------------------

        if is_key_pressed(KeyCode::Key1) {
            selected_class = 0.0;
        }

        if is_key_pressed(KeyCode::Key2) {
            selected_class = 1.0;
        }

        if is_key_pressed(KeyCode::Space) {
            training = !training;
        }

        if is_key_pressed(KeyCode::N) {
            model.train(&dataset, learning_rate);
            iterations += 1;
        }

        if is_key_pressed(KeyCode::R) {
            model = LogisticRegression::new(polynomial_degree);

            iterations = 0;
            training = false;
        }

        if is_key_pressed(KeyCode::G) {
            dataset.generate();
            model = LogisticRegression::new(polynomial_degree);
            iterations = 0;
            training = false;
        }

        if is_key_pressed(KeyCode::Equal)
            || is_key_pressed(KeyCode::KpAdd) {
            learning_rate *= 1.5;
        }

        if is_key_pressed(KeyCode::Minus)
            || is_key_pressed(KeyCode::KpSubtract) {
            learning_rate /= 1.5;
        }

        // Increase polynomial order
        if is_key_pressed(KeyCode::Up) {

            if polynomial_degree < 6 {

                polynomial_degree += 1;

                model.set_degree(polynomial_degree);
            }
        }

        // Decrease polynomial order
        if is_key_pressed(KeyCode::Down) {

            if polynomial_degree > 1 {

                polynomial_degree -= 1;

                model.set_degree(polynomial_degree);
            }
        }

        
        // --------------------------
        // MOUSE INPUT
        // --------------------------

        let (mouse_x, mouse_y) = mouse_position();

        // Mouse button initially pressed
        if is_mouse_button_pressed(MouseButton::Left) {

            if mouse_x >= PLOT_LEFT
                && mouse_x <= PLOT_LEFT + PLOT_WIDTH
                && mouse_y >= PLOT_TOP
                && mouse_y <= PLOT_TOP + PLOT_HEIGHT {

                let mut closest_point: Option<usize> = None;
                let mut closest_distance = 15.0 * 15.0;

                // Check whether we clicked an existing point
                for i in 0..dataset.points.len() {

                    let point = &dataset.points[i];

                    let (px, py) = world_to_screen(point.x1, point.x2);

                    let dx = mouse_x - px;
                    let dy = mouse_y - py;

                    let distance = dx * dx + dy * dy;

                    if distance < closest_distance {

                        closest_distance = distance;
                        closest_point = Some(i);
                    }
                }

                // Select existing point or create a new one
                if let Some(index) = closest_point {

                    dragged_point = Some(index);

                } else {

                    let (x, y) = screen_to_world(mouse_x, mouse_y);

                    dataset.add_point(x, y, selected_class);

                    // Immediately allow the new point to be dragged
                    dragged_point = Some(dataset.points.len() - 1);
                }
            }
        }

        // Move selected point while mouse is held
        if is_mouse_button_down(MouseButton::Left) {

            if let Some(index) = dragged_point {

                // Keep point inside coordinate plane
                let x = mouse_x.clamp(
                    PLOT_LEFT,
                    PLOT_LEFT + PLOT_WIDTH
                );

                let y = mouse_y.clamp(
                    PLOT_TOP,
                    PLOT_TOP + PLOT_HEIGHT
                );

                let (world_x, world_y) = screen_to_world(x, y);

                dataset.points[index].x1 = world_x;
                dataset.points[index].x2 = world_y;
            }
        }

        // Release selected point
        if is_mouse_button_released(MouseButton::Left) {

            dragged_point = None;
        }


        // --------------------------
        // GRADIENT DESCENT
        // --------------------------

        if training {

            model.train(&dataset, learning_rate);
            iterations += 1;
        }

        // --------------------------
        // RENDER
        // --------------------------

        draw_plane();

        draw_heatmap(&model);

        //draw_decision_boundary(&model);

        draw_points(&dataset);

        draw_panel(
            &model,
            &dataset,
            learning_rate,
            training,
            selected_class,
            iterations,
        );

        next_frame().await;
    }
}
