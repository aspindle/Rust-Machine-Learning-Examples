use macroquad::prelude::*;

struct Point {
    x: f32,
    y: f32,
}

// Calculate the regression line using ordinary least squares.
fn linear_regression(points: &Vec<Point>) -> Option<(f32, f32, f32)> {

    if points.len() < 2 {
        return None;
    }

    let mut sum_x = 0.0;
    let mut sum_y = 0.0;

    for point in points {
        sum_x += point.x;
        sum_y += point.y;
    }

    let mean_x = sum_x / points.len() as f32;
    let mean_y = sum_y / points.len() as f32;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for point in points {
        numerator += (point.x - mean_x) * (point.y - mean_y);
        denominator += (point.x - mean_x) * (point.x - mean_x);
    }

    if denominator == 0.0 {
        return None;
    }

    let slope = numerator / denominator;
    let intercept = mean_y - slope * mean_x;

    // Calculate mean squared error.
    let mut total_error = 0.0;

    for point in points {
        let prediction = intercept + slope * point.x;
        let error = point.y - prediction;

        total_error += error * error;
    }

    let mse = total_error / points.len() as f32;

    Some((slope, intercept, mse))
}

#[macroquad::main("Linear Regression")]
async fn main() {

    let mut points: Vec<Point> = Vec::new();

    loop {

        clear_background(WHITE);

        let width = screen_width();
        let height = screen_height();

        // Plot boundaries.
        let left = 60.0;
        let top = 50.0;
        let right = width - 30.0;
        let bottom = height - 60.0;

        let plot_width = right - left;
        let plot_height = bottom - top;

        // Draw coordinate axes.
        draw_line(left, top, left, bottom, 2.0, BLACK);
        draw_line(left, bottom, right, bottom, 2.0, BLACK);

        draw_text("X", right - 15.0, bottom + 30.0, 20.0, BLACK);
        draw_text("Y", left - 35.0, top + 10.0, 20.0, BLACK);

        // Draw grid.
        for i in 1..10 {
            let x = left + plot_width * i as f32 / 10.0;
            let y = top + plot_height * i as f32 / 10.0;

            draw_line(x, top, x, bottom, 1.0, LIGHTGRAY);
            draw_line(left, y, right, y, 1.0, LIGHTGRAY);
        }

        // Add a point when the user clicks inside the plot.
        if is_mouse_button_pressed(MouseButton::Left) {

            let (mouse_x, mouse_y) = mouse_position();

            if mouse_x >= left && mouse_x <= right
                && mouse_y >= top && mouse_y <= bottom {

                let x = (mouse_x - left) / plot_width;
                let y = 1.0 - (mouse_y - top) / plot_height;

                points.push(Point { x, y });
            }
        }

        // Clear all points with C.
        if is_key_pressed(KeyCode::C) {
            points.clear();
        }

        // Draw data points.
        for point in &points {

            let screen_x = left + point.x * plot_width;
            let screen_y = bottom - point.y * plot_height;

            draw_circle(screen_x, screen_y, 6.0, BLUE);
        }

        // Calculate and draw regression line.
        if let Some((slope, intercept, mse)) =
            linear_regression(&points) {

            let y1 = intercept;
            let y2 = intercept + slope;

            let screen_y1 = bottom - y1 * plot_height;
            let screen_y2 = bottom - y2 * plot_height;

            draw_line(
                left,
                screen_y1,
                right,
                screen_y2,
                3.0,
                RED,
            );

            draw_text(
                &format!("Slope: {:.4}", slope),
                70.0, 25.0, 20.0, BLACK,
            );

            draw_text(
                &format!("Intercept: {:.4}", intercept),
                230.0, 25.0, 20.0, BLACK,
            );

            draw_text(
                &format!("MSE: {:.6}", mse),
                450.0, 25.0, 20.0, BLACK,
            );

        } else {
            draw_text(
                "Add at least two points",
                70.0, 25.0, 20.0, BLACK,
            );
        }

        draw_text(
            "Click to add points | C to clear",
            60.0, height - 15.0, 18.0, DARKGRAY,
        );

        next_frame().await;
    }
}

