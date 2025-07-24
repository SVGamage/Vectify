use clap::Parser;
use image::{DynamicImage, GrayImage, ImageBuffer, Luma};
use imageproc::edges;
use imageproc::filter::gaussian_blur_f32;
use std::fs::File;
use std::io::Write;
use svg::Document;
use svg::node::element::path::Data;
use svg::node::element::{Path as SvgPath, Rectangle};

#[derive(Parser, Debug)]
#[clap(author, version, about)]
struct Args {
    /// Input bitmap file path
    #[clap(value_parser, default_value = "test.png")]
    input: String,

    /// Output SVG file path
    #[clap(value_parser, default_value = "output.svg")]
    output: String,
}

fn main() {
    let args = Args::parse();

    println!("Read the input bitmap image: {}", args.input);
    let input_image = match image::open(&args.input) {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Error opening image '{}': {}", args.input, e);
            return;
        }
    };

    println!("Preprocess the image (optional)");
    let preprocessed = preprocess_image(&input_image);

    println!("Apply the tracing algorithm");
    let vector_data = trace_bitmap(&preprocessed);

    println!("Export the vector data to SVG: {}", args.output);
    if let Err(e) = export_vector_data(
        &vector_data,
        &args.output,
        input_image.width(),
        input_image.height(),
    ) {
        eprintln!("Failed to export vector data: {}", e);
    } else {
        println!("Vector data successfully exported to '{}'.", args.output);
    }
}

fn preprocess_image(image: &DynamicImage) -> GrayImage {
    let gray = image.to_luma8();
    gaussian_blur_f32(&gray, 1.5)
}

// ——————————————————————————————————————————————————————————
//   Ramer–Douglas–Peucker simplification
// ——————————————————————————————————————————————————————————

// Perpendicular distance from p to the line p1–p2.
fn perpendicular_distance(p: &(f64, f64), p1: &(f64, f64), p2: &(f64, f64)) -> f64 {
    let dx = p2.0 - p1.0;
    let dy = p2.1 - p1.1;
    (p.0 * dy - p.1 * dx + p2.0 * p1.1 - p2.1 * p1.0).abs() / dx.hypot(dy)
}

// Recursive RDP helper.
fn rdp(points: &[(f64, f64)], eps: f64, out: &mut Vec<(f64, f64)>) {
    if points.len() < 2 {
        return;
    }
    let mut index = 0;
    let mut max_dist = 0.0;
    let last = points.len() - 1;
    for i in 1..last {
        let d = perpendicular_distance(&points[i], &points[0], &points[last]);
        if d > max_dist {
            max_dist = d;
            index = i;
        }
    }
    if max_dist > eps {
        rdp(&points[..=index], eps, out);
        rdp(&points[index..], eps, out);
    } else {
        out.push(points[last]);
    }
}

// Simplify a polyline.
fn ramer_douglas_peucker(points: Vec<(f64, f64)>, eps: f64) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    if !points.is_empty() && eps >= 0.0 {
        result.push(points[0]);
        rdp(&points, eps, &mut result);
    }
    result
}

// Adapter: Vec<Point> → Vec<(f64,f64)> → Vec<Point>
fn simplify_points(pts: &[Point], eps: f64) -> Vec<Point> {
    let tuples: Vec<(f64, f64)> = pts.iter().map(|p| (p.x, p.y)).collect();
    ramer_douglas_peucker(tuples, eps)
        .into_iter()
        .map(|(x, y)| Point::new(x, y))
        .collect()
}

// ——————————————————————————————————————————————————————————
//   Quadratic Bézier fitting
// ——————————————————————————————————————————————————————————

fn polyline_to_quadratic_bezier(points: &[Point], tol: f64) -> Path {
    let mut path = Path::new(points[0]);
    let n = points.len();
    if n < 2 {
        return path;
    }
    if n == 2 {
        path.line_to(points[1]);
        return path;
    }
    for window in points.windows(3) {
        let p0 = window[0];
        let p1 = window[1];
        let p2 = window[2];
        let dist =
            perpendicular_distance(&(p1.x, p1.y), &(p0.x, p0.y), &(p2.x, p2.y));
        if dist < tol {
            path.line_to(p2);
        } else {
            let ctrl = Point::new(
                2.0 * p1.x - 0.5 * (p0.x + p2.x),
                2.0 * p1.y - 0.5 * (p0.y + p2.y),
            );
            path.quadratic_bezier_to(ctrl, p2);
        }
    }
    path
}

// ——————————————————————————————————————————————————————————
//   Full trace pipeline
// ——————————————————————————————————————————————————————————

fn trace_bitmap(image: &GrayImage) -> Vec<Path> {
    println!("• Canny edge detection");
    let edged = edges::canny(image, 10.0, 50.0);

    println!("• Extracting contours");
    let raw = extract_contours(&edged);

    println!("• Simplify + Bézier‐fit");
    let eps = 1.0;      // RDP tolerance (px)
    let bez_tol = 0.5;  // flatness threshold

    raw.into_iter()
        .map(|poly| {
            let pts = poly.points_along_path(1);
            let simp = simplify_points(&pts, eps);
            polyline_to_quadratic_bezier(&simp, bez_tol)
        })
        .collect()
}

fn extract_contours(image: &GrayImage) -> Vec<Path> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut visited = vec![vec![false; height]; width];
    let mut contours = Vec::new();

    for (x, y, pixel) in image.enumerate_pixels() {
        if pixel == &Luma([255u8]) && !visited[x as usize][y as usize] {
            let pts = moore_neighbor_tracing(image, &mut visited, x, y);
            if !pts.is_empty() {
                let mut path = Path::new(pts[0]);
                for p in pts.iter().skip(1) {
                    path.line_to(*p);
                }
                contours.push(path);
            }
        }
    }
    contours
}

fn moore_neighbor_tracing(
    image: &GrayImage,
    visited: &mut Vec<Vec<bool>>,
    sx: u32,
    sy: u32,
) -> Vec<Point> {
    let mut contour = Vec::new();
    let mut current = (sx, sy);
    let mut backtrack = (0, 0);
    let mut backtracked = false;
    let neigh = [
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
    ];

    while !visited[current.0 as usize][current.1 as usize] || !backtracked {
        visited[current.0 as usize][current.1 as usize] = true;
        contour.push(Point::new(current.0 as f64, current.1 as f64));

        let start_i = if backtracked { (backtrack.0 + 1) % 8 } else { 0 };
        backtracked = false;
        let mut found = false;

        for i in start_i..8 {
            let nx = current.0 as i32 + neigh[i].0;
            let ny = current.1 as i32 + neigh[i].1;
            if nx >= 0
                && ny >= 0
                && nx < image.width() as i32
                && ny < image.height() as i32
                && image.get_pixel(nx as u32, ny as u32) == &Luma([255u8])
                && !visited[nx as usize][ny as usize]
            {
                backtrack = (i, i);
                current = (nx as u32, ny as u32);
                found = true;
                break;
            }
            if i == 7 {
                backtracked = true;
                backtrack = (backtrack.1, (backtrack.1 + 1) % 8);
            }
        }
        if !found && backtracked {
            break;
        }
    }
    contour
}

fn export_vector_data(
    paths: &[Path],
    file_name: &str,
    width: u32,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut doc = Document::new()
        .set("viewBox", (0, 0, width, height))
        .set("width", width)
        .set("height", height);

    let bg = Rectangle::new()
        .set("width", "100%")
        .set("height", "100%")
        .set("fill", "white");
    doc = doc.add(bg);

    for path in paths {
        let mut data = Data::new().move_to((path.start.x, path.start.y));
        for seg in &path.segments {
            match seg {
                Segment::Line(p) => {
                    data = data.line_to((p.x, p.y));
                }
                Segment::QuadraticBezier(c, p) => {
                    data = data.quadratic_curve_to(((c.x, c.y), (p.x, p.y)));
                }
            }
        }
        let svg_path = SvgPath::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 1)
            .set("d", data);
        doc = doc.add(svg_path);
    }

    let mut file = File::create(file_name)?;
    file.write_all(doc.to_string().as_bytes())?;
    Ok(())
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}
impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

pub enum Segment {
    Line(Point),
    QuadraticBezier(Point, Point),
}

pub struct Path {
    pub start: Point,
    pub segments: Vec<Segment>,
}
impl Path {
    pub fn new(start: Point) -> Self {
        Self {
            start,
            segments: Vec::new(),
        }
    }
    pub fn line_to(&mut self, end: Point) {
        self.segments.push(Segment::Line(end));
    }
    pub fn quadratic_bezier_to(&mut self, control: Point, end: Point) {
        self.segments.push(Segment::QuadraticBezier(control, end));
    }
    pub fn points_along_path(&self, resolution: usize) -> Vec<Point> {
        let mut pts = vec![self.start];
        for seg in &self.segments {
            match seg {
                Segment::Line(end) => {
                    pts.push(*end);
                }
                Segment::QuadraticBezier(ctrl, end) => {
                    for i in 1..=resolution {
                        let t = i as f64 / resolution as f64;
                        let p = quadratic_bezier(self.start, *ctrl, *end, t);
                        pts.push(p);
                    }
                }
            }
        }
        pts
    }
}

fn quadratic_bezier(p0: Point, p1: Point, p2: Point, t: f64) -> Point {
    let x = (1.0 - t).powi(2) * p0.x
        + 2.0 * (1.0 - t) * t * p1.x
        + t.powi(2) * p2.x;
    let y = (1.0 - t).powi(2) * p0.y
        + 2.0 * (1.0 - t) * t * p1.y
        + t.powi(2) * p2.y;
    Point::new(x, y)
}
