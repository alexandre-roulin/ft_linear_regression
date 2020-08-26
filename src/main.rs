extern crate flot;
use std::error::Error;
#[derive(Default)]
struct LinearRegression {
    vec_km: Vec<f64>,
    vec_price: Vec<f64>,
    learning_rate: f64,
    num_iteration: usize,
    theta0: f64,
    theta1: f64,
}

impl LinearRegression {
    fn new(vec_km: Vec<f64>, vec_price: Vec<f64>) -> Self {
        Self {
            vec_km,
            vec_price,
            learning_rate: 0.01_f64,
            num_iteration: 2000,
            theta0: 0_f64,
            theta1: 0_f64,
        }
    }

    #[allow(dead_code)]
    fn cost(&self) -> f64 {
        let mut total_error = 0.0;
        for index in 0..self.vec_km.len() {
            total_error += (self.vec_price[index]
                - (self.theta1 * self.vec_km[index] + self.theta0))
                .powf(2.0);
        }
        total_error / self.vec_km.len() as f64
    }

    fn gradient_descent(&mut self) {
        let len = self.vec_km.len() as f64;

        for _ in 0..self.num_iteration {
            let mut gradient0 = 0.0;
            let mut gradient1 = 0.0;
            for idx in 0..self.vec_km.len() {
                gradient0 += (self.theta0 + (self.theta1 * self.vec_km[idx])) - self.vec_price[idx];
                gradient1 += ((self.theta0 + (self.theta1 * self.vec_km[idx]))
                    - self.vec_price[idx])
                    * self.vec_km[idx];
            }
            self.theta0 -= self.learning_rate * 1_f64 / len * gradient0;
            self.theta1 -= self.learning_rate * 1_f64 / len * gradient1;
        }
    }
}
fn read_csv() -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    // Build the CSV reader and iterate over each record.
    let rdr = csv::Reader::from_path("./data.csv")?;
    let mut vec_km = vec![];
    let mut vec_price = vec![];
    for result in rdr.into_records() {
        // The iterator yields Result<StringRecord, Error>, so we check the
        // error here.
        let record = result?;
        vec_km.push(record[0].parse::<f64>()?);
        vec_price.push(record[1].parse::<f64>()?);
    }
    Ok((vec_km, vec_price))
}

fn main() {
    //Read CSV
    let (vec_km, vec_price) = read_csv().unwrap();
    //Normalize Vector to protect overflow data
    let vec_km2 = vec_km.iter().map(|x| x / 10000_f64).collect::<Vec<f64>>();
    let mut lr = LinearRegression::new(vec_km2, vec_price.clone());

    // Train
    lr.gradient_descent();

    //Resolve the theta1 to be as it should be
    lr.theta1 /= 10000_f64;

    //Get Max and Min to create the linear regression
    let min = vec_km
        .iter()
        .min_by(|v1, v2| (**v1 as i32).cmp(&(**v2 as i32)))
        .unwrap()
        .clone();
    let max = vec_km
        .iter()
        .max_by(|v1, v2| (**v1 as i32).cmp(&(**v2 as i32)))
        .unwrap()
        .clone();

    //Linear Regression
    let line_data = vec![
        (min, lr.theta1 * min + lr.theta0),
        (max, lr.theta1 * max + lr.theta0),
    ];
    let page = flot::Page::new("Ft_Linear_Regression");

    //Merge the vector
    let vec = vec_km
        .into_iter()
        .zip(vec_price)
        .collect::<Vec<(f64, f64)>>(); 
    // Some Prediction
    let prediction = vec![
        (42000_f64, lr.theta1 * 42000_f64 + lr.theta0),
        (100000_f64, lr.theta1 * 100000_f64 + lr.theta0),
        (38600_f64, lr.theta1 * 38600_f64 + lr.theta0),
    ];
    //bottom
    let plot = page.plot("Price Y and Mileage in X");
    plot.points("Initial Data", vec);
    plot.points("Prediction", prediction).color("red");
    plot.lines("Linear regression", line_data)
        .fill(0.0)
        .line_width(1);
    page.render("linear_regression.html").expect("error render : ");
}
