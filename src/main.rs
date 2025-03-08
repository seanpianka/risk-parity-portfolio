mod test;

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::Read;
use time::OffsetDateTime;
use yahoo_finance_api as yahoo;

const NUM_YEARS: i64 = 5;
const WINDOW_SIZE: usize = 12;

#[derive(Debug, Serialize, Deserialize)]
struct AssetClasses {
    equities: Vec<String>,
    commodities: Vec<String>,
    bonds: Vec<String>,
}

#[derive(Debug, Clone, Copy)]
struct Price {
    value: f64,
    date: OffsetDateTime,
}

#[derive(Debug, Clone, Copy)]
struct Return {
    value: f64,
    start_date: OffsetDateTime,
    end_date: OffsetDateTime,
}

fn load_yaml_config(path: &str) -> AssetClasses {
    let mut file = File::open(path).expect("Failed to open YAML file");
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .expect("Failed to read YAML file");
    serde_yaml::from_str(&contents).expect("Failed to parse YAML")
}

/// Fetch historical monthly closing prices for a given stock symbol (last 5 years)
///
/// Ordered earliest to latest.
async fn fetch_monthly_historical_prices(symbol: &str) -> Result<Vec<Price>, Box<dyn Error>> {
    let provider = yahoo::YahooConnector::new()?;

    // Define the time range (last 5 years)
    let end = OffsetDateTime::now_utc();
    let start = end - time::Duration::days(NUM_YEARS * 365);

    // Fetch data from Yahoo Finance
    let response = provider
        .get_quote_history_interval(symbol, start, end, "1d")
        .await
        .unwrap();
    let quotes = response.quotes().unwrap();

    // Extract (date, closing price) pairs
    let mut prices: Vec<(OffsetDateTime, f64)> = quotes
        .iter()
        .filter_map(|q| {
            Some((
                OffsetDateTime::from_unix_timestamp(q.timestamp as i64).unwrap(),
                f64::from(q.close),
            ))
        })
        .collect();

    // Sort by date to ensure correct order
    prices.sort_by_key(|&(date, _)| date.unix_timestamp());

    // **Resample to the last trading day of each month**
    let mut monthly_prices = Vec::new();
    let mut last_seen_month: Option<u8> = None;

    for (date, price) in prices.iter().rev() {
        let month = date.month() as u8;
        if last_seen_month != Some(month) {
            monthly_prices.push((*date, *price));
            // println!("found price for date: {}", date.date());
            last_seen_month = Some(month);
        }
    }

    // Reverse to maintain chronological order
    monthly_prices.reverse();

    if monthly_prices.is_empty() {
        return Err(format!("No historical data found for {}", symbol).into());
    }
    // println!(
    //     "monthly price data for ticker {}: {:#?}",
    //     symbol, monthly_prices
    // );

    Ok(monthly_prices
        .into_iter()
        .map(|(dt, p)| Price { value: p, date: dt })
        .collect())
}

/// Calculate the monthly returns from a series of prices.
///
/// A "log return" is a method of calculating the percentage change of an asset
/// price that takes into account compounding effects by using the natural
/// logarithm of the price ratio, while a "simple return" is a basic calculation
/// of percentage change, simply subtracting the initial price from the final
/// price and dividing by the initial price, without considering compounding over
/// time; log returns are generally preferred for time series analysis due to
/// their ability to accurately represent compounding, while simple returns are
/// easier to understand for single-period calculations.
fn compute_monthly_returns(prices: &[Price]) -> Vec<Return> {
    prices
        .windows(2)
        .map(|w| Return {
            value: (w[1].value / w[0].value).ln(),
            start_date: w[0].date,
            end_date: w[1].date,
        })
        .collect()
}

/// Calculates the volatility/standard deviation of the given returns.
fn compute_volatility(returns: &[Return]) -> (f64, OffsetDateTime) {
    let mean: f64 = returns.iter().map(|r| r.value).sum::<f64>() / returns.len() as f64;
    let variance: f64 = returns
        .iter()
        .map(|r| (r.value - mean).powi(2))
        .sum::<f64>()
        / (returns.len() - 1) as f64;

    let last_date = returns.last().unwrap().end_date; // Last valid date

    (variance.sqrt(), last_date) // Standard deviation as volatility
}

fn compute_rolling_volatility(
    returns: &[Return],
    window_size: usize,
) -> Vec<(f64, OffsetDateTime)> {
    let mut volatilities = returns
        .windows(window_size)
        .map(|w| compute_volatility(w))
        .collect();
    volatilities
}

fn compute_covariance(asset1: &[Return], asset2: &[Return]) -> (f64, OffsetDateTime) {
    let mean1 = asset1.iter().map(|r| r.value).sum::<f64>() / asset1.len() as f64;
    let mean2 = asset2.iter().map(|r| r.value).sum::<f64>() / asset2.len() as f64;

    let covariance = asset1
        .iter()
        .zip(asset2.iter())
        .map(|(&r1, &r2)| (r1.value - mean1) * (r2.value - mean2))
        .sum::<f64>()
        / (asset1.len() as f64 - 1.0);
    let last_date = asset1.last().unwrap().end_date;

    (covariance, last_date)
}

fn compute_rolling_covariance(
    asset1: &[Return],
    asset2: &[Return],
    window_size: usize,
) -> Vec<(f64, OffsetDateTime)> {
    asset1
        .windows(window_size)
        .zip(asset2.windows(window_size))
        .map(|(w1, w2)| compute_covariance(w1, w2))
        .collect()
}

fn compute_static_correlation(
    asset1: &[Return],
    asset2: &[Return],
    lookback_months: usize,
) -> (f64, OffsetDateTime) {
    let len = asset1.len().min(asset2.len());

    if len < lookback_months {
        return (f64::NAN, OffsetDateTime::now_utc()); // Not enough data
    }

    let start = len - lookback_months;
    let slice1 = &asset1[start..];
    let slice2 = &asset2[start..];

    let mean1 = slice1.iter().map(|r| r.value).sum::<f64>() / slice1.len() as f64;
    let mean2 = slice2.iter().map(|r| r.value).sum::<f64>() / slice2.len() as f64;

    let numerator: f64 = slice1
        .iter()
        .zip(slice2.iter())
        .map(|(&r1, &r2)| (r1.value - mean1) * (r2.value - mean2))
        .sum();

    let denom1: f64 = slice1
        .iter()
        .map(|&r| (r.value - mean1).powi(2))
        .sum::<f64>()
        .sqrt();
    let denom2: f64 = slice2
        .iter()
        .map(|&r| (r.value - mean2).powi(2))
        .sum::<f64>()
        .sqrt();

    let last_date = slice1.last().unwrap().end_date; // Last valid date

    if denom1 > 0.0 && denom2 > 0.0 {
        (numerator / (denom1 * denom2), last_date)
    } else {
        (f64::NAN, last_date)
    }
}

fn compute_rolling_correlation_matrix(
    returns: &[Vec<Return>],
    window_size: usize,
) -> Vec<(DMatrix<f64>, OffsetDateTime)> {
    let num_assets = returns.len();
    let num_periods = returns[0].len() - window_size + 1;

    let mut correlation_matrices = Vec::new();

    for t in 0..num_periods {
        let mut matrix = DMatrix::<f64>::identity(num_assets, num_assets);

        let last_date = returns[0][t + window_size - 1].end_date; // Date of last return in window

        for i in 0..num_assets {
            for j in i..num_assets {
                if i == j {
                    continue;
                }
                let vol_i = compute_rolling_volatility(&returns[i], window_size)[t].0;
                let vol_j = compute_rolling_volatility(&returns[j], window_size)[t].0;
                let cov_ij = compute_rolling_covariance(&returns[i], &returns[j], window_size)[t].0;

                let correlation = if vol_i > 0.0 && vol_j > 0.0 {
                    cov_ij / (vol_i * vol_j)
                } else {
                    f64::NAN
                };

                matrix[(i, j)] = correlation;
                matrix[(j, i)] = correlation; // Ensure symmetry
            }
        }

        correlation_matrices.push((matrix, last_date));
    }

    correlation_matrices
}
// fn compute_correlation_matrix(
//     cov_matrix: &DMatrix<f64>,
//     volatilities: &Vec<f64>,
//     returns_matrix: &[Vec<f64>],
// ) -> DMatrix<f64> {
//     let mut correlation_matrix = cov_matrix.clone();
//
//     for i in 0..volatilities.len() {
//         for j in 0..volatilities.len() {
//             if i == j {
//                 correlation_matrix[(i, j)] = 1.0;
//             } else {
//                 correlation_matrix[(i, j)] =
//                     cov_matrix[(i, j)] / (volatilities[i] * volatilities[j]);
//             }
//         }
//     }
//
//     correlation_matrix
// }

fn compute_risk_parity_weights(
    volatilities: &[f64],
    correlation_matrix: &DMatrix<f64>,
) -> Vec<f64> {
    // let inv_vol: Vec<f64> = volatilities.iter().map(|&v| 1.0 / v).collect();
    // let total_weight: f64 = inv_vol.iter().sum();
    // inv_vol.iter().map(|&w| w / total_weight).collect()
    let num_assets = volatilities.len();
    let mut risk_contributions = vec![0.0; num_assets];

    for i in 0..num_assets {
        let mut risk_contribution = 0.0;
        for j in 0..num_assets {
            risk_contribution += volatilities[i] * volatilities[j] * correlation_matrix[(i, j)];
        }
        risk_contributions[i] = risk_contribution.sqrt();
    }

    let total_risk: f64 = risk_contributions.iter().sum();
    risk_contributions.iter().map(|w| w / total_risk).collect()
}

async fn calculate_correlation_adjusted_portfolio_weights(
    asset_classes: [(&str, &Vec<String>); 3],
) -> Result<(), Box<dyn Error>> {
    for (category, assets) in asset_classes.iter() {
        println!("Fetching data for {}", category);

        let mut all_returns = Vec::new();
        let mut volatilities = Vec::new();

        for asset in assets.iter() {
            let prices = fetch_monthly_historical_prices(asset).await?;
            let returns = compute_monthly_returns(&prices);
            let (vol, dt) = compute_volatility(&returns);

            println!("Asset: {} | Volatility: {:.4}", asset, vol);
            all_returns.push(returns);
            volatilities.push(vol);
        }
        // if volatilities.len() > 1 {
        //     println!("GLD Volatility: {:.6}", volatilities[0]);
        //     println!("PDBC Volatility: {:.6}", volatilities[1]);
        // }

        let correlation_matrix = compute_rolling_correlation_matrix(&all_returns, WINDOW_SIZE);
        let (last_cm, dt) = correlation_matrix.last().unwrap();
        let risk_parity_weights = compute_risk_parity_weights(&volatilities, &last_cm);

        println!(
            "Risk Parity Weights for {}: {:?}",
            category, risk_parity_weights
        );
        println!("{} Correlation Matrix for {}:\n{}", dt, category, last_cm);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let config = load_yaml_config("tickers.yaml");

    let asset_classes = [
        ("Equities", &config.equities),
        ("Commodities", &config.commodities),
        ("Bonds", &config.bonds),
    ];

    // let gld_prices = fetch_historical_data("gld").await?;
    // let pdbc_prices = fetch_historical_data("pdbc").await?;
    // let gld_returns = compute_monthly_returns(&gld_prices);
    // let pdbc_returns = compute_monthly_returns(&pdbc_prices);
    //
    // let rolling_correlation = compute_rolling_correlation(&gld_returns, &pdbc_returns, 36);
    // for (idx, corr) in rolling_correlation {
    //     println!("Index {}: {:.4}", idx, corr);
    // }

    calculate_correlation_adjusted_portfolio_weights(asset_classes).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_get_prices() {
        let gld_prices = fetch_monthly_historical_prices("GLD").await.unwrap();
        let pdbc_prices = fetch_monthly_historical_prices("PDBC").await.unwrap();
        let gld_returns = compute_monthly_returns(&gld_prices);
        let pdbc_returns = compute_monthly_returns(&pdbc_prices);
        // println!("GLD Returns: {:?}", gld_returns);
        // println!("PDBC Returns: {:?}", pdbc_returns);
        println!("ticker,price,date");
        for r in gld_prices {
            println!("GLD,{},{}", r.value, r.date.date());
        }
        for r in pdbc_prices {
            println!("PDBC,{},{}", r.value, r.date.date());
        }
    }

    #[tokio::test]
    async fn test_compute_rolling_correlation_matrix_gld_pdbc() {
        let asset1 = "SPY";
        let asset2 = "QQQ";
        let asset1_prices = fetch_monthly_historical_prices(asset1).await.unwrap();
        let asset2_prices = fetch_monthly_historical_prices(asset2).await.unwrap();
        let asset1_returns = compute_monthly_returns(&asset1_prices);
        let asset2_returns = compute_monthly_returns(&asset2_prices);
        assert_eq!(asset1_returns.len(), asset2_returns.len());

        let window_size = 12;
        let returns = vec![asset1_returns.clone(), asset2_returns.clone()];
        let correlation_matrices = compute_rolling_correlation_matrix(&returns, window_size);

        for (cm, dt) in &correlation_matrices {
            let correlation = &cm[(0, 1)];
            println!(
                "{}-{} Correlation on {}: {:.4}",
                asset1,
                asset2,
                dt.date(),
                correlation
            );
        }
        println!(
            "num returns: {}, num matricies: {}",
            asset1_returns.len(),
            correlation_matrices.len()
        );

        // let last_correlation_matrix = correlation_matrices.last().unwrap();
        // let gld_pdbc_correlation = last_correlation_matrix[(0, 1)];

        // let expected_correlation = -0.22;
        // assert!(
        //     (gld_pdbc_correlation - expected_correlation).abs() < 0.01,
        //     "Expected correlation {:.4}, got {:.4}",
        //     expected_correlation,
        //     gld_pdbc_correlation
        // );
    }

    #[test]
    fn test_compute_monthly_returns() {
        let base_date = OffsetDateTime::now_utc();
        let prices = vec![
            Price {
                value: 100.0,
                date: base_date,
            },
            Price {
                value: 105.0,
                date: base_date + time::Duration::days(30),
            },
            Price {
                value: 103.0,
                date: base_date + time::Duration::days(60),
            },
            Price {
                value: 108.0,
                date: base_date + time::Duration::days(90),
            },
        ];

        let expected_returns = vec![0.04879, -0.01923, 0.04879]; // Log returns

        let actual_returns = compute_monthly_returns(&prices);

        assert_eq!(actual_returns.len(), expected_returns.len());
        for (a, e) in actual_returns.iter().zip(expected_returns.iter()) {
            assert!((a.value - e).abs() < 1e-4);
        }
    }
    #[test]
    fn test_compute_volatility() {
        let base_date = OffsetDateTime::now_utc();
        let returns = vec![
            Return {
                value: 0.01,
                start_date: base_date,
                end_date: base_date + time::Duration::days(30),
            },
            Return {
                value: -0.02,
                start_date: base_date + time::Duration::days(30),
                end_date: base_date + time::Duration::days(60),
            },
            Return {
                value: 0.015,
                start_date: base_date + time::Duration::days(60),
                end_date: base_date + time::Duration::days(90),
            },
            Return {
                value: 0.03,
                start_date: base_date + time::Duration::days(90),
                end_date: base_date + time::Duration::days(120),
            },
            Return {
                value: -0.01,
                start_date: base_date + time::Duration::days(120),
                end_date: base_date + time::Duration::days(150),
            },
            Return {
                value: 0.005,
                start_date: base_date + time::Duration::days(150),
                end_date: base_date + time::Duration::days(180),
            },
        ];

        let (actual_vol, last_date) = compute_volatility(&returns);

        let expected_vol = 0.017748; // Precomputed expected result
        assert!((actual_vol - expected_vol).abs() < 1e-6);
        assert_eq!(last_date, returns.last().unwrap().end_date);
    }
    #[test]
    fn test_compute_rolling_volatility() {
        let base_date = OffsetDateTime::now_utc();
        let returns = vec![
            Return {
                value: 0.01,
                start_date: base_date,
                end_date: base_date + time::Duration::days(30),
            },
            Return {
                value: -0.02,
                start_date: base_date + time::Duration::days(30),
                end_date: base_date + time::Duration::days(60),
            },
            Return {
                value: 0.015,
                start_date: base_date + time::Duration::days(60),
                end_date: base_date + time::Duration::days(90),
            },
            Return {
                value: 0.03,
                start_date: base_date + time::Duration::days(90),
                end_date: base_date + time::Duration::days(120),
            },
            Return {
                value: -0.01,
                start_date: base_date + time::Duration::days(120),
                end_date: base_date + time::Duration::days(150),
            },
            Return {
                value: 0.005,
                start_date: base_date + time::Duration::days(150),
                end_date: base_date + time::Duration::days(180),
            },
        ];

        let window_size = 3;
        let rolling_vol = compute_rolling_volatility(&returns, window_size);

        assert_eq!(rolling_vol.len(), returns.len() - window_size + 1);
        assert!(rolling_vol.iter().any(|&(v, _)| !v.is_nan())); // Should contain valid volatilities
    }

    #[test]
    fn test_compute_rolling_covariance() {
        let base_date = OffsetDateTime::now_utc();

        let asset1 = vec![
            Return {
                value: 0.01,
                start_date: base_date,
                end_date: base_date + time::Duration::days(30),
            },
            Return {
                value: -0.02,
                start_date: base_date + time::Duration::days(30),
                end_date: base_date + time::Duration::days(60),
            },
            Return {
                value: 0.015,
                start_date: base_date + time::Duration::days(60),
                end_date: base_date + time::Duration::days(90),
            },
            Return {
                value: 0.03,
                start_date: base_date + time::Duration::days(90),
                end_date: base_date + time::Duration::days(120),
            },
            Return {
                value: -0.01,
                start_date: base_date + time::Duration::days(120),
                end_date: base_date + time::Duration::days(150),
            },
            Return {
                value: 0.005,
                start_date: base_date + time::Duration::days(150),
                end_date: base_date + time::Duration::days(180),
            },
        ];

        let asset2 = vec![
            Return {
                value: -0.01,
                start_date: base_date,
                end_date: base_date + time::Duration::days(30),
            },
            Return {
                value: 0.005,
                start_date: base_date + time::Duration::days(30),
                end_date: base_date + time::Duration::days(60),
            },
            Return {
                value: 0.02,
                start_date: base_date + time::Duration::days(60),
                end_date: base_date + time::Duration::days(90),
            },
            Return {
                value: -0.015,
                start_date: base_date + time::Duration::days(90),
                end_date: base_date + time::Duration::days(120),
            },
            Return {
                value: 0.01,
                start_date: base_date + time::Duration::days(120),
                end_date: base_date + time::Duration::days(150),
            },
            Return {
                value: 0.03,
                start_date: base_date + time::Duration::days(150),
                end_date: base_date + time::Duration::days(180),
            },
        ];

        let window_size = 3;
        let rolling_cov = compute_rolling_covariance(&asset1, &asset2, window_size);

        assert_eq!(rolling_cov.len(), asset1.len() - window_size + 1);

        for (cov, dt) in &rolling_cov {
            assert!(!cov.is_nan()); // Ensure valid covariance values
        }
    }
    #[test]
    fn test_compute_rolling_correlation_matrix() {
        let base_date = OffsetDateTime::now_utc();
        let returns = vec![
            vec![
                Return {
                    value: 0.02,
                    start_date: base_date,
                    end_date: base_date + time::Duration::days(30),
                },
                Return {
                    value: -0.01,
                    start_date: base_date + time::Duration::days(30),
                    end_date: base_date + time::Duration::days(60),
                },
                Return {
                    value: 0.015,
                    start_date: base_date + time::Duration::days(60),
                    end_date: base_date + time::Duration::days(90),
                },
                Return {
                    value: 0.03,
                    start_date: base_date + time::Duration::days(90),
                    end_date: base_date + time::Duration::days(120),
                },
                Return {
                    value: -0.02,
                    start_date: base_date + time::Duration::days(120),
                    end_date: base_date + time::Duration::days(150),
                },
                Return {
                    value: 0.01,
                    start_date: base_date + time::Duration::days(150),
                    end_date: base_date + time::Duration::days(180),
                },
            ],
            vec![
                Return {
                    value: -0.01,
                    start_date: base_date,
                    end_date: base_date + time::Duration::days(30),
                },
                Return {
                    value: 0.005,
                    start_date: base_date + time::Duration::days(30),
                    end_date: base_date + time::Duration::days(60),
                },
                Return {
                    value: 0.02,
                    start_date: base_date + time::Duration::days(60),
                    end_date: base_date + time::Duration::days(90),
                },
                Return {
                    value: -0.015,
                    start_date: base_date + time::Duration::days(90),
                    end_date: base_date + time::Duration::days(120),
                },
                Return {
                    value: 0.01,
                    start_date: base_date + time::Duration::days(120),
                    end_date: base_date + time::Duration::days(150),
                },
                Return {
                    value: 0.03,
                    start_date: base_date + time::Duration::days(150),
                    end_date: base_date + time::Duration::days(180),
                },
            ],
        ];

        let window_size = 3;
        let correlation_matrices = compute_rolling_correlation_matrix(&returns, window_size);

        assert_eq!(
            correlation_matrices.len(),
            returns[0].len() - window_size + 1
        );
        assert_eq!(correlation_matrices[0].0.nrows(), 2);
        assert_eq!(correlation_matrices[0].0.ncols(), 2);

        // Ensure last matrix corresponds to the last date in returns
        let last_expected_date = returns[0].last().unwrap().end_date;
        assert_eq!(correlation_matrices.last().unwrap().1, last_expected_date);
    }

    #[test]
    fn test_compute_risk_parity_weights() {
        let volatilities = vec![0.02, 0.03, 0.01];

        let correlation_matrix =
            DMatrix::from_vec(3, 3, vec![1.0, 0.5, 0.3, 0.5, 1.0, 0.4, 0.3, 0.4, 1.0]);

        let weights = compute_risk_parity_weights(&volatilities, &correlation_matrix);

        assert_eq!(weights.len(), volatilities.len());

        let sum_weights: f64 = weights.iter().sum();
        assert!((sum_weights - 1.0).abs() < 1e-6); // Ensure weights sum to 1

        for w in &weights {
            assert!(*w > 0.0 && *w <= 1.0); // Ensure valid weight values
        }
    }
}
