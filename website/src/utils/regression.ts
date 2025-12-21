import { levenbergMarquardt } from "ml-levenberg-marquardt";

export interface RegressionPoint {
  x: number;
  y: number;
}

export interface RegressionResult {
  slope: number;
  intercept: number;
  r2: number;
}

/**
 * Result from power law with offset fitting: L = intercept * x^slope + asymptote
 * where asymptote is the irreducible loss floor and slope < 0.
 *
 * Field names and sign convention match RegressionResult for consistency:
 * - intercept: scaling coefficient
 * - slope: scaling exponent (negative, matching log-log linear convention)
 * - asymptote: irreducible loss floor (not present in simple power law)
 */
export interface PowerLawResult {
  intercept: number;
  slope: number;
  asymptote: number;
  r2: number;
}

/**
 * Computes the least-squares linear regression for the supplied points.
 * Returns null when fewer than two valid points are provided or when all x values are identical.
 */
export function computeLinearRegression(
  points: RegressionPoint[]
): RegressionResult | null {
  if (points.length < 2) {
    return null;
  }

  const sumX = points.reduce((sum, point) => sum + point.x, 0);
  const sumY = points.reduce((sum, point) => sum + point.y, 0);
  const sumXY = points.reduce((sum, point) => sum + point.x * point.y, 0);
  const sumX2 = points.reduce((sum, point) => sum + point.x * point.x, 0);
  const n = points.length;

  const denominator = n * sumX2 - sumX * sumX;
  if (denominator === 0) {
    return null;
  }

  const slope = (n * sumXY - sumX * sumY) / denominator;
  const intercept = (sumY - slope * sumX) / n;

  const yMean = sumY / n;
  const ssRes = points.reduce((sum, point) => {
    const predicted = intercept + slope * point.x;
    const residual = point.y - predicted;
    return sum + residual * residual;
  }, 0);
  const ssTot = points.reduce((sum, point) => {
    const diff = point.y - yMean;
    return sum + diff * diff;
  }, 0);
  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

  return { slope, intercept, r2 };
}

/**
 * Fits a power law with irreducible loss: L = a * x^(-b) + c
 *
 * This is the Chinchilla-style scaling law form that accounts for
 * the entropy floor of the data. Uses Levenberg-Marquardt nonlinear
 * least squares optimization.
 *
 * @param points - Array of {x, y} where x is compute (FLOPs) and y is loss (BPB)
 * @returns PowerLawResult or null if fitting fails
 */
export function computePowerLawWithOffset(
  points: RegressionPoint[]
): PowerLawResult | null {
  // Need at least 3 points to fit 3 parameters (a, b, c)
  if (points.length < 3) {
    return null;
  }

  // Filter and sort points
  const validPoints = points
    .filter((p) => p.x > 0 && Number.isFinite(p.x) && Number.isFinite(p.y))
    .sort((a, b) => a.x - b.x);

  if (validPoints.length < 3) {
    return null;
  }

  const xData = validPoints.map((p) => p.x);
  const yData = validPoints.map((p) => p.y);

  // Estimate initial parameters from the data
  const yMin = Math.min(...yData);
  const yMax = Math.max(...yData);
  const xMin = Math.min(...xData);
  const xMax = Math.max(...xData);

  // Initial guess for asymptote

  const ENTROPY_FLOOR_ESTIMATE = 0.7;
  const cInit = Math.min(ENTROPY_FLOOR_ESTIMATE, yMin * 0.7);

  // Use log-log regression on (y - cInit) to estimate a and b
  const adjustedPoints = validPoints
    .map((p) => ({ x: p.x, y: p.y - cInit }))
    .filter((p) => p.y > 0);

  let aInit = 1;
  let bInit = 0.1;

  if (adjustedPoints.length >= 2) {
    const logPoints = adjustedPoints.map((p) => ({
      x: Math.log(p.x),
      y: Math.log(p.y),
    }));
    const logReg = computeLinearRegression(logPoints);
    if (logReg) {
      // log(y - c) = log(a) + slope * log(x)
      // slope should be negative for scaling laws, so b = -slope
      bInit = Math.max(0.01, -logReg.slope);
      aInit = Math.exp(logReg.intercept);
    }
  }

  // Define the parametric function: f(x) = intercept * x^(-slope) + asymptote
  // Parameters: [intercept, slope, asymptote]
  const parametricFunction =
    ([intercept, slope, asymptote]: number[]) =>
    (x: number) => {
      return intercept * Math.pow(x, -slope) + asymptote;
    };

  const asymptoteMax = yMin * 0.999;

  try {
    const result = levenbergMarquardt(
      { x: xData, y: yData },
      parametricFunction,
      {
        initialValues: [aInit, bInit, cInit],
        damping: 1.5,
        gradientDifference: 1e-6,
        maxIterations: 200,
        errorTolerance: 1e-8,
        // Constraints: intercept > 0, slope > 0, asymptote >= 0
        minValues: [1e-10, 1e-6, 0],
        maxValues: [1e30, 2, asymptoteMax],
      }
    );

    const [intercept, slope, asymptote] = result.parameterValues;

    // Validate the result
    if (
      !Number.isFinite(intercept) ||
      !Number.isFinite(slope) ||
      !Number.isFinite(asymptote)
    ) {
      return null;
    }
    if (intercept <= 0 || slope <= 0 || asymptote < 0) {
      return null;
    }

    // Calculate R² for the fit
    const yMean = yData.reduce((sum, y) => sum + y, 0) / yData.length;
    const predictFn = parametricFunction([intercept, slope, asymptote]);

    const ssRes = xData.reduce((sum, x, i) => {
      const predicted = predictFn(x);
      const residual = yData[i] - predicted;
      return sum + residual * residual;
    }, 0);

    const ssTot = yData.reduce((sum, y) => {
      const diff = y - yMean;
      return sum + diff * diff;
    }, 0);

    const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;

    // Return slope as negative to match log-log linear convention
    return { intercept, slope: -slope, asymptote, r2 };
  } catch {
    // Levenberg-Marquardt can fail to converge
    return null;
  }
}

/**
 * Predict loss at a given compute value using the power law with offset.
 * Formula: L = intercept * x^slope + asymptote (slope is negative)
 */
export function predictWithPowerLaw(result: PowerLawResult, x: number): number {
  return result.intercept * Math.pow(x, result.slope) + result.asymptote;
}
