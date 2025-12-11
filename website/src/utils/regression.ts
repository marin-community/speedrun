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
 * Computes the least-squares linear regression for the supplied points.
 * Returns null when fewer than two valid points are provided or when all x values are identical.
 */
export function computeLinearRegression(points: RegressionPoint[]): RegressionResult | null {
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
