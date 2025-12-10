export function getScalingGroupName(runName: string): string {
  const parts = runName.split('/').filter(Boolean);
  if (parts.length <= 1) {
    return runName;
  }
  // Group scaling runs by their shared folder (everything but the leaf)
  return parts.slice(0, -1).join('/');
}

