/**
 * Sprite sheet texture utilities for testing animation functions.
 */

/**
 * Create a sprite sheet texture for testing animation functions.
 * Creates a grid where each cell has a distinct color.
 * @param device - GPU device
 * @param cols - Number of columns in the grid
 * @param rows - Number of rows in the grid
 * @param size - Total texture size (default 256x256)
 */
export function createSpriteSheetTexture(
  device: GPUDevice,
  cols: number,
  rows: number,
  size = 256,
): GPUTexture {
  const cellWidth = size / cols;
  const cellHeight = size / rows;
  // Use Uint8Array for rgba8unorm format (filterable texture)
  const data = new Uint8Array(size * size * 4);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cellX = Math.floor(x / cellWidth);
      const cellY = Math.floor(y / cellHeight);
      const frameIndex = cellY * cols + cellX;

      // Generate distinct colors using a simple gradient per frame
      // Each frame gets a different primary color combination
      const r = ((frameIndex * 37) % 256) / 255; // Pseudo-random but deterministic
      const g = ((frameIndex * 113) % 256) / 255;
      const b = ((frameIndex * 191) % 256) / 255;

      const pixelIndex = (y * size + x) * 4;
      // Convert from [0,1] float to [0,255] uint8
      data[pixelIndex] = Math.floor(r * 255);
      data[pixelIndex + 1] = Math.floor(g * 255);
      data[pixelIndex + 2] = Math.floor(b * 255);
      data[pixelIndex + 3] = 255;
    }
  }

  const texture = device.createTexture({
    size: [size, size, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  device.queue.writeTexture(
    { texture },
    data,
    { bytesPerRow: size * 4, rowsPerImage: size },
    [size, size, 1],
  );

  return texture;
}

/**
 * Create a simple sprite sheet where each frame's color encodes its frame number.
 * Frame N has color (N/totalFrames, 0, 0, 1) for easy verification.
 * @param device - GPU device
 * @param cols - Number of columns in the grid
 * @param rows - Number of rows in the grid
 * @param size - Total texture size (default 256x256)
 */
export function createSimpleSpriteSheet(
  device: GPUDevice,
  cols: number,
  rows: number,
  size = 256,
): GPUTexture {
  const totalFrames = cols * rows;
  const cellWidth = size / cols;
  const cellHeight = size / rows;
  const data = new Uint8Array(size * size * 4);

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const cellX = Math.floor(x / cellWidth);
      const cellY = Math.floor(y / cellHeight);
      const frameIndex = cellY * cols + cellX;

      const pixelIndex = (y * size + x) * 4;
      // Encode frame number in red channel: frame N -> red = N/totalFrames
      data[pixelIndex] = Math.floor((frameIndex / totalFrames) * 255);
      data[pixelIndex + 1] = 0;
      data[pixelIndex + 2] = 0;
      data[pixelIndex + 3] = 255;
    }
  }

  const texture = device.createTexture({
    size: [size, size, 1],
    format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  device.queue.writeTexture(
    { texture },
    data,
    { bytesPerRow: size * 4, rowsPerImage: size },
    [size, size, 1],
  );

  return texture;
}
