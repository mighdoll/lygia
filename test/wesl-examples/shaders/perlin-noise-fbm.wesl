import lygia::generative::cnoise::cnoise2;

@fragment
fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let st = pos.xy / 512.0;  // Normalize coordinates

  // Multi-octave FBM (Fractional Brownian Motion)
  var n = 0.0;
  var amplitude = 0.5;
  var frequency = 3.0;

  for (var i = 0; i < 5; i++) {
    n += cnoise2(st * frequency) * amplitude;
    frequency *= 2.0;
    amplitude *= 0.5;
  }

  // Map [-1, 1] noise to [0, 1] color range
  let color = vec3f(n * 0.5 + 0.5);
  return vec4f(color, 1.0);
}
