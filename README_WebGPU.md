This page describes how to link Lygia shader functions 
into your WebGPU application using WESL tools.[^1][^2]

[WESL](https://wesl-lang.dev) is a superset of WGSL that adds
features with community-supported tools.
WESL tools are available in Rust and JavaScript/TypeScript.

_Most (but not all) Lygia GLSL shaders are now available 
for WebGPU using WESL.
If a Lygia function you need is missing, 
please file an [issue](https://github.com/patriciogonzalezvivo/lygia/issues) 
or help [contribute](./README_WESL.md)._

## Using JavaScript or TypeScript

Install with `npm install lygia` or `pnpm install lygia`
([lygia npm package](https://www.npmjs.com/package/lygia)).
Once you install, 500+ Lygia functions
and constants will be available for you to use
via `import` statements in your application shader code.

```rs
import lygia::math::consts::PI;

fn main() {
  let p = PI;
}
```

Note that tree shaking is automatic in the JS tools 
because JS applications are often very sensitive to bundle size -
your application will include only the Lygia functions you use.

WESL tooling is available to integrate with a variety of popular
JavaScript / TypeScript build environments. 

### Using a JavaScript / TypeScript Bundler

If you build your application with a JavaScript / TypeScript bundler
like `vite`, `webpack` or `rollup`,
install the
[wesl](https://www.npmjs.com/package/wesl?activeTab=readme) and
[wesl-plugin](https://www.npmjs.com/package/wesl-plugin) packages.

#### Runtime Linking with a Bundler
Import shader code into your JavaScript or TypeScript application using the 
import statements suffixed with `?link` and the shaders will be linked together at runtime:

```ts
import appWesl from "../shaders/app.wesl?link";
import { link } from "wesl";

const linked = await link(appWesl);
linked.createShaderModule(gpuDevice);
```

For more details, check the [WESL bundler documentation](https://wesl-lang.dev/docs/JavaScript-Builds#wesl-with-javascript-bundlers) or refer to this
[lygia example using vite](https://stackblitz.com/github/wgsl-tooling-wg/examples/tree/main/lygia-example?file=README.md).

#### Static Linking with a Bundler
Alternatively,
you can statically link your shaders in advance using the `?static` suffix:
```ts
import appWgsl from "../shaders/app.wesl?static";
```
See the [lygia static linking example](https://stackblitz.com/github/wgsl-tooling-wg/examples/tree/main/lygia-static-example)
for details. 


#### Runtime vs. Static Linking
Static linking reduces your application bundle by ~15KB but offers less flexibility.

Some of the Lygia shader functions include `@if` [conditions](https://wesl-lang.dev/spec/ConditionalTranslation).
For example see `@if(YUV_SDTV)`
in [`yuv2rgb`](https://github.com/patriciogonzalezvivo/lygia/blob/main/color/space/yuv2rgb.wesl).
It's up to you whether you want to be able to set `YUV_SDTV` and similar conditions
in your applications shaders dynamically at runtime.

### Command Line Linking
For custom build pipelines, 
use the command-line tools to statically link Lygia shaders in your application:

```sh
npx wesl-link ./shaders/main.wesl
```

See the [Lygia CLI linking example](https://stackblitz.com/github/wgsl-tooling-wg/examples/tree/main/lygia-cli-example?file=README.md) for details.

If you just want a Lygia shader to copy and paste, you can
do this if you have installed the lygia npm package: 

```sh
npx wesl-link lygia::color::layer::addSourceOver
```

If you're working in a copy of the lygia source repository:

```sh
npx wesl-link package::color::layer::addSourceOver

# alternate syntax
npx wesl-link color/layer/addSourceOver.wesl
```

Either way will produce the WGSL for the requested Lygia function,
along with its dependencies.

### Link Using the API
You can use the linking API directly to build custom solutions
to link either at runtime or at compile time.
See the [API documentation](https://wesl-lang.dev/docs/JavaScript-Builds) for details.

### Additional WESL Examples

More WESL examples are available [here](https://github.com/wgsl-tooling-wg/examples).
Most examples run with one click in a browser sandbox.
The examples can also be used as starter templates with `degit`.

## Using Rust

```sh
cargo add lygia
```

### Linking at build time
```sh
cargo add --build wesl
```

```rs
/// build.rs
fn main() {
    wesl::Wesl::new("src/shaders").build_artifact("main.wesl", "my_shader");
}
```

### Linking at run-time

```sh
cargo add wesl
```

```rs
let shader_string = Wesl::new("src/shaders")
    .compile("main.wesl")
    .inspect_err(|e| eprintln!("WESL error: {e}")) // pretty errors with `display()`
    .unwrap()
    .to_string();
```

### Using the Rust CLI tool
```sh
cargo install wesl-cli
wesl compile <path/to/shader.wesl>
```


### WESL Rust Documentation
See [Getting Started Rust](https://wesl-lang.dev/docs/Getting-Started-Rust), 
the [wesl crate documentation](https://docs.rs/wesl/latest/wesl/),
and [WESL rust examples](https://github.com/wgsl-tooling-wg/wesl-rs/tree/main/examples).


## About WESL
- Use `import` statements to split shader code across files and
load npm/cargo libraries.
- Use `@if @else @elseif` statements to assemble specialized shaders
at build time or runtime.
- WESL tools include linkers in rust and javascript
(to combine WGSL/WESL files into applications), and syntax highlighters for zed, helix and nvim.
- More WESL tools are coming, including
an HTML [documentation generator](https://github.com/jannik4/wesldoc),
a [language server](https://github.com/wgsl-analyzer/wgsl-analyzer),
a VSCode plugin,
and a code formatter.

Read more about WESL at [wesl-lang.dev](https://wesl-lang.dev).

[^1]: Lygia functions are small and self-contained.
You can just copy, paste and edit them into your app if you prefer doing things manually!

[^2]: Lygia currently hosts two versions of WebGPU shaders. 
The original versions extend WGSL with custom `#include` statements and have a `.wgsl` suffix. 
The newer versions use the WESL language and have a `.wesl` suffix.
For new Lygia users, we recommend WESL.