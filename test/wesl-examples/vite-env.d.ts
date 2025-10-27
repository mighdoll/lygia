/// <reference types="vite/client" />

import "vitest";

declare module "vitest" {
  interface Assertion<T = unknown> {
    toMatchImage(nameOrOptions?: string | import("vitest-image-snapshot").MatchImageOptions): Promise<void>;
  }
}
