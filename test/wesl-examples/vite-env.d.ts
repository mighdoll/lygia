/// <reference types="vite/client" />

import "vitest";
import type { MatchImageOptions } from "vitest-image-snapshot";

declare module "vitest" {
  interface Assertion<T> {
    toMatchImage(nameOrOptions?: string | MatchImageOptions): Promise<void>;
  }
}
