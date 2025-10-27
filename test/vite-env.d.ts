// Type declarations for Vite special imports
// See: https://vitejs.dev/guide/assets.html#importing-asset-as-string

declare module "*.wesl?raw" {
  const content: string;
  export default content;
}
