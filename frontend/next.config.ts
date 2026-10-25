import type { NextConfig } from "next";
import { PHASE_DEVELOPMENT_SERVER } from "next/constants";

const nextConfig = (phase: string): NextConfig => ({
  distDir: phase === PHASE_DEVELOPMENT_SERVER ? ".next-dev" : ".next",
  output: "export",
  trailingSlash: true,
  devIndicators: false,
  eslint: { ignoreDuringBuilds: true },
});

export default nextConfig;
