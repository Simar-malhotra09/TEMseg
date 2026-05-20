import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  trailingSlash: true,
  devIndicators:false,
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
