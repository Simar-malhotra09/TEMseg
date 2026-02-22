"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const client = new QueryClient();

export default function Providers({ children }) {
  return (
    <QueryClientProvider client={client}>
      <div style={{ height: "100%", margin: 0, padding: 0 }}>
        {children}
      </div>
    </QueryClientProvider>
  );
}
