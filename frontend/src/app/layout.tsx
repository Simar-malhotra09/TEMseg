import Providers from "./providers";

export default function RootLayout({ children }) {
  return (
    <html lang="en" style={{ height: "100%" }}>
      <body style={{ margin: 0, padding: 0, height: "100%", background: "#0d0d0d" }}>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
