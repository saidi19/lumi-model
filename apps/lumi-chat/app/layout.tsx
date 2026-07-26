import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

const DESCRIPTION =
  "Lumi L1 private AI workspace — natural chat, local history, evaluation, and release readiness.";

export async function generateMetadata(): Promise<Metadata> {
  const requestHeaders = await headers();
  const forwardedHost = requestHeaders.get("x-forwarded-host")?.split(",")[0]?.trim();
  const host = forwardedHost || requestHeaders.get("host") || "localhost:3000";
  const forwardedProtocol = requestHeaders
    .get("x-forwarded-proto")
    ?.split(",")[0]
    ?.trim();
  const protocol =
    forwardedProtocol || (host.startsWith("localhost") ? "http" : "https");
  let metadataBase: URL;
  try {
    metadataBase = new URL(`${protocol}://${host}`);
  } catch {
    metadataBase = new URL("http://localhost:3000");
  }

  return {
    description: DESCRIPTION,
    icons: {
      icon: "/favicon.svg",
      shortcut: "/favicon.svg",
    },
    metadataBase,
    openGraph: {
      description: DESCRIPTION,
      images: [
        {
          alt: "Lumi private AI workspace",
          height: 910,
          url: "/og.png",
          width: 1729,
        },
      ],
      title: "Lumi — AI workspace",
      type: "website",
    },
    title: {
      default: "Lumi — AI workspace",
      template: "%s · Lumi",
    },
    twitter: {
      card: "summary_large_image",
      description: DESCRIPTION,
      images: ["/og.png"],
      title: "Lumi — AI workspace",
    },
  };
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
