export const dynamic = "force-dynamic";

export async function GET() {
  const endpointConfigured = Boolean(process.env.LUMI_MODEL_BASE_URL?.trim());
  const contractSha = process.env.LUMI_SERVING_CONTRACT_SHA256?.trim() ?? "";
  const contractBound = /^[0-9a-f]{64}$/.test(contractSha);
  const connected = endpointConfigured && contractBound;
  return Response.json(
    {
      blockers: connected
        ? []
        : endpointConfigured
          ? ["serving_contract"]
          : [
              "final_80b_checkpoint",
              "sft_checkpoint",
              "benchmark_report",
              "safety_report",
            ],
      mode: connected ? "connected" : "staging",
      model: process.env.LUMI_MODEL_ID?.trim() || "lumi-4b-instruct",
      ready: connected,
    },
    {
      headers: {
        "Cache-Control": "no-store",
      },
    },
  );
}
