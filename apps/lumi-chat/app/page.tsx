import type { Metadata } from "next";
import { LumiStudio } from "./components/LumiStudio";

export const metadata: Metadata = {
  title: {
    absolute: "Lumi — AI workspace",
  },
  description:
    "The staging workspace for Lumi L1 chat, evaluation, and deployment readiness.",
  other: {
    "lumi-release-state": "pretraining",
  },
};

export default function Home() {
  return <LumiStudio />;
}
