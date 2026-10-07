"use client";

import { useCallback, useState } from "react";
import { toast } from "sonner";

import { aiClient } from "@/lib/ai-client";
import { useTranscriptStore } from "@/stores/transcript-store";
import { useTextTimelineBridge } from "@/hooks/use-text-timeline-bridge";
import type { TimeRange } from "@/lib/text-timeline-sync";

export type EenokiAutoCutDecision = {
        segment_id: string;
        action: "KEEP" | "CUT" | "REVIEW";
        reason: string;
        confidence: number;
};

export type EenokiAutoCutPlan = {
        decisions: EenokiAutoCutDecision[];
        cut_ranges: Array<{
                segment_id: string;
                start: number;
                end: number;
                reason: string;
                confidence: number;
        }>;
        summary: string;
};

export function useEenokiAutoCut() {
        const { handleDeleteSegments } = useTextTimelineBridge();

        const [plan, setPlan] = useState<EenokiAutoCutPlan | null>(null);
        const [isAnalyzing, setIsAnalyzing] = useState(false);
        const [isApplying, setIsApplying] = useState(false);

        const analyze = useCallback(async () => {
                const segments = useTranscriptStore.getState().segments;

                if (segments.length === 0) {
                        toast.error("Transcript required", {
                                description: "Generate the transcript before running EENOKI Auto Cut.",
                        });
                        return null;
                }

                setIsAnalyzing(true);

                try {
                        const result = await aiClient.eenokiAutoCutPlan(
                                segments.map((segment) => ({
                                        id: segment.id,
                                        text: segment.text,
                                        start: segment.start,
                                        end: segment.end,
                                })),
                        );

                        setPlan(result);

                        const cutCount = result.decisions.filter(
                                (decision) => decision.action === "CUT",
                        ).length;

                        const reviewCount = result.decisions.filter(
                                (decision) => decision.action === "REVIEW",
                        ).length;

                        toast.success("EENOKI Auto Cut analysis complete", {
                                description:
                                        `${cutCount} cut candidate${cutCount === 1 ? "" : "s"}, ` +
                                        `${reviewCount} review item${reviewCount === 1 ? "" : "s"}.`,
                        });

                        return result;
                } catch (error) {
                        const message =
                                error instanceof Error
                                        ? error.message
                                        : "EENOKI Auto Cut analysis failed";

                        toast.error("Auto Cut analysis failed", {
                                description: message,
                        });

                        return null;
                } finally {
                        setIsAnalyzing(false);
                }
        }, []);

        const applyPlan = useCallback(
                (sourcePlan: EenokiAutoCutPlan | null = plan, minConfidence = 0.9) => {
                        if (!sourcePlan) {
                                toast.error("No Auto Cut plan", {
                                        description: "Run analysis first.",
                                });
                                return {
                                        applied: 0,
                                        skipped: 0,
                                };
                        }

                        const safeCuts = sourcePlan.decisions.filter(
                                (decision) =>
                                        decision.action === "CUT" &&
                                        decision.confidence >= minConfidence,
                        );

                        const skippedCuts = sourcePlan.decisions.filter(
                                (decision) =>
                                        decision.action === "CUT" &&
                                        decision.confidence < minConfidence,
                        );

                        if (safeCuts.length === 0) {
                                toast.info("No safe cuts to apply", {
                                        description:
                                                skippedCuts.length > 0
                                                        ? `${skippedCuts.length} low-confidence cut candidate(s) preserved.`
                                                        : "Gemini did not identify anything that should be removed.",
                                });

                                return {
                                        applied: 0,
                                        skipped: skippedCuts.length,
                                };
                        }

                        const segmentMap = new Map(
                                useTranscriptStore
                                        .getState()
                                        .segments.map((segment) => [
                                                String(segment.id),
                                                segment,
                                        ]),
                        );

                        const segmentIds: string[] = [];
                        const cuts: TimeRange[] = [];

                        for (const decision of safeCuts) {
                                const segment = segmentMap.get(decision.segment_id);

                                if (!segment) continue;

                                segmentIds.push(decision.segment_id);
                                cuts.push({
                                        start: segment.start,
                                        end: segment.end,
                                });
                        }

                        if (cuts.length === 0) {
                                toast.info("No matching transcript segments found");
                                return {
                                        applied: 0,
                                        skipped: skippedCuts.length,
                                };
                        }

                        setIsApplying(true);

                        try {
                                handleDeleteSegments(segmentIds, cuts);

                                toast.success("EENOKI Auto Cut applied", {
                                        description:
                                                `${cuts.length} high-confidence section` +
                                                `${cuts.length === 1 ? "" : "s"} removed. Undo is available.`,
                                });

                                setPlan(null);

                                return {
                                        applied: cuts.length,
                                        skipped: skippedCuts.length,
                                };
                        } finally {
                                setIsApplying(false);
                        }
                },
                [handleDeleteSegments, plan],
        );

        const clearPlan = useCallback(() => {
                setPlan(null);
        }, []);

        return {
                plan,
                analyze,
                applyPlan,
                clearPlan,
                isAnalyzing,
                isApplying,
        };
}
