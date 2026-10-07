"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { useEditor } from "@/hooks/use-editor";
import {
	useKeybindingsListener,
	useKeybindingDisabler,
} from "@/hooks/use-keybindings";
import { useEditorActions } from "@/hooks/actions/use-editor-actions";
import { useEmbeddingIndexer } from "@/hooks/use-embedding-indexer";
import { prefetchFontAtlas } from "@/lib/fonts/google-fonts";
import { storageService } from "@/services/storage/service";
import { useTranscriptStore } from "@/stores/transcript-store";

interface EditorProviderProps {
	projectId: string;
	children: React.ReactNode;
}

export function EditorProvider({ projectId, children }: EditorProviderProps) {
	const editor = useEditor();
	const router = useRouter();
	const [isLoading, setIsLoading] = useState(true);
	const [error, setError] = useState<string | null>(null);
	const { disableKeybindings, enableKeybindings } = useKeybindingDisabler();
	const activeProject = editor.project.getActiveOrNull();

	useEffect(() => {
		if (isLoading) {
			disableKeybindings();
		} else {
			enableKeybindings();
		}
	}, [isLoading, disableKeybindings, enableKeybindings]);

	useEffect(() => {
		let cancelled = false;

		const loadProject = async () => {
			try {
				setIsLoading(true);
				await editor.project.loadProject({ id: projectId });

				if (cancelled) return;

				setIsLoading(false);
				prefetchFontAtlas();
			} catch (err) {
				if (cancelled) return;

				const isNotFound =
					err instanceof Error &&
					(err.message.includes("not found") ||
						err.message.includes("does not exist"));

				if (isNotFound) {
					try {
						const newProjectId = await editor.project.createNewProject({
							name: "Untitled Project",
						});
						router.replace(`/editor/${newProjectId}`);
					} catch (_createErr) {
						setError("Failed to create project");
						setIsLoading(false);
					}
				} else {
					setError(
						err instanceof Error ? err.message : "Failed to load project",
					);
					setIsLoading(false);
				}
			}
		};

		loadProject();

		return () => {
			cancelled = true;
		};
	}, [projectId, editor, router]);

        useEffect(() => {
                if (isLoading || error) return;

                const currentProject = editor.project.getActiveOrNull();
                if (currentProject?.metadata.id !== projectId) return;

                let saveTimer: ReturnType<typeof setTimeout> | null = null;

                const saveCurrentTranscript = async () => {
                        const state = useTranscriptStore.getState();

                        await storageService.saveTranscript({
                                projectId,
                                transcript: {
                                        segments: state.segments,
                                        language: state.language,
                                        duration: state.duration,
                                        fillers: state.fillers,
                                        silences: state.silences,
                                        chapters: state.chapters,
                                        translations: state.translations,
                                        speakerNames: state.speakerNames,
                                        speakerPositions: state.speakerPositions,
                                        emotions: state.emotions,
                                },
                        });
                };

                const unsubscribe = useTranscriptStore.subscribe(
                        (state, previousState) => {
                                const changed =
                                        state.segments !== previousState.segments ||
                                        state.language !== previousState.language ||
                                        state.duration !== previousState.duration ||
                                        state.fillers !== previousState.fillers ||
                                        state.silences !== previousState.silences ||
                                        state.chapters !== previousState.chapters ||
                                        state.translations !== previousState.translations ||
                                        state.speakerNames !== previousState.speakerNames ||
                                        state.speakerPositions !== previousState.speakerPositions ||
                                        state.emotions !== previousState.emotions;

                                if (!changed) return;

                                if (saveTimer) clearTimeout(saveTimer);

                                saveTimer = setTimeout(() => {
                                        void saveCurrentTranscript().catch((saveError) => {
                                                console.error(
                                                        "Failed to save project transcript:",
                                                        saveError,
                                                );
                                        });
                                }, 500);
                        },
                );

                void saveCurrentTranscript().catch((saveError) => {
                        console.error(
                                "Failed to save project transcript:",
                                saveError,
                        );
                });

                return () => {
                        unsubscribe();

                        if (saveTimer) {
                                clearTimeout(saveTimer);
                        }

                        void saveCurrentTranscript().catch((saveError) => {
                                console.error(
                                        "Failed to save project transcript:",
                                        saveError,
                                );
                        });
                };
        }, [projectId, isLoading, error, editor]);
	if (error) {
		return (
			<div className="bg-background flex h-screen w-screen items-center justify-center">
				<div className="flex flex-col items-center gap-4">
					<p className="text-destructive text-sm">{error}</p>
				</div>
			</div>
		);
	}

	if (isLoading) {
		return (
			<div className="bg-background flex h-screen w-screen items-center justify-center">
				<div className="flex flex-col items-center gap-4">
					<Loader2 className="text-muted-foreground size-8 animate-spin" />
					<p className="text-muted-foreground text-sm">Loading project...</p>
				</div>
			</div>
		);
	}

	if (!activeProject) {
		return (
			<div className="bg-background flex h-screen w-screen items-center justify-center">
				<div className="flex flex-col items-center gap-4">
					<Loader2 className="text-muted-foreground size-8 animate-spin" />
					<p className="text-muted-foreground text-sm">Exiting project...</p>
				</div>
			</div>
		);
	}

	return (
		<>
			<EditorRuntimeBindings />
			{children}
		</>
	);
}

function EditorRuntimeBindings() {
	const editor = useEditor();

	useEffect(() => {
		const handleBeforeUnload = (event: BeforeUnloadEvent) => {
			if (!editor.save.getIsDirty()) return;
			event.preventDefault();
			(event as unknown as { returnValue: string }).returnValue = "";
		};

		window.addEventListener("beforeunload", handleBeforeUnload);
		return () => window.removeEventListener("beforeunload", handleBeforeUnload);
	}, [editor]);

	useEditorActions();
	useKeybindingsListener();
	useEmbeddingIndexer();
	return null;
}
