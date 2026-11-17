import { useMemo } from 'react';
import { useReactFlow } from 'reactflow';
import { useContextSelector } from 'use-context-selector';
import { OutputId } from '../../common/common-types';
import { parseSourceHandle, stringifyTargetHandle } from '../../common/util';
import { GlobalVolatileContext } from '../contexts/GlobalNodeState';

export interface PreviewImage {
    readonly width: number;
    readonly height: number;
    readonly url: string;
}
export interface PreviewData {
    readonly previews: readonly PreviewImage[];
    readonly width: number;
    readonly height: number;
}
const isPreviewData = (data: unknown): data is PreviewData =>
    typeof data === 'object' &&
    data !== null &&
    Array.isArray((data as Partial<PreviewData>).previews) &&
    (data as PreviewData).previews.length > 0;

/** The smallest preview that is at least `realWidth` wide, or else the largest one. */
export const pickImage = (previews: readonly PreviewImage[], realWidth: number): PreviewImage => {
    const sorted = [...previews].sort((a, b) => a.width - b.width);
    return sorted.find((p) => p.width >= realWidth) ?? sorted[sorted.length - 1];
};

/**
 * The image connected to an image input of a node, for inputs that draw over it (e.g. point or
 * corner editors): the preview of whatever is connected to the input (e.g. Load Image, available
 * right away), or else the node's own input preview output from its last run.
 */
export const useInputImagePreview = (
    nodeId: string | undefined,
    imageInput: number,
    previewOutput: OutputId | null | undefined
): PreviewData | undefined => {
    const { getEdges } = useReactFlow();
    const edgeChanges = useContextSelector(GlobalVolatileContext, (c) => c.edgeChanges);
    const sourceHandle = useMemo(() => {
        if (!nodeId) return undefined;
        const targetHandle = stringifyTargetHandle({ nodeId, inputId: imageInput as never });
        return getEdges().find((e) => e.targetHandle === targetHandle)?.sourceHandle ?? undefined;
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [nodeId, imageInput, getEdges, edgeChanges]);

    const upstream = useContextSelector(GlobalVolatileContext, (c) => {
        if (!sourceHandle) return undefined;
        const { nodeId: sourceId, outputId } = parseSourceHandle(sourceHandle);
        return c.outputDataMap.get(sourceId)?.data?.[outputId];
    });
    const own = useContextSelector(GlobalVolatileContext, (c) => {
        if (!nodeId || previewOutput == null) return undefined;
        return c.outputDataMap.get(nodeId)?.data?.[previewOutput];
    });

    if (isPreviewData(upstream)) return upstream;
    if (sourceHandle && isPreviewData(own)) return own;
    return undefined;
};
