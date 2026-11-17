import { Box, Center, HStack, Image, Text } from '@chakra-ui/react';
import { Resizable } from 're-resizable';
import { KeyboardEvent, PointerEvent, memo, useCallback, useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useContextSelector } from 'use-context-selector';
import { Size } from '../../../common/common-types';
import { Point, parsePoint, pointToPixel, stringifyPoint } from '../../../common/point';
import { GlobalVolatileContext } from '../../contexts/GlobalNodeState';
import { useDevicePixelRatio } from '../../hooks/useDevicePixelRatio';
import { pickImage, useInputImagePreview } from '../../hooks/useInputImagePreview';
import { DragHandleSVG } from '../CustomIcons';
import { WithLabel } from './InputContainer';
import { InputProps } from './props';

const DEFAULT_SIZE: Readonly<Size> = { width: 240, height: 240 };
const MIN_SIZE = 160;
const MARKER_SIZE = 16;

const toHex = (n: number) => n.toString(16).padStart(2, '0');

/**
 * The color of the preview image under the point. Only an approximation of the picked color
 * (the preview may be downscaled and is 8-bit), but it gives feedback without running the chain.
 */
const usePreviewColor = (url: string | undefined, point: Point): string | undefined => {
    const [pixels, setPixels] = useState<ImageData>();
    useEffect(() => {
        setPixels(undefined);
        if (!url) return undefined;
        let cancelled = false;
        const img = new window.Image();
        img.onload = () => {
            if (cancelled) return;
            const canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            if (!ctx) return;
            ctx.drawImage(img, 0, 0);
            try {
                setPixels(ctx.getImageData(0, 0, canvas.width, canvas.height));
            } catch {
                // e.g. a tainted canvas; the swatch is optional
            }
        };
        img.src = url;
        return () => {
            cancelled = true;
        };
    }, [url]);

    if (!pixels || pixels.width === 0 || pixels.height === 0) return undefined;
    const [px, py] = pointToPixel(point, pixels.width, pixels.height);
    const i = (py * pixels.width + px) * 4;
    const { data } = pixels;
    return `#${toHex(data[i])}${toHex(data[i + 1])}${toHex(data[i + 2])}`;
};

export const PointInput = memo(
    ({ value, setValue, input, isLocked, nodeId, size, setSize }: InputProps<'point', string>) => {
        const { t } = useTranslation();
        const { imageInput, previewOutput } = input;
        const committed = parsePoint(value) ?? parsePoint(input.def) ?? ([0.5, 0.5] as const);

        // while dragging, the point is kept locally and committed once on release
        const [dragPoint, setDragPoint] = useState<Point>();
        // the latest dragged point, independent of rendering (a release can come before the
        // last move has been rendered)
        const dragPointRef = useRef<Point>();
        const draggingRef = useRef(false);
        const point = dragPoint ?? committed;
        const areaRef = useRef<HTMLDivElement>(null);

        const preview = useInputImagePreview(nodeId, imageInput, previewOutput);
        const aspect = preview ? preview.width / preview.height : 1;

        // size (resizable like image previews). The default is only rendered, never stored:
        // storing it would also set the node's width, making the node jump on the first edit.
        const zoom = useContextSelector(GlobalVolatileContext, (c) => c.zoom);
        const dpr = useDevicePixelRatio();
        const startSize = useRef(size ?? DEFAULT_SIZE);
        const box = size ?? DEFAULT_SIZE;
        // fit the image (with its aspect ratio) into the box
        const fitWidth = Math.min(box.width, box.height * aspect);
        const fitHeight = fitWidth / aspect;
        const image = preview ? pickImage(preview.previews, fitWidth * zoom * dpr) : undefined;
        // the largest preview gives the most accurate swatch
        const largest = preview ? pickImage(preview.previews, Number.POSITIVE_INFINITY) : undefined;
        const swatch = usePreviewColor(largest?.url, point);

        const pointFromEvent = useCallback((e: PointerEvent): Point | undefined => {
            const rect = areaRef.current?.getBoundingClientRect();
            if (!rect || rect.width === 0 || rect.height === 0) return undefined;
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;
            return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
        }, []);

        const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
            if (isLocked || e.button !== 0) return;
            e.stopPropagation();
            areaRef.current?.focus();
            e.currentTarget.setPointerCapture(e.pointerId);
            draggingRef.current = true;
            const next = pointFromEvent(e);
            if (next) {
                dragPointRef.current = next;
                setDragPoint(next);
            }
        };
        const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
            if (!draggingRef.current) return;
            const next = pointFromEvent(e);
            if (!next) return;
            dragPointRef.current = next;
            setDragPoint(next);
        };
        const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
            if (!draggingRef.current) return;
            draggingRef.current = false;
            e.currentTarget.releasePointerCapture(e.pointerId);
            const finished = dragPointRef.current;
            dragPointRef.current = undefined;
            if (finished) setValue(stringifyPoint(finished));
            setDragPoint(undefined);
        };

        const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
            if (isLocked) return;
            // one pixel of the real image per press (10 with Shift)
            const stepX = (e.shiftKey ? 10 : 1) / Math.max(1, preview?.width ?? 1000);
            const stepY = (e.shiftKey ? 10 : 1) / Math.max(1, preview?.height ?? 1000);
            const [x, y] = committed;
            let next: Point | undefined;
            if (e.key === 'ArrowLeft') next = [x - stepX, y];
            if (e.key === 'ArrowRight') next = [x + stepX, y];
            if (e.key === 'ArrowUp') next = [x, y - stepY];
            if (e.key === 'ArrowDown') next = [x, y + stepY];
            if (!next) return;
            // don't let the editor (e.g. moving selected nodes) handle the arrow keys
            e.preventDefault();
            e.stopPropagation();
            setValue(stringifyPoint(next));
        };

        const [x, y] = point;
        const readout = preview
            ? (() => {
                  const [px, py] = pointToPixel(point, preview.width, preview.height);
                  return `X ${px}, Y ${py} px`;
              })()
            : `${(x * 100).toFixed(1)}%, ${(y * 100).toFixed(1)}%`;

        return (
            <WithLabel input={input}>
                <Center w="full">
                    <Resizable
                        className="nodrag"
                        defaultSize={box}
                        enable={{
                            top: false,
                            right: false,
                            bottom: false,
                            left: false,
                            topRight: false,
                            bottomRight: !isLocked,
                            bottomLeft: false,
                            topLeft: false,
                        }}
                        handleComponent={{
                            bottomRight: (
                                <Center
                                    cursor="nwse-resize"
                                    h="full"
                                    ml={-1}
                                    mt={-1}
                                    w="full"
                                >
                                    <DragHandleSVG
                                        color="var(--fg-300)"
                                        opacity={0.75}
                                    />
                                </Center>
                            ),
                        }}
                        maxHeight={1024}
                        maxWidth={1024}
                        minHeight={MIN_SIZE}
                        minWidth={MIN_SIZE}
                        scale={zoom}
                        size={box}
                        onResize={(e, direction, ref, d) => {
                            setSize({
                                width: startSize.current.width + d.width,
                                height: startSize.current.height + d.height,
                            });
                        }}
                        onResizeStart={() => {
                            startSize.current = box;
                        }}
                    >
                        <Center
                            bgColor="var(--node-image-preview-bg)"
                            borderRadius="md"
                            h="full"
                            w="full"
                        >
                            <Box
                                className="nodrag nowheel"
                                cursor={isLocked ? 'default' : 'crosshair'}
                                h={`${fitHeight}px`}
                                outline="none"
                                position="relative"
                                ref={areaRef}
                                tabIndex={0}
                                w={`${fitWidth}px`}
                                onKeyDown={onKeyDown}
                                onPointerDown={onPointerDown}
                                onPointerMove={onPointerMove}
                                onPointerUp={onPointerUp}
                            >
                                {image ? (
                                    <Image
                                        draggable={false}
                                        h="full"
                                        pointerEvents="none"
                                        src={image.url}
                                        userSelect="none"
                                        w="full"
                                    />
                                ) : (
                                    <Center
                                        border="1px dashed var(--fg-300)"
                                        h="full"
                                        p={4}
                                        w="full"
                                    >
                                        <Text
                                            color="var(--fg-300)"
                                            fontSize="sm"
                                            textAlign="center"
                                        >
                                            {t(
                                                'inputs.point.noImage',
                                                'Connect an image, or run the chain once, to see it here.'
                                            )}
                                        </Text>
                                    </Center>
                                )}
                                {/* crosshair lines across the image */}
                                <svg
                                    height="100%"
                                    preserveAspectRatio="none"
                                    style={{
                                        position: 'absolute',
                                        inset: 0,
                                        pointerEvents: 'none',
                                    }}
                                    viewBox="0 0 1 1"
                                    width="100%"
                                >
                                    {[
                                        { stroke: 'black', width: 3, opacity: 0.4 },
                                        { stroke: 'white', width: 1, opacity: 0.8 },
                                    ].map((s) => (
                                        <g
                                            key={s.stroke}
                                            opacity={s.opacity}
                                            stroke={s.stroke}
                                            strokeWidth={s.width}
                                            vectorEffect="non-scaling-stroke"
                                        >
                                            <line
                                                vectorEffect="non-scaling-stroke"
                                                x1={x}
                                                x2={x}
                                                y1={0}
                                                y2={1}
                                            />
                                            <line
                                                vectorEffect="non-scaling-stroke"
                                                x1={0}
                                                x2={1}
                                                y1={y}
                                                y2={y}
                                            />
                                        </g>
                                    ))}
                                </svg>
                                <Box
                                    bg={swatch ?? 'transparent'}
                                    border="2px solid white"
                                    borderRadius="full"
                                    boxShadow="0 0 0 1px rgba(0,0,0,0.6)"
                                    h={`${MARKER_SIZE}px`}
                                    left={`calc(${x * 100}% - ${MARKER_SIZE / 2}px)`}
                                    pointerEvents="none"
                                    position="absolute"
                                    top={`calc(${y * 100}% - ${MARKER_SIZE / 2}px)`}
                                    w={`${MARKER_SIZE}px`}
                                />
                            </Box>
                        </Center>
                    </Resizable>
                </Center>
                <HStack
                    justifyContent="center"
                    pt={1}
                    spacing={2}
                >
                    {swatch && (
                        <Box
                            bg={swatch}
                            border="1px solid var(--fg-300)"
                            borderRadius="sm"
                            flexShrink={0}
                            h="12px"
                            title={swatch}
                            w="12px"
                        />
                    )}
                    {/* wraps within the editor's width, so the text never changes the node's width */}
                    <Text
                        color="var(--fg-300)"
                        fontSize="xs"
                        maxW={`${box.width}px`}
                        textAlign="center"
                        whiteSpace="normal"
                    >
                        {readout}
                    </Text>
                </HStack>
                <Text
                    color="var(--fg-300)"
                    fontSize="xs"
                    maxW={`${box.width}px`}
                    mx="auto"
                    textAlign="center"
                    whiteSpace="normal"
                >
                    {t(
                        'inputs.point.hint',
                        'Click or drag on the image, then use the arrow keys to nudge the point.'
                    )}
                </Text>
            </WithLabel>
        );
    }
);
