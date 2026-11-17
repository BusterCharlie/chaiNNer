import { Box, Center, HStack, Image, Text } from '@chakra-ui/react';
import { Resizable } from 're-resizable';
import { KeyboardEvent, PointerEvent, memo, useCallback, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useContextSelector } from 'use-context-selector';
import { Size } from '../../../common/common-types';
import {
    Point,
    Quad,
    isValidQuad,
    moveCorner,
    parseQuad,
    stringifyQuad,
} from '../../../common/quad';
import { GlobalVolatileContext } from '../../contexts/GlobalNodeState';
import { useDevicePixelRatio } from '../../hooks/useDevicePixelRatio';
import { pickImage, useInputImagePreview } from '../../hooks/useInputImagePreview';
import { DragHandleSVG } from '../CustomIcons';
import { WithLabel } from './InputContainer';
import { InputProps } from './props';

const CORNER_NAMES = ['top left', 'top right', 'bottom right', 'bottom left'] as const;
const DEFAULT_SIZE: Readonly<Size> = { width: 240, height: 240 };
const MIN_SIZE = 160;
const HANDLE_SIZE = 14;

export const QuadInput = memo(
    ({ value, setValue, input, isLocked, nodeId, size, setSize }: InputProps<'quad', string>) => {
        const { t } = useTranslation();
        const { minDistance, imageInput, previewOutput } = input;
        const defaultQuad = parseQuad(input.def) ?? parseQuad('0,0,1,0,1,1,0,1')!;
        const committed = parseQuad(value) ?? defaultQuad;

        // while dragging, corners are kept locally and committed once on release
        const [dragQuad, setDragQuad] = useState<Quad>();
        // the latest dragged corners, independent of rendering (a release can come before the
        // last move has been rendered)
        const dragQuadRef = useRef<Quad>();
        const quad = dragQuad ?? committed;
        const [selected, setSelected] = useState<number>();
        const draggingRef = useRef<number>();
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

        const pointFromEvent = useCallback((e: PointerEvent): Point | undefined => {
            const rect = areaRef.current?.getBoundingClientRect();
            if (!rect || rect.width === 0 || rect.height === 0) return undefined;
            return [(e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height];
        }, []);

        const onPointerDown = (index: number) => (e: PointerEvent<HTMLDivElement>) => {
            if (isLocked) return;
            e.stopPropagation();
            e.currentTarget.setPointerCapture(e.pointerId);
            draggingRef.current = index;
            setSelected(index);
            dragQuadRef.current = committed;
            setDragQuad(committed);
        };
        const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
            const index = draggingRef.current;
            if (index === undefined) return;
            const point = pointFromEvent(e);
            if (!point) return;
            const base = dragQuadRef.current ?? committed;
            const next = moveCorner(base, index, point);
            // refuse moves that would put corners on top of each other or fold the shape
            if (isValidQuad(next, minDistance)) {
                dragQuadRef.current = next;
                setDragQuad(next);
            }
        };
        const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
            if (draggingRef.current === undefined) return;
            draggingRef.current = undefined;
            e.currentTarget.releasePointerCapture(e.pointerId);
            const finished = dragQuadRef.current;
            dragQuadRef.current = undefined;
            if (finished) setValue(stringifyQuad(finished));
            setDragQuad(undefined);
        };

        const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
            if (selected === undefined || isLocked) return;
            const step = (e.shiftKey ? 10 : 1) / Math.max(1, preview?.width ?? 1000);
            const stepY = (e.shiftKey ? 10 : 1) / Math.max(1, preview?.height ?? 1000);
            const [x, y] = committed[selected];
            let point: Point | undefined;
            if (e.key === 'ArrowLeft') point = [x - step, y];
            if (e.key === 'ArrowRight') point = [x + step, y];
            if (e.key === 'ArrowUp') point = [x, y - stepY];
            if (e.key === 'ArrowDown') point = [x, y + stepY];
            if (e.key === 'Tab') {
                e.preventDefault();
                e.stopPropagation();
                setSelected((selected + (e.shiftKey ? 3 : 1)) % 4);
                return;
            }
            if (!point) return;
            // don't let the editor (e.g. moving selected nodes) handle the arrow keys
            e.preventDefault();
            e.stopPropagation();
            const next = moveCorner(committed, selected, point);
            if (isValidQuad(next, minDistance)) setValue(stringifyQuad(next));
        };

        const polygon = quad.map(([x, y]) => `${x},${y}`).join(' ');
        const outside = `M0,0 H1 V1 H0 Z M${quad.map(([x, y]) => `${x},${y}`).join(' L')} Z`;

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
                                h={`${fitHeight}px`}
                                outline="none"
                                position="relative"
                                ref={areaRef}
                                tabIndex={0}
                                w={`${fitWidth}px`}
                                onKeyDown={onKeyDown}
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
                                                'inputs.quad.noImage',
                                                'Connect an image, or run the chain once, to see it here.'
                                            )}
                                        </Text>
                                    </Center>
                                )}
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
                                    <path
                                        d={outside}
                                        fill="black"
                                        fillOpacity={0.35}
                                        fillRule="evenodd"
                                    />
                                    <polygon
                                        fill="none"
                                        points={polygon}
                                        stroke="var(--selection-color, #3182ce)"
                                        strokeWidth={2}
                                        vectorEffect="non-scaling-stroke"
                                    />
                                </svg>
                                {quad.map(([x, y], i) => (
                                    <Box
                                        aria-label={CORNER_NAMES[i]}
                                        bg={
                                            selected === i
                                                ? 'var(--selection-color, #3182ce)'
                                                : 'white'
                                        }
                                        border="2px solid"
                                        borderColor="var(--selection-color, #3182ce)"
                                        borderRadius="full"
                                        boxShadow="0 0 0 1px rgba(0,0,0,0.5)"
                                        cursor={isLocked ? 'default' : 'grab'}
                                        h={`${HANDLE_SIZE}px`}
                                        key={CORNER_NAMES[i]}
                                        left={`calc(${x * 100}% - ${HANDLE_SIZE / 2}px)`}
                                        position="absolute"
                                        top={`calc(${y * 100}% - ${HANDLE_SIZE / 2}px)`}
                                        w={`${HANDLE_SIZE}px`}
                                        onPointerDown={onPointerDown(i)}
                                    />
                                ))}
                            </Box>
                        </Center>
                    </Resizable>
                </Center>
                <HStack
                    justifyContent="center"
                    pt={1}
                >
                    {/* wraps within the editor's width, so the text never changes the node's width */}
                    <Text
                        color="var(--fg-300)"
                        fontSize="xs"
                        maxW={`${box.width}px`}
                        textAlign="center"
                        whiteSpace="normal"
                    >
                        {selected === undefined
                            ? t(
                                  'inputs.quad.hint',
                                  'Drag the corners. Click one, then use the arrow keys to nudge it.'
                              )
                            : `${t(`inputs.quad.corner.${selected}`, CORNER_NAMES[selected])}: ${(
                                  quad[selected][0] * 100
                              ).toFixed(2)}%, ${(quad[selected][1] * 100).toFixed(2)}%`}
                    </Text>
                </HStack>
            </WithLabel>
        );
    }
);
