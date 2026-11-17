/** A position as fractions of an image's width and height (0..1). */
export type Point = readonly [x: number, y: number];

const clamp01 = (n: number): number => Math.min(1, Math.max(0, n));

/** Parses "x,y" (both 0..1). Returns `undefined` for anything else. */
export const parsePoint = (value: unknown): Point | undefined => {
    if (typeof value !== 'string') return undefined;
    const parts = value.split(',');
    if (parts.length !== 2 || parts.some((p) => p.trim() === '')) return undefined;
    const [x, y] = parts.map(Number);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return undefined;
    if (x < 0 || x > 1 || y < 0 || y > 1) return undefined;
    return [x, y];
};

export const clampPoint = ([x, y]: Point): Point => [clamp01(x), clamp01(y)];

/** Rounded to 6 decimals, which is well below a pixel for any realistic image. */
export const stringifyPoint = (point: Point): string => {
    const [x, y] = clampPoint(point);
    const round = (n: number) => String(Math.round(n * 1e6) / 1e6);
    return `${round(x)},${round(y)}`;
};

/**
 * The pixel a point falls on, matching the backend: `min(floor(x * width), width - 1)`.
 */
export const pointToPixel = ([x, y]: Point, width: number, height: number): [number, number] => [
    Math.min(Math.floor(x * width), width - 1),
    Math.min(Math.floor(y * height), height - 1),
];
