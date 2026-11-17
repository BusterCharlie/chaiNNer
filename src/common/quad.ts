/**
 * Helpers for quad inputs: four corners (top left, top right, bottom right, bottom left) given as
 * fractions of an image's size, stored as "x1,y1,x2,y2,x3,y3,x4,y4".
 */

export type Point = readonly [x: number, y: number];
export type Quad = readonly [Point, Point, Point, Point];

/** Parses "x1,y1,...,x4,y4". Returns undefined for anything that isn't 8 numbers. */
export const parseQuad = (value: string | undefined): Quad | undefined => {
    if (!value) return undefined;
    const n = value.split(',').map(Number);
    if (n.length !== 8 || n.some((v) => !Number.isFinite(v))) return undefined;
    return [
        [n[0], n[1]],
        [n[2], n[3]],
        [n[4], n[5]],
        [n[6], n[7]],
    ];
};

export const stringifyQuad = (quad: Quad): string =>
    quad.map(([x, y]) => `${Number(x.toFixed(5))},${Number(y.toFixed(5))}`).join(',');

/**
 * The same rules the backend enforces: corners inside the image, at least `minDistance` apart,
 * and forming a convex shape in clockwise order (top left, top right, bottom right, bottom left).
 */
export const isValidQuad = (quad: Quad, minDistance: number): boolean => {
    for (const [x, y] of quad) {
        if (x < 0 || x > 1 || y < 0 || y > 1) return false;
    }
    for (let i = 0; i < 4; i += 1) {
        for (let j = i + 1; j < 4; j += 1) {
            if (Math.hypot(quad[i][0] - quad[j][0], quad[i][1] - quad[j][1]) < minDistance - 1e-9) {
                return false;
            }
        }
    }
    for (let i = 0; i < 4; i += 1) {
        const [ax, ay] = quad[i];
        const [bx, by] = quad[(i + 1) % 4];
        const [cx, cy] = quad[(i + 2) % 4];
        const cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx);
        if (cross <= 0) return false;
    }
    return true;
};

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

export const moveCorner = (quad: Quad, index: number, point: Point): Quad => {
    const next = [...quad] as [Point, Point, Point, Point];
    next[index] = [clamp01(point[0]), clamp01(point[1])];
    return next;
};
