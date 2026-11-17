import { expect, test } from 'vitest';
import { Quad, isValidQuad, moveCorner, parseQuad, stringifyQuad } from '../../src/common/quad';

const full: Quad = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
];

test('parse and stringify round-trip', () => {
    expect(parseQuad('0,0,1,0,1,1,0,1')).toEqual(full);
    expect(stringifyQuad(full)).toBe('0,0,1,0,1,1,0,1');
    expect(parseQuad(stringifyQuad(moveCorner(full, 0, [0.123456789, 0.5])))).toEqual([
        [0.12346, 0.5],
        [1, 0],
        [1, 1],
        [0, 1],
    ]);
});

test('parse rejects malformed values', () => {
    expect(parseQuad(undefined)).toBeUndefined();
    expect(parseQuad('')).toBeUndefined();
    expect(parseQuad('0,0,1,0,1,1')).toBeUndefined();
    expect(parseQuad('0,0,1,0,1,1,0,x')).toBeUndefined();
});

test('valid shapes', () => {
    expect(isValidQuad(full, 0.05)).toBe(true);
    expect(
        isValidQuad(
            [
                [0.2, 0.1],
                [0.8, 0.2],
                [0.9, 0.9],
                [0.1, 0.8],
            ],
            0.05
        )
    ).toBe(true);
});

test('invalid shapes are rejected (same rules as the backend)', () => {
    // two corners closer than the minimum distance
    expect(isValidQuad(moveCorner(full, 1, [0.03, 0]), 0.05)).toBe(false);
    // crossing (bow tie): top right dragged past bottom right
    expect(isValidQuad(moveCorner(full, 1, [1, 1.2]), 0.05)).toBe(false);
    expect(
        isValidQuad(
            [
                [0, 0],
                [1, 1],
                [1, 0],
                [0, 1],
            ],
            0.05
        )
    ).toBe(false);
    // concave: top right dragged inwards past the diagonal
    expect(isValidQuad(moveCorner(full, 1, [0.4, 0.6]), 0.05)).toBe(false);
    // counter-clockwise order
    expect(
        isValidQuad(
            [
                [1, 0],
                [0, 0],
                [0, 1],
                [1, 1],
            ],
            0.05
        )
    ).toBe(false);
});

test('moving a corner clamps it to the image', () => {
    expect(moveCorner(full, 2, [1.5, -0.2])[2]).toEqual([1, 0]);
});
