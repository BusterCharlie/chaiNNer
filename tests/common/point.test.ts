import { expect, test } from 'vitest';
import { clampPoint, parsePoint, pointToPixel, stringifyPoint } from '../../src/common/point';

test('parsePoint', () => {
    expect(parsePoint('0.5,0.5')).toEqual([0.5, 0.5]);
    expect(parsePoint('0,1')).toEqual([0, 1]);
    expect(parsePoint(' 0.25 , 0.75 ')).toEqual([0.25, 0.75]);
    expect(parsePoint('1.5,0')).toBeUndefined();
    expect(parsePoint('-0.1,0')).toBeUndefined();
    expect(parsePoint('0.5')).toBeUndefined();
    expect(parsePoint('0.5,0.5,0.5')).toBeUndefined();
    expect(parsePoint('a,b')).toBeUndefined();
    expect(parsePoint(',0.5')).toBeUndefined();
    expect(parsePoint(0.5)).toBeUndefined();
    expect(parsePoint(undefined)).toBeUndefined();
});

test('stringifyPoint round-trips and clamps', () => {
    expect(stringifyPoint([0.5, 0.25])).toBe('0.5,0.25');
    expect(stringifyPoint([1 / 3, 2 / 3])).toBe('0.333333,0.666667');
    expect(stringifyPoint([-1, 2])).toBe('0,1');
    expect(parsePoint(stringifyPoint([0.123456789, 0.987654321]))).toEqual([0.123457, 0.987654]);
    expect(clampPoint([1.2, -3])).toEqual([1, 0]);
});

test('pointToPixel matches the backend', () => {
    expect(pointToPixel([0, 0], 100, 50)).toEqual([0, 0]);
    expect(pointToPixel([1, 1], 100, 50)).toEqual([99, 49]);
    expect(pointToPixel([0.5, 0.5], 100, 50)).toEqual([50, 25]);
    expect(pointToPixel([0.999, 0.019], 100, 50)).toEqual([99, 0]);
});
