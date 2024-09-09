package brain.math;

import java.util.function.IntFunction;

/**
 * 16.03.2022
 * Emilio Zottel
 * 3CHIF
 */
public class CpuVector extends Vector {

    public CpuVector(int size) {
        super(size);
    }

    public CpuVector(float... values) {
        super(values);
    }

    /**
     * Useful in Matrix.java when reading n floats per row and constructing a Vector using these floats
     */
    public static CpuVector of(float... values) {
        return new CpuVector(values);
    }

    public static Vector[] makeArray(int cols, int rows) {
        return Vector.makeArray(cols, rows, CpuVector::new);
    }

    @Override
    public Matrix toRowVector() {
        return new CpuMatrix(this);
    }

    @Override
    public Vector negate() {
        return map(e -> -e);
    }

    @Override
    public Vector add(Vector v) {
        return withEach(i -> get(i) + v.get(i), v.size());
    }

    @Override
    public Vector sub(Vector v) {
        return withEach(i -> get(i) - v.get(i), v.size());
    }

    @Override
    public Vector mult(float factor) {
        return map(e -> e * factor);
    }

    @Override
    public Vector mult(Vector v) {
        return withEach(i -> get(i) * v.get(i), v.size());
    }

    /**
     * Multiplies this column vector with a row vector and returns the resulting matrix
     *
     * @param rowVector the row vector that this column vector should be multiplied with
     * @return the resulting matrix
     */
    @Override
    public Matrix mult(Matrix rowVector) {
        if (rowVector.rows != 1) {
            throw new IllegalArgumentException("Parameter m must be a row vector, which means it must have exactly 1 row");
        }

        var result = new CpuMatrix(rowVector.cols, size());  // size() == rows of this column vector
        var row = rowVector.getRow(0);
        return result.withEachRow(j -> row.mult(get(j)));
    }

    @Override
    public Vector div(float divisor) {
        return map(e -> e / divisor);
    }

    @Override
    public float dot(Vector v) {
        check(v.size());
        float result = 0;

        for (int i = 0; i < size(); i++) {
            result += get(i) * v.get(i);
        }

        return result;
    }

    @Override
    public void setAll(Vector v) {
        check(v.size());
        setEach(v::get);
    }

    @Override
    public Vector withEach(IntFunction<Float> function) {
        Vector v = new CpuVector(size());
        v.setEach(function);
        return v;
    }

}