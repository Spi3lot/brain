package brain.math;

import brain.domain.Brain;
import lombok.EqualsAndHashCode;
import lombok.ToString;

import java.util.function.BiConsumer;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.function.UnaryOperator;

/**
 * 16.03.2022
 * Emilio Zottel
 * 3CHIF
 */
@EqualsAndHashCode
@ToString
public class Vector {
    private final float[] values;

    public Vector(int size) {
        values = new float[size];
    }

    // private and float[] instead of float... to avoid possible overlappings with the public brain.math.Vector(int) constructor
    private Vector(float[] values) {
        this.values = values;
    }

    /**
     * Useful in Matrix.java when reading n floats per row and constructing a Vector using these floats
     */
    public static Vector of(float... values) {
        return new Vector(values);
    }

    public static Vector[] makeArray(int cols, int rows) {
        Vector[] arr = new Vector[rows];

        for (int i = 0; i < rows; i++) {
            arr[i] = new Vector(cols);
        }

        return arr;
    }


    protected void check(int len, String message) {
        assert size() == len : message;
    }

    private void check(int len) {
        check(len, "Vector sizes must match");
    }

    public Matrix toRowVector() {
        return new Matrix(this);
    }

    public Vector negate() {
        return map(e -> -e);
    }

    public Vector add(Vector v) {
        return withEach(i -> get(i) + v.get(i), v.size());
    }

    public Vector sub(Vector v) {
        return withEach(i -> get(i) - v.get(i), v.size());
    }

    public Vector mult(float factor) {
        return map(e -> e * factor);
    }

    public Vector mult(Vector v) {
        return withEach(i -> get(i) * v.get(i), v.size());
    }

    /**
     * Multiplies this column vector with a row vector and returns the resulting matrix
     *
     * @param rowVector the row vector that this column vector should be multiplied with
     * @return the resulting matrix
     */
    public Matrix mult(Matrix rowVector) {
        assert rowVector.rows == 1 : "Parameter m must be a row vector, which means it must have exactly 1 row";
        Matrix result = new Matrix(rowVector.cols, size());  // size() == rows of this column vector
        Vector row = rowVector.getRow(0);

        return result.withEachRow(j -> row.mult(get(j)));
    }

    //public Vector mult(Matrix m) {
    //    return m/*.transpose()*/.mult(this);
    //}

    public Vector div(float divisor) {
        return map(e -> e / divisor);
    }

    public float dot(Vector v) {
        check(v.size());
        float result = 0;

        for (int i = 0; i < size(); i++) {
            result += get(i) * v.get(i);
        }

        return result;
    }

    public float sum() {
        float[] result = {0.0f};
        forEach((_, x) -> result[0] += x);

        return result[0];
    }

    public int argmax() {
        float[] max_maxIdx = {0.0f, 0};

        forEach((i, x) -> {
            if (x > max_maxIdx[0]) {
                max_maxIdx[0] = x;
                max_maxIdx[1] = i;
            }
        });

        return (int) max_maxIdx[1];
    }

    public int size() {
        return values.length;
    }

    public float get(int i) {
        return values[i];
    }

    public void set(int i, float value) {
        values[i] = value;
    }

    public void setAll(Vector v) {
        assert size() == v.size();
        setEach(v::get);
    }

    public void setEach(IntFunction<Float> function) {
        for (int i = 0; i < size(); i++) {
            set(i, function.apply(i));
        }
    }

    public Vector withEach(IntFunction<Float> function) {
        Vector v = new Vector(size());
        v.setEach(function);

        return v;
    }

    public Vector withEach(IntFunction<Float> function, int len) {
        check(len);
        return withEach(function);
    }

    public Vector map(UnaryOperator<Float> function) {
        return withEach(i -> function.apply(get(i)));
    }

    public void forEach(IntConsumer consumer) {
        for (int i = 0; i < size(); i++) {
            consumer.accept(i);
        }
    }

    public void forEach(BiConsumer<Integer, Float> biConsumer) {
        forEach(i -> biConsumer.accept(i, get(i)));
    }

    public Vector fillWithRandomValues(float min, float maxExclusive) {
        setEach(_ -> Brain.random.nextFloat(maxExclusive - min) + min);
        return this;
    }
}
