package brain.math;

import lombok.EqualsAndHashCode;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;

/**
 * 16.03.2022
 * Emilio Zottel
 * 3CHIF
 */
@EqualsAndHashCode
public class Matrix {
    public final int cols, rows;
    private final Vector[] values;

    public Matrix(int cols, int rows) {
        this.cols = cols;  // "input size" (for matrix multiplication)
        this.rows = rows;  // "output size" (for matrix multiplication)
        this.values = Vector.makeArray(cols, rows);
    }

    public Matrix(Vector... values) {
        assert values != null;

        this.rows = values.length;
        this.cols = (values.length == 0) ? 0 : values[0].size();
        this.values = values;
    }


    public Matrix add(Matrix m) {
        assert rows == m.rows;  // cols are checked in Vector.add()
        return withEachRow(j -> getRow(j).add(m.getRow(j)));
    }

    public Matrix sub(Matrix m) {
        assert rows == m.rows;  // cols are checked in Vector.sub()
        return withEachRow(j -> getRow(j).sub(m.getRow(j)));
    }

    public Matrix mult(float factor) {
        return withEachRow(j -> getRow(j).mult(factor));
    }

    public Vector mult(Vector v) {
        v.check(cols, "Matrix column amount must match vector size");
        return new Vector(rows).withEach(j -> v.dot(getRow(j)));
    }

    public Matrix mult(Matrix m) {
        assert cols == m.rows;
        Matrix result = new Matrix(m.cols, rows);
        Matrix mTransposed = m.transpose();  // More efficient than calling getCol(i) over and over again

        for (int j = 0; j < rows; j++) {
            Vector row = getRow(j);
            float[] resultRow = new float[m.cols];

            for (int i = 0; i < m.cols; i++) {
                // Reading the transposed columns (which are now rows) from the top down
                // Technically they should be read from the bottom up, but since we're transposing, which is a
                // 90° rotation instead of a 270° rotation, the effect cancels out and we can just read from the top down
                resultRow[i] = row.dot(mTransposed.getRow(i));
            }

            result.setRow(j, Vector.of(resultRow));
        }

        return result;
    }

    public Matrix div(float divisor) {
        return withEachRow(j -> getRow(j).div(divisor));
    }

    public Matrix multHadamard(Matrix m) {
        assert rows == m.rows;  // cols are checked in Vector.mult()
        return withEachRow(j -> getRow(j).mult(m.getRow(j)));
    }

    public Matrix transpose() {
        // Notice: Matrix constructor is reversed, normally it is used like 'new Matrix(cols, rows)'
        return new Matrix(rows, cols).withEachRow(this::getCol);
    }

    public Vector getCol(int i) {
        return new Vector(rows).withEach(j -> get(i, j));
    }

    public void setCol(int i, Vector values) {
        assert values.size() == rows : "Vector size must match matrix row amount";
        forEachRow(j -> set(i, j, values.get(j)));
    }

    public Vector getRow(int j) {
        return values[j];
    }

    public void setRow(int j, Vector values) {
        assert values.size() == cols : "Vector size must match matrix column amount";
        this.values[j] = values;
    }

    public float get(int i, int j) {
        return getRow(j).get(i);
    }

    public void set(int i, int j, float value) {
        values[j].set(i, value);
    }

    public void setAll(float... values) {
        assert values.length == cols * rows : "Amount of values must equal 'matrix column amount * matrix row amount'";
        forEachRow(j -> setRow(j, Vector.of(Arrays.copyOfRange(values, j * cols, (j + 1) * cols))));
        /*
        for (int v = 0; v < values.length; v++) {
            set(v % cols, v / cols, values[v]);
        }
        */
    }

    public void forEachRow(IntConsumer consumer) {
        for (int j = 0; j < rows; j++) {
            consumer.accept(j);
        }
    }

    public void forEachRow(BiConsumer<Integer, Vector> biConsumer) {
        forEachRow(j -> biConsumer.accept(j, getRow(j)));
    }

    public void setEachRow(IntFunction<Vector> function) {
        forEachRow(j -> setRow(j, function.apply(j)));
    }

    public Matrix withEachRow(IntFunction<Vector> function) {
        Matrix m = new Matrix(cols, rows);
        m.setEachRow(function);

        return m;
    }

    public Matrix fillWithRandomValues(float min, float maxExclusive) {
        forEachRow((j, row) -> row.fillWithRandomValues(min, maxExclusive));
        return this;
    }


    // TODO: matrix from and to bytes
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(1 + rows * (2 * cols - 1));
        //StringBuilder sb = new StringBuilder(1 + rows * cols + rows * (cols - 1));
        //StringBuilder sb = new StringBuilder(1 + rows * cols + (rows - 1) * (cols - 1));

        for (int j = 0; j < rows; j++) {
            Vector row = getRow(j);

            for (int i = 0; i < cols; i++) {
                sb.append(row.get(i)).append(" ");
            }

            sb.append('\n');
        }

        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }
}
