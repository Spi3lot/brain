package brain.math;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.EqualsAndHashCode;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.BiConsumer;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;

/**
 * @author Emilio Zottel
 * @since 09.09.2024, Mo.
 */
@AllArgsConstructor(access = AccessLevel.PROTECTED)
@EqualsAndHashCode
public abstract class Matrix {

    public final int cols;  // "input size" (for matrix multiplication)

    public final int rows;  // "output size" (for matrix multiplication)

    protected final Vector[] values;

    protected Matrix(Vector... values) {
        Objects.requireNonNull(values);
        this.rows = values.length;
        this.cols = (values.length == 0) ? 0 : values[0].size();
        this.values = values;
    }

    public abstract Matrix add(Matrix m);

    public abstract Matrix sub(Matrix m);

    public abstract Matrix div(float divisor);

    public abstract Matrix mult(float factor);

    public abstract Vector mult(Vector v);

    public abstract Matrix mult(Matrix m);

    public abstract Matrix multHadamard(Matrix m);

    public abstract Matrix transpose();

    public abstract Matrix withEachRow(IntFunction<Vector> function);

    public abstract Vector getCol(int i);

    public void setCol(int i, Vector values) {
        values.check(rows, "Vector size must match matrix row amount");
        forEachRow(j -> set(i, j, values.get(j)));
    }

    public Vector getRow(int j) {
        return values[j];
    }

    public void setRow(int j, Vector values) {
        values.check(cols, "Vector size must match matrix column amount");
        this.values[j] = values;
    }

    public float get(int i, int j) {
        return getRow(j).get(i);
    }

    public void set(int i, int j, float value) {
        values[j].set(i, value);
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

    public void setAll(float... values) {
        if (values.length != cols * rows) {
            throw new IllegalArgumentException("Amount of values must equal 'matrix column amount * matrix row amount'");
        }

        forEachRow(j -> setRow(j, CpuVector.of(Arrays.copyOfRange(values, j * cols, (j + 1) * cols))));
        /*
        for (int v = 0; v < values.length; v++) {
            set(v % cols, v / cols, values[v]);
        }
        */
    }

    public Matrix fillWithRandomValues(float min, float maxExclusive) {
        forEachRow((_, row) -> row.fillWithRandomValues(min, maxExclusive));
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