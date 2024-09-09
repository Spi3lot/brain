package brain.math;

import java.util.Arrays;
import java.util.function.IntFunction;

/**
 * 16.03.2022
 * Emilio Zottel
 * 3CHIF
 */
public class CpuMatrix extends Matrix {

    public CpuMatrix(int cols, int rows) {
        super(cols, rows, CpuVector.makeArray(cols, rows));
    }

    public CpuMatrix(Vector... values) {
        super(values);
    }

    @Override
    public Matrix add(Matrix m) {
        // cols are checked in Vector.add()
        if (rows != m.rows) {
            throw new IllegalArgumentException("Matrix row amount must match");
        }

        return withEachRow(j -> getRow(j).add(m.getRow(j)));
    }

    @Override
    public Matrix sub(Matrix m) {
        // cols are checked in Vector.sub()
        if (rows != m.rows) {
            throw new IllegalArgumentException("Matrix row amount must match");
        }

        return withEachRow(j -> getRow(j).sub(m.getRow(j)));
    }

    @Override
    public Matrix mult(float factor) {
        return withEachRow(j -> getRow(j).mult(factor));
    }

    @Override
    public Vector mult(Vector v) {
        v.check(cols, "Matrix column amount must match vector size");
        return new CpuVector(rows).withEach(j -> v.dot(getRow(j)));
    }

    @Override
    public Matrix mult(Matrix m) {
        if (cols != m.rows) {
            throw new IllegalArgumentException("Matrix column amount must match");
        }

        var result = new CpuMatrix(m.cols, rows);
        var mTransposed = m.transpose();  // More efficient than calling getCol(i) over and over again

        for (int j = 0; j < rows; j++) {
            Vector row = getRow(j);
            float[] resultRow = new float[m.cols];

            for (int i = 0; i < m.cols; i++) {
                // Reading the transposed columns (which are now rows) from the top down
                // Technically they should be read from the bottom up, but since we're transposing, which is a
                // 90° rotation instead of a 270° rotation, the effect cancels out and we can just read from the top down
                resultRow[i] = row.dot(mTransposed.getRow(i));
            }

            result.setRow(j, CpuVector.of(resultRow));
        }

        return result;
    }

    @Override
    public Matrix div(float divisor) {
        return withEachRow(j -> getRow(j).div(divisor));
    }

    @Override
    public Matrix multHadamard(Matrix m) {
        // cols are checked in Vector.mult()
        if (rows != m.rows) {
            throw new IllegalArgumentException("Matrix row amount must match");
        }

        return withEachRow(j -> getRow(j).mult(m.getRow(j)));
    }

    @Override
    public Matrix transpose() {
        // Notice: Matrix constructor is reversed, normally it is used like 'new Matrix(cols, rows)'
        return new CpuMatrix(rows, cols).withEachRow(this::getCol);
    }

    @Override
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

    @Override
    public Vector getCol(int i) {
        return new CpuVector(rows).withEach(j -> get(i, j));
    }

    @Override
    public Matrix withEachRow(IntFunction<Vector> function) {
        var m = new CpuMatrix(cols, rows);
        m.setEachRow(function);
        return m;
    }

}
