package brain.math;

import java.util.function.IntFunction;

/**
 * @author Emilio Zottel
 * @since 09.09.2024, Mo.
 */
public class GpuMatrix extends Matrix {

    static {
        System.loadLibrary("natives/matrix");
    }

    public GpuMatrix(int cols, int rows) {
        super(cols, rows, GpuVector.makeArray(cols, rows));
    }

    public GpuMatrix(Vector... values) {
        super(values);
    }

    @Override
    public native Matrix add(Matrix m);

    @Override
    public native Matrix sub(Matrix m);

    @Override
    public native Matrix div(float divisor);

    @Override
    public native Matrix mult(float factor);

    @Override
    public native Vector mult(Vector v);

    @Override
    public native Matrix mult(Matrix m);

    @Override
    public native Matrix multHadamard(Matrix m);

    @Override
    public native Matrix transpose();

    @Override
    public Vector getCol(int i) {
        return new GpuVector(rows).withEach(j -> get(i, j));
    }

    @Override
    public Matrix withEachRow(IntFunction<Vector> function) {
        var m = new GpuMatrix(cols, rows);
        m.setEachRow(function);
        return m;
    }

}