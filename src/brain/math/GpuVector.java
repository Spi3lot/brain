package brain.math;

import java.util.function.IntFunction;

/**
 * @author Emilio Zottel
 * @since 09.09.2024, Mo.
 */
public class GpuVector extends Vector {

    static {
        System.loadLibrary("vector");
    }

    public GpuVector(int size) {
        super(size);
    }

    public GpuVector(float... values) {
        super(values);
    }

    /**
     * Useful in Matrix.java when reading n floats per row and constructing a Vector using these floats
     */
    public static GpuVector of(float... values) {
        return new GpuVector(values);
    }

    public static Vector[] makeArray(int cols, int rows) {
        return Vector.makeArray(cols, rows, GpuVector::new);
    }

    @Override
    public Matrix toRowVector() {
        return new GpuMatrix(this);
    }

    @Override
    public native Vector negate();

    @Override
    public native Vector add(Vector v);

    @Override
    public native Vector sub(Vector v);

    @Override
    public native Vector mult(float factor);

    @Override
    public native Vector mult(Vector v);

    @Override
    public native Matrix mult(Matrix rowVector);

    @Override
    public native Vector div(float divisor);

    @Override
    public native float dot(Vector v);

    @Override
    public native void setAll(Vector v);

    @Override
    public Vector withEach(IntFunction<Float> function) {
        Vector v = new GpuVector(size());
        v.setEach(function);
        return v;
    }

}
