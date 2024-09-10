package brain.math;

import brain.domain.Brain;
import lombok.EqualsAndHashCode;
import lombok.ToString;

import java.util.function.BiConsumer;
import java.util.function.IntConsumer;
import java.util.function.IntFunction;
import java.util.function.UnaryOperator;

/**
 * @author Emilio Zottel
 * @since 09.09.2024, Mo.
 */
@EqualsAndHashCode
@ToString
public abstract class Vector {

    protected final float[] values;

    protected Vector(int size) {
        values = new float[size];
    }

    // protected and float[] instead of float... to avoid possible overlappings with the protected brain.math.Vector(int) constructor
    protected Vector(float[] values) {
        this.values = values;
    }

    protected static Vector[] makeArray(int cols, int rows, IntFunction<Vector> constructor) {
        var arr = new Vector[rows];

        for (int i = 0; i < rows; i++) {
            arr[i] = constructor.apply(cols);
        }

        return arr;
    }

    public abstract Matrix toRowVector();

    public abstract Vector negate();

    public abstract Vector add(Vector v);

    public abstract Vector sub(Vector v);

    public abstract Vector mult(float factor);

    public abstract Vector mult(Vector v);

    public abstract Matrix mult(Matrix rowVector);

    public abstract Vector div(float divisor);

//    public Vector mult(Matrix m) {
//        return m.transpose().mult(this);
//    }

    public abstract float dot(Vector v);

    public abstract Vector withEach(IntFunction<Float> function);

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
        check(v.size());
        setEach(v::get);
    }

    public void setEach(IntFunction<Float> function) {
        for (int i = 0; i < size(); i++) {
            set(i, function.apply(i));
        }
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
        setEach(_ -> Brain.RANDOM.nextFloat(maxExclusive - min) + min);
        return this;
    }

    protected void check(int len) {
        check(len, "Vector sizes must match");
    }

    protected void check(int len, String message) {
        if (len != size()) {
            throw new IllegalArgumentException(message);
        }
    }

}
