package brain.math;

import org.junit.Assert;
import org.junit.Test;

/**
 * @author Emilio Zottel
 * @since 09.09.2024, Mo.
 */
public class GpuVectorTest implements VectorTest {

    private static final Vector VECTOR_1 = new GpuVector(SIZE).withEach(i -> (float) i);

    private static final Vector VECTOR_2 = new GpuVector(SIZE).withEach(i -> (float) i);

    private static final Vector EXPECTED_NEGATE = new GpuVector(SIZE).withEach(i -> -(float) i);

    private static final Vector EXPECTED_ADD = new GpuVector(SIZE).withEach(i -> (float) (i + i));

    private static final Vector EXPECTED_SUB = new GpuVector(SIZE);

    private static final Vector EXPECTED_MULT = new GpuVector(SIZE).withEach(i -> (float) (i * i));

    private static final Vector EXPECTED_MULT_FACTOR = new GpuVector(SIZE).withEach(i -> i * 2f);

    private static final Vector EXPECTED_MULT_VECTOR = new GpuVector(SIZE).withEach(i -> (float) (i * i));

    private static final Vector EXPECTED_DIV = new GpuVector(SIZE).withEach(i -> i / 2f);

    private static final double EXPECTED_DOT = (double) SIZE * (SIZE - 1) * (2 * SIZE - 1) / 6;

    @Test
    @Override
    public void testNegate() {
        Assert.assertEquals(EXPECTED_NEGATE, VECTOR_1.negate());
    }

    @Test
    @Override
    public void testAdd() {
        Assert.assertEquals(EXPECTED_ADD, VECTOR_1.add(VECTOR_2));
    }

    @Test
    @Override
    public void testSub() {
        Assert.assertEquals(EXPECTED_SUB, VECTOR_1.sub(VECTOR_2));
    }

    @Test
    @Override
    public void testMult() {
        Assert.assertEquals(EXPECTED_MULT, VECTOR_1.mult(VECTOR_2));
    }

    @Test
    @Override
    public void testMultFactor() {
        Assert.assertEquals(EXPECTED_MULT_FACTOR, VECTOR_1.mult(2));
    }

    @Test
    @Override
    public void testMultVector() {
        Assert.assertEquals(EXPECTED_MULT_VECTOR, VECTOR_1.mult(VECTOR_1));
    }

    @Test
    @Override
    public void testMultMatrix() {
        // Not implemented
    }

    @Test
    @Override
    public void testDiv() {
        Assert.assertEquals(EXPECTED_DIV, VECTOR_1.div(2));
    }

    @Test
    @Override
    public void testDot() {
        Assert.assertEquals(EXPECTED_DOT, VECTOR_1.dot(VECTOR_2), 0.0001);
    }

    @Test
    @Override
    public void testSetAll() {
        var v1 = new GpuVector(1 << 28).fillWithRandomValues(-1, 1);
        var v2 = new GpuVector(1 << 28);
        v2.setAll(v1);
        Assert.assertEquals(v1, v2);
    }

}