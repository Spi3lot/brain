package brain.domain;

import brain.math.ActivationFunction;
import brain.math.Matrix;
import brain.math.Vector;
import brain.misc.LayerDefinition;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public class BrainTest {

    @Test
    public void test() {
        assertEquals(
                13002,
                new Brain(
                        new LayerDefinition(28 * 28, ActivationFunction.RELU),
                        new LayerDefinition(16, ActivationFunction.RELU),
                        new LayerDefinition(16, ActivationFunction.RELU),
                        new LayerDefinition(10, ActivationFunction.RELU)
                ).totalSize()
        );

        Brain brain = new Brain(
                new LayerDefinition(28 * 28, ActivationFunction.RELU),
                new LayerDefinition(16, ActivationFunction.RELU),
                new LayerDefinition(10, ActivationFunction.RELU)
        );

        Vector inputs = new Vector(28 * 28).fillWithRandomValues(-1, 1);
        Vector actual = brain.predict(inputs);

        Vector expected = brain.getLayer(2)
                .getWeights()
                .mult(brain.getLayer(1)
                        .getWeights()
                        .mult(inputs)
                        .add(brain.getLayer(1)
                                .getBiases())
                        .map(brain.getLayer(1)
                                .getActivationFunction()::apply))
                .add(brain.getLayer(2)
                        .getBiases())
                .map(brain.getLayer(2)
                        .getActivationFunction()::apply
                );

        assertEquals(expected, actual);
    }

    @Test
    public void matmul_test() {
        Matrix m1 = new Matrix(
                Vector.of(9, 8, 7),
                Vector.of(6, 5, 4),
                Vector.of(3, 2, 1)
        );

        Matrix m2 = new Matrix(
                Vector.of(1, 2, 3, 4),
                Vector.of(5, 6, 7, 8),
                Vector.of(9, 10, 11, 12)
        );

        Matrix m1_m2 = new Matrix(
                Vector.of(112, 136, 160, 184),
                Vector.of(67, 82, 97, 112),
                Vector.of(22, 28, 34, 40)
        );

        Vector v1 = Vector.of(1, 2, 3, 4);
        Vector v2 = Vector.of(5, 4, 3, 2, 1);

        Matrix v1_v2r = new Matrix(
                Vector.of(5, 4, 3, 2, 1),
                Vector.of(10, 8, 6, 4, 2),
                Vector.of(15, 12, 9, 6, 3),
                Vector.of(20, 16, 12, 8, 4)
        );

        assertEquals(m1_m2, m1.mult(m2));
        assertThrows(AssertionError.class, () -> m2.mult(m1));
        assertEquals(v1_v2r, v1.mult(v2.toRowVector()));
        assertThrows(AssertionError.class, () -> v2.toRowVector().mult(v1));
    }

}
