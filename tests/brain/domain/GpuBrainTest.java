package brain.domain;

import brain.math.*;
import brain.misc.LayerDefinition;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

/**
 * @author Emilio Zottel
 * @since 09.09.2024, Mo.
 */
public class GpuBrainTest implements BrainTest {

    @Test
    public void brain_test() {
        assertEquals(
                13002,
                new Brain(
                        GpuVector::new,
                        GpuMatrix::new,
                        new LayerDefinition(28 * 28, ActivationFunction.RELU),
                        new LayerDefinition(16, ActivationFunction.RELU),
                        new LayerDefinition(16, ActivationFunction.RELU),
                        new LayerDefinition(10, ActivationFunction.RELU)
                ).totalSize()
        );

        Brain brain = new Brain(
                GpuVector::new,
                GpuMatrix::new,
                new LayerDefinition(28 * 28, ActivationFunction.RELU),
                new LayerDefinition(16, ActivationFunction.RELU),
                new LayerDefinition(10, ActivationFunction.RELU)
        );

        Vector inputs = new GpuVector(28 * 28).fillWithRandomValues(-1, 1);
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
        GpuMatrix m1 = new GpuMatrix(
                GpuVector.of(9, 8, 7),
                GpuVector.of(6, 5, 4),
                GpuVector.of(3, 2, 1)
        );

        GpuMatrix m2 = new GpuMatrix(
                GpuVector.of(1, 2, 3, 4),
                GpuVector.of(5, 6, 7, 8),
                GpuVector.of(9, 10, 11, 12)
        );

        Matrix m1_m2 = new GpuMatrix(
                GpuVector.of(112, 136, 160, 184),
                GpuVector.of(67, 82, 97, 112),
                GpuVector.of(22, 28, 34, 40)
        );

        var v1 = GpuVector.of(1, 2, 3, 4);
        var v2 = GpuVector.of(5, 4, 3, 2, 1);

        Matrix v1_v2r = new GpuMatrix(
                GpuVector.of(5, 4, 3, 2, 1),
                GpuVector.of(10, 8, 6, 4, 2),
                GpuVector.of(15, 12, 9, 6, 3),
                GpuVector.of(20, 16, 12, 8, 4)
        );

        assertEquals(m1_m2, m1.mult(m2));
        assertThrows(IllegalArgumentException.class, () -> m2.mult(m1));
        assertEquals(v1_v2r, v1.mult(v2.toRowVector()));
        assertThrows(IllegalArgumentException.class, () -> v2.toRowVector().mult(v1));
    }

}
