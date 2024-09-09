package brain.domain;

import brain.math.*;
import brain.misc.LayerDefinition;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public class CpuBrainTest implements BrainTest {

    @Test
    public void brain_test() {
        assertEquals(
                13002,
                new Brain(
                        CpuVector::new,
                        CpuMatrix::new,
                        new LayerDefinition(28 * 28, ActivationFunction.RELU),
                        new LayerDefinition(16, ActivationFunction.RELU),
                        new LayerDefinition(16, ActivationFunction.RELU),
                        new LayerDefinition(10, ActivationFunction.RELU)
                ).totalSize()
        );

        Brain brain = new Brain(
                CpuVector::new,
                CpuMatrix::new,
                new LayerDefinition(28 * 28, ActivationFunction.RELU),
                new LayerDefinition(16, ActivationFunction.RELU),
                new LayerDefinition(10, ActivationFunction.RELU)
        );

        Vector inputs = new CpuVector(28 * 28).fillWithRandomValues(-1, 1);
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
        CpuMatrix m1 = new CpuMatrix(
                CpuVector.of(9, 8, 7),
                CpuVector.of(6, 5, 4),
                CpuVector.of(3, 2, 1)
        );

        CpuMatrix m2 = new CpuMatrix(
                CpuVector.of(1, 2, 3, 4),
                CpuVector.of(5, 6, 7, 8),
                CpuVector.of(9, 10, 11, 12)
        );

        Matrix m1_m2 = new CpuMatrix(
                CpuVector.of(112, 136, 160, 184),
                CpuVector.of(67, 82, 97, 112),
                CpuVector.of(22, 28, 34, 40)
        );

        var v1 = CpuVector.of(1, 2, 3, 4);
        var v2 = CpuVector.of(5, 4, 3, 2, 1);

        Matrix v1_v2r = new CpuMatrix(
                CpuVector.of(5, 4, 3, 2, 1),
                CpuVector.of(10, 8, 6, 4, 2),
                CpuVector.of(15, 12, 9, 6, 3),
                CpuVector.of(20, 16, 12, 8, 4)
        );

        assertEquals(m1_m2, m1.mult(m2));
        assertThrows(IllegalArgumentException.class, () -> m2.mult(m1));
        assertEquals(v1_v2r, v1.mult(v2.toRowVector()));
        assertThrows(IllegalArgumentException.class, () -> v2.toRowVector().mult(v1));
    }

}
