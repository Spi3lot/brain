package brain.domain;

import brain.math.ActivationFunction;
import brain.math.Matrix;
import brain.math.Vector;
import brain.misc.LayerDefinition;
import brain.misc.WeightBias;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.StringJoiner;

/**
 * 16.03.2022
 * Emilio Zottel
 * 3CHIF
 */
// Layer of Neurons
@RequiredArgsConstructor
@Getter
public class Layer {

    /**
     * Weights and biases that will be multiplied with the activations of the neurons of the PREVIOUS layer
     * The activation, a_j, of the current neuron will be evaluated as W_j * a_[j-1] + b_j
     * This makes it easier to do calculus using these variables, since changing
     * W_j or b_j influences the value of a_j, not the value of a_[j+1]
     */
    private final WeightBias weightBias;

    /**
     * -- GETTER --
     * Always contains the linear activations of the previous prediction<br>
     * All zeros if there have not been any predictions yet
     */
    private final Vector activationsLinear;

    /**
     * -- GETTER --
     * Always contains the activations of the previous prediction<br>
     * All zeros if there have not been any predictions yet
     */
    private final Vector activations;

    private final ActivationFunction activationFunction;

    public Layer(LayerDefinition layerDefinition,
                 int previousLayerSize,
                 float min,
                 float maxExcl) {
        this(
                new WeightBias(
                        new Matrix(previousLayerSize, layerDefinition.size()).fillWithRandomValues(min, maxExcl),
                        new Vector(layerDefinition.size()).fillWithRandomValues(min, maxExcl)
                ),
                layerDefinition.activationFunction()
        );
    }

    public Layer(WeightBias weightBias, ActivationFunction activationFunction) {
        this(
                weightBias,
                new Vector(weightBias.outputs()),
                new Vector(weightBias.outputs()),
                activationFunction
        );
    }

    // Only for the InputLayer
    Layer(LayerDefinition layerDefinition) {
        this(
                null,
                new Vector(layerDefinition.size()),
                new Vector(layerDefinition.size()),
                layerDefinition.activationFunction()
        );
    }

    public void add(WeightBias delta) {
        weightBias.add(delta);
    }

    public void sub(WeightBias delta) {
        weightBias.sub(delta);
    }

    public void feedforward(Layer next) {
        Vector z = next.weightBias.apply(activations);
        Vector a = z.map(activationFunction::apply);

        next.setActivationsLinear(z);
        next.setActivations(a);
    }

    public Vector getNablaBiases(Vector deltas, ActivationFunction activationFunction) {
        return activationsLinear.withEach(i -> {
            float z = activationsLinear.get(i);
            float do_dz = activationFunction.applyDerivative(z);
            return do_dz * deltas.get(i);
        });
    }

    public Matrix getWeights() {
        return weightBias.getWeights();
    }

    public Vector getBiases() {
        return weightBias.getBiases();
    }

    public int size() {
        return activations.size();
        //return weightBias.outputs();  // Same as above
    }

    public void setActivationsLinear(Vector activationsLinear) {
        this.activationsLinear.setAll(activationsLinear);
    }

    public void setActivations(Vector activations) {
        this.activations.setAll(activations);
    }

    /**
     * @param i index of this layer in the domain so that the variables can be properly named
     * @return C-style code that will calculate the activations of this layer as single float values
     */
    public String generateSourceCode(int i) {
        throw new UnsupportedOperationException("Not implemented yet");
    }


    @Override
    public String toString() {
        StringJoiner joiner = new StringJoiner("\n");

        for (int i = 0; i < size(); i++) {
            joiner.add(STR."\{activations.get(i)}");
        }

        return joiner.toString();
    }

}
