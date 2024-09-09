package brain.domain;

import brain.math.*;
import brain.misc.LayerDefinition;
import brain.misc.WeightBias;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.util.StringJoiner;
import java.util.function.BiFunction;
import java.util.function.IntFunction;

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
                 float maxExcl,
                 IntFunction<Vector> vectorConstructor,
                 BiFunction<Integer, Integer, Matrix> matrixConstructor) {
        this(
                new WeightBias(
                        matrixConstructor.apply(previousLayerSize, layerDefinition.size()).fillWithRandomValues(min, maxExcl),
                        vectorConstructor.apply(layerDefinition.size()).fillWithRandomValues(min, maxExcl)
                ),
                layerDefinition.activationFunction(),
                vectorConstructor
        );
    }

    public Layer(WeightBias weightBias,
                 ActivationFunction activationFunction,
                 IntFunction<Vector> vectorContructor) {
        this(
                weightBias,
                vectorContructor.apply(weightBias.outputs()),
                vectorContructor.apply(weightBias.outputs()),
                activationFunction
        );
    }

    // Only for the InputLayer
    Layer(LayerDefinition layerDefinition, IntFunction<Vector> vectorConstructor) {
        this(
                null,
                vectorConstructor.apply(layerDefinition.size()),
                vectorConstructor.apply(layerDefinition.size()),
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
        next.activate(next.weightBias.apply(activations));
    }

    public void activate(Vector activationsLinear) {
        setActivationsLinear(activationsLinear);
        setActivations(activationsLinear.map(activationFunction::apply));
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

    private void setActivationsLinear(Vector activationsLinear) {
        this.activationsLinear.setAll(activationsLinear);
    }

    private void setActivations(Vector activations) {
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
        var joiner = new StringJoiner("\n");

        for (int i = 0; i < size(); i++) {
            joiner.add(STR."\{activations.get(i)}");
        }

        return joiner.toString();
    }

}
