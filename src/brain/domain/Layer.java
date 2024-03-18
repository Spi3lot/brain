package brain.domain;

import brain.math.ActivationFunction;
import brain.math.Matrix;
import brain.math.Vector;
import brain.misc.WeightBias;
import lombok.Getter;

import java.util.StringJoiner;

/**
 * 16.03.2022
 * Emilio Zottel
 * 3CHIF
 */
// Layer of Neurons
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
    @Getter
    private final Vector activationsLinear;

    /**
     * -- GETTER --
     * Always contains the activations of the previous prediction<br>
     * All zeros if there have not been any predictions yet
     */
    @Getter
    private final Vector activations;

    // Only for the InputLayer
    protected Layer(int size) {
        weightBias = null;
        activationsLinear = new Vector(size);
        activations = new Vector(size);
    }

    public Layer(WeightBias weightBias) {
        assert weightBias != null;

        this.weightBias = weightBias;
        this.activationsLinear = new Vector(weightBias.outputs());
        this.activations = new Vector(weightBias.outputs());
    }


    public Layer(int previousLayerSize, int currentLayerSize, float min, float maxExcl) {
        Matrix weights = new Matrix(previousLayerSize, currentLayerSize).fillWithRandomValues(min, maxExcl);
        Vector biases = new Vector(currentLayerSize).fillWithRandomValues(min, maxExcl);

        weightBias = new WeightBias(weights, biases);
        activationsLinear = new Vector(currentLayerSize);
        activations = new Vector(currentLayerSize);
    }


    public void add(WeightBias delta) {
        weightBias.add(delta);
    }

    public void sub(WeightBias delta) {
        weightBias.sub(delta);
    }

    public void feedforward(Layer next, ActivationFunction activationFunction) {
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
