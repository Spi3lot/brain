package brain.domain;

import brain.math.Matrix;
import brain.math.Vector;
import brain.misc.*;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;

import java.util.Arrays;
import java.util.Random;
import java.util.function.BiConsumer;

/**
 * 16.03.2022
 * Emilio Zottel
 * 3CHIF
 */
@RequiredArgsConstructor
@Getter
@Setter
public class Brain {

    public static final long SEED = 12345;
    public static final Random RANDOM = new Random();  // SEED not used here
    private final Layer[] layers;
    private float learningRate = 1.0f;
    private int miniBatchSize = 100;

    // Weight initialization:
    // https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    public Brain(LayerDefinition... layerDefinitions) {
        layers = new Layer[layerDefinitions.length];
        layers[0] = new Layer(layerDefinitions[0]);

        for (int i = 1; i < size(); i++) {
            var currentLayerDefinition = layerDefinitions[i];
            int previousLayerSize = layers[i - 1].size();
            float min = (float) (-Math.sqrt(6 / Math.sqrt(previousLayerSize + currentLayerDefinition.size())));
            float maxExcl = -min;
            layers[i] = new Layer(currentLayerDefinition, previousLayerSize, min, maxExcl);
        }
    }

    /**
     * @param testExamples inputs to be tested and desired index of the highest activation of the output layer for each test input
     */
    public void test(TestExample[] testExamples/*, Function like argmax/round/...*/) {
        for (TestExample testExample : testExamples) {
            Vector output = predict(testExample.input());
            int argmax = output.argmax();

            if (argmax != testExample.label())
                throw new RuntimeException(STR."Expected: \{testExample.label()}\nActual: \{argmax}\n\nOutputs: \{output}");
            //assert argmax == testExample.label() : "Expected: " + testExample.label() + "\nActual: " + argmax + "\n\nOutputs: " + output;
        }
    }

    public Vector predict(Vector input) {
        Layer curr = getInputLayer();
        curr.activate(input);

        for (int i = 1; i < size(); i++) {
            Layer next = getLayer(i);
            curr.feedforward(next);
            curr = next;
        }

        return getOutputLayer().getActivations();
    }

    public Vector predict(float... input) {
        return predict(Vector.of(input));
    }

    public void train(TrainingExample[] trainingExamples) {
        MiniBatch[] miniBatches = MiniBatch.shuffleAndChop(miniBatchSize, trainingExamples);

        for (MiniBatch miniBatch : miniBatches) {
            WeightBias[] step = new WeightBias[size() - 1];

            for (int i = 0; i < miniBatch.size(); i++) {
                WeightBias[] deltas = backpropagate(miniBatch.getExample(i));
                WeightBias.add(step, deltas);  // Adding the deltas to our steps
            }

            // Averaging the deltas and multiplying with the learning rate, then "stepping downhill"
            WeightBias.mult(step, learningRate / miniBatch.size());
            sub(step);
        }
    }

    private WeightBias[] backpropagate(TrainingExample trainingExample) {
        WeightBias[] deltas = new WeightBias[size() - 1];
        Vector output = predict(trainingExample.input());
        Vector error = output.sub(trainingExample.target());  // Derivative of the cost function 1/2 * (o - t)²
        Layer curr = getOutputLayer();

        for (int i = outputLayerIndex() - 1; i >= 0; i--) {
            Layer prev = getLayer(i);
            Vector nablaBiases = curr.getNablaBiases(error, prev.getActivationFunction());
            Matrix nablaWeights = nablaBiases.mult(prev.getActivations().toRowVector());
            deltas[i] = new WeightBias(nablaWeights, nablaBiases);

            if (i > 0) {
                error = curr.getWeights().transpose().mult(error);
                curr = prev;
            }
        }

        return deltas;
    }

    private void add(WeightBias[] deltas) {
        assert deltas.length == size() - 1;
        forEachHiddenLayer((i, layer) -> layer.add(deltas[i - 1]));
    }

    private void sub(WeightBias[] deltas) {
        assert deltas.length == size() - 1;
        forEachHiddenLayer((i, layer) -> layer.sub(deltas[i - 1]));
    }

    /**
     * Includes the output layer
     *
     * @param biConsumer biConsumer to be executed for every layer except for the input layer
     */
    private void forEachHiddenLayer(BiConsumer<Integer, Layer> biConsumer) {
        for (int i = 1; i < size(); i++) {
            biConsumer.accept(i, getLayer(i));
        }
    }

    public int size() {
        return layers.length;
    }

    public int totalSize() {
        int size = 0;

        for (int i = 0; i < outputLayerIndex(); i++) {
            size += (getLayer(i).size() + 1) * getLayer(i + 1).size();  // size()+1 because of the biases
        }

        return size;
    }


    int outputLayerIndex() {
        return layers.length - 1;
    }

    int maxLayerSize() {
        return Arrays.stream(layers)
                .mapToInt(Layer::size)
                .max()
                .orElseThrow();
    }

    Layer getLayer(int i) {
        return layers[i];
    }

    Layer getInputLayer() {
        return layers[0];
    }

    Layer getOutputLayer() {
        return layers[outputLayerIndex()];
    }


    public String getLayerComment(int i) {
        StringBuilder sb = new StringBuilder(32);  // 32 > 27 = len("// --- Layer N : Output ---")
        sb.append("// --- Layer ").append(i);

        if (i == 0) {
            sb.append(" : Input");
        } else if (i == outputLayerIndex()) {
            sb.append(" : Output");
        }

        sb.append(" ---");
        return sb.toString();
    }

    /**
     * Best use case: <a href="https://shadertoy.com">Shadertoy</a>
     */
    public String generateSourceCode() {
        StringBuilder sb = new StringBuilder(totalSize());

        for (int i = 0; i < size(); i++) {
            sb.append(getLayerComment(i)).append('\n');
            sb.append(getLayer(i).generateSourceCode(i));
        }

        return sb.toString();
    }

    @Override
    public String toString() {
        var stringBuilder = new StringBuilder();

        for (int j = 0; j < maxLayerSize(); j++) {
            for (int i = 0; i < size(); i++) {
                if (j < layers[i].size()) {
                    stringBuilder.append(layers[i].getActivations().get(j)).append('\t');
                } else {
                    stringBuilder.append("#\t");
                }
            }

            stringBuilder.append('\n');
        }

        return stringBuilder.toString();
    }

}
