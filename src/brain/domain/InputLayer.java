package brain.domain;

public class InputLayer extends Layer {
    // The input layer has no predecessor, which means its weights and biases are completely useless
    // because they can't be multiplied with anything
    public InputLayer(int size) {
        super(size);
    }
}
