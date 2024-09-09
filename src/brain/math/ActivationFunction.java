package brain.math;

import lombok.AllArgsConstructor;

import java.util.function.UnaryOperator;

/**
 * 13.07.2022
 * Emilio Zottel
 * 3CHIF-4AHIF
 */
@AllArgsConstructor
public enum ActivationFunction {

    LINEAR(
            x -> x,
            _ -> 1.0f
    ),
    SIGMOID(ActivationFunction::sigmoid, x -> {
        float s = sigmoid(x);
        return s * (1.0f - s);
    }),
    TANH(ActivationFunction::tanh, x -> {
        float tanh = tanh(x);
        return 1.0f - tanh * tanh;
    }),
    RELU(
            ActivationFunction::relu,
            x -> (x < 0) ? 0.0f : 1.0f   // Math.max(0, Math.signum(x))
    ),
    LRELU(
            ActivationFunction::lrelu,
            x -> (x < 0) ? 0.5f : 1.0f
    ),
    ELU(
            ActivationFunction::elu,
            x -> (x < 0) ? exp(x) : 1.0f
    );

    private final UnaryOperator<Float> function;

    private final UnaryOperator<Float> derivative;

    private static float exp(float x) {
        return (float) Math.exp(x);
    }

    private static float tanh(float x) {
        return (float) Math.tanh(x);
    }

    /**
     * <b>Rectified Linear Unit</b>
     */
    private static float relu(float x) {
        return Math.max(0, x);
    }

    /**
     * <b>Leaky Rectified Linear Unit</b>
     */
    private static float lrelu(float x) {
        return x * ((x <= 0) ? 0.5f : 1.0f);
    }

    /**
     * <b>Exponential Linear Unit<br></b>
     * The following text only applies to the commented out version of ELU:<br>
     * Its derivative with respect to x is equal to the sigmoid squishification function, which means that <br>
     * ELU(x) is equal to the integral of sigmoid(x) with respect to x<br>
     * <br>
     * ELU(x) = ∫ σ(x) dx<br>
     * ELU'(x) = σ(x)<br>
     * ELU''(x) = σ'(x) = σ(x) *  σ(-x) * (1 - σ(x))<br>
     */
    private static float elu(float x) {
        return (x < 0) ? exp(x) - 1.0f : x;

        // Expression inside the log blows up too fast
        //return (float) (Math.log1p(Math.exp(x)) - 1.0);  // ln(1 + e^x) - 1
    }

    /**
     * <b>Sigmoid</b> squishification function<br>
     * It is the derivative of the ELU whichs means that<br>
     * the integral of sigmoid(x) is equal to ELU(x)<br>
     * <br>
     * <p>
     * ∫ σ(x) dx = ELU(x)<br>
     * σ(x) = ELU'(x)<br>
     * σ'(x) = σ(x) * σ(-x) = σ(x) * (1 - σ(x)) = ELU''(x)
     */
    private static float sigmoid(float x) {
        return 1.0f / (1.0f + exp(-x));
    }

    public float apply(float x) {
        return function.apply(x);
    }

    public float applyDerivative(float x) {
        return derivative.apply(x);
    }

}
