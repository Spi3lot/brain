package brain.misc;

import brain.math.Matrix;
import brain.math.Vector;
import lombok.Getter;

/**
 * 17.07.2022
 * Emilio Zottel
 * 3CHIF-4AHIF
 */
@Getter
public class WeightBias {

    private Matrix weights;
    private Vector biases;

    public WeightBias(Matrix weights, Vector biases) {
        if (weights.rows != biases.size()) {
            throw new IllegalArgumentException("Matrix row amount must match vector size");
        }

        this.weights = weights;
        this.biases = biases;
    }

    public static void add(WeightBias[] arr1, WeightBias[] arr2) {
        int len = arr1.length;

        if (len != arr2.length) {
            throw new IllegalArgumentException("Array lengths must match");
        }

        for (int i = 0; i < len; i++) {
            WeightBias wb1 = arr1[i];

            if (wb1 == null) {
                WeightBias wb2 = arr2[i];
                arr1[i] = new WeightBias(wb2.weights, wb2.biases);
                continue;
            }

            wb1.add(arr2[i]);
        }
    }

    public static void mult(WeightBias[] arr, float factor) {
        for (WeightBias weightBias : arr) {
            weightBias.mult(factor);
        }
    }

    public int inputs() {
        return weights.cols;
    }

    public int outputs() {
        return weights.rows;
        //return biases.size();  // Same as above
    }

    /*
    // TODO maybe?
    public static WeightBias[] makeArray(int... layerSizes) {
        WeightBias[] arr = new WeightBias[length];

        for (int i = 0; i < length; i++) {
            arr[i] = new WeightBias();
        }

        return arr;
    }
     */

    public Vector apply(Vector activations) {
        return weights.mult(activations).add(biases);
    }

    public void add(WeightBias delta) {
        weights = weights.add(delta.weights);
        biases = biases.add(delta.biases);
    }

    public void sub(WeightBias delta) {
        weights = weights.sub(delta.weights);
        biases = biases.sub(delta.biases);
    }

    public void mult(float divisor) {
        weights = weights.mult(divisor);
        biases = biases.mult(divisor);
    }

}
