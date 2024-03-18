package brain.misc;

import brain.math.Vector;

/**
 * 17.07.2022
 * Emilio Zottel
 * 3CHIF-4AHIF
 */
public record TrainingExample(Vector input, Vector target) {
    @Override
    public String toString() {
        return "TrainingExample(" + input + ", " + target + ')';
    }
}
