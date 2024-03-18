package brain.misc;

import brain.math.ActivationFunction;

/**
 * @author Emilio Zottel (5AHIF)
 * @since 18.03.2024, Mo.
 */
public record LayerDefinition(int size,
                              ActivationFunction activationFunction) {

}
