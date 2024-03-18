package brain.misc;

import java.util.Arrays;

import static brain.domain.Brain.random;

public class MiniBatch {
    private final TrainingExample[] trainingExamples;

    public MiniBatch(TrainingExample[] trainingExamples) {
        this.trainingExamples = trainingExamples;
    }

    /**
     * Fischer-Yates Shuffle
     *
     * @param array array to be shuffled (will not be modified)
     * @param <T>   datatype of the array's contents
     * @return shuffled array
     */
    public static <T> T[] shuffle(T[] array) {
        T[] arr = array.clone();

        for (int i = arr.length - 1; i >= 1; i--) {
            int idx = random.nextInt(i + 1);
            T temp = arr[idx];
            arr[idx] = arr[i];
            arr[i] = temp;
        }

        return arr;
    }

    public static MiniBatch[] shuffleAndChop(int miniBatchSize, TrainingExample[] trainingTrainingExamples) {
        int len = (int) Math.ceil((float) trainingTrainingExamples.length / miniBatchSize);
        MiniBatch[] miniBatches = new MiniBatch[len];
        trainingTrainingExamples = shuffle(trainingTrainingExamples);

        for (int i = 0; i < len; i++) {
            int from = i * miniBatchSize;
            int to = Math.min(from + miniBatchSize, trainingTrainingExamples.length);
            TrainingExample[] examples = Arrays.copyOfRange(trainingTrainingExamples, from, to);
            miniBatches[i] = new MiniBatch(examples);
        }

        return miniBatches;
    }

    public int size() {
        return trainingExamples.length;
    }

    public TrainingExample getExample(int i) {
        return trainingExamples[i];
    }

    @Override
    public String toString() {
        return Arrays.toString(trainingExamples);
    }
}
