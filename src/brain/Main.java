package brain;

import brain.domain.Brain;
import brain.math.ActivationFunction;
import brain.math.Vector;
import brain.misc.TrainingExample;

import javax.imageio.ImageIO;
import javax.imageio.stream.FileImageInputStream;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;

/**
 * @author : Emilio Zottel (5AHIF)
 * @since : 13.11.2023, Mo.
 **/
public class Main {

    public static void main(String[] args) {
        assert false;
        System.out.println(new Brain(5));
//        test();
//        var brain = new Brain(1024, 64, 64, 2);
    }

    public static native void test();

    // TODO: domain from and to bytes (single file, instead of an entire folder)
    private static void image() {
        System.out.println("Starting...");

        /*
        Brain domain = new Brain(28 * 28, 16, 10);
        writeToFolder("resources/Test");

        Brain readTest = Brain.fromFolder("resources/Test");
        */
        /*
        float x = 1.0f;
        float dx = 0.001f;
        float a = (ActivationFunctions.ELU(x + dx) - ActivationFunctions.ELU(x)) / dx;
        float b = ActivationFunctions.sigmoid(x);
        System.out.println(a);
        System.out.println(b);
        System.out.println(Math.abs(a - b));
         */

        /*
        Brain domain = new Brain(28 * 28, 16, 10);
        Vector input = new Vector(28 * 28);
        input.fillWithRandomValues(-1, 1);

        Vector output = domain.predict(input);
        System.out.println(output.argmax());
        System.out.println(output);
        */
        /* */
        String format = "png";
        String path = "pencil." + format;
        URL url = Brain.class.getResource("../" + path);
        assert url != null;
        File file = new File(url.getFile().replace("%20", " "));

        try {
            BufferedImage image = ImageIO.read(new FileImageInputStream(file));
            int w = image.getWidth();
            int h = image.getHeight();
            BufferedImage imageOut = new BufferedImage(w, h, image.getType());

            TrainingExample[] trainingExamples = new TrainingExample[w * h];
            //int hiddenNeurons = trainingExamples.length / (2 * (2 + 1));
            //int neuronsPerLayer = hiddenNeurons / 4;
            //Brain domain = new Brain(2, neuronsPerLayer, neuronsPerLayer, neuronsPerLayer, 1);
            Brain brain = new Brain(2, 256, 256, 128, 3);
            brain.setActivationFunction(ActivationFunction.SIGMOID);
            brain.setLearningRate(1.0f);
            brain.setMiniBatchSize(32);
            //domain.setMiniBatchSize(Integer.MAX_VALUE);

            for (int j = 0; j < h; j++) {
                float y = (float) j / h;

                for (int i = 0; i < w; i++) {
                    float x = (float) i / w;

                    int rgb = image.getRGB(i, j);
                    float r = (rgb >> 16 & 0xFF) / 255.0f;
                    float g = (rgb >> 8 & 0xFF) / 255.0f;
                    float b = (rgb & 0xFF) / 255.0f;

                    //trainingExamples[j * w + i] = new TrainingExample(Vector.of(x, y), Vector.of(b));
                    trainingExamples[j * w + i] = new TrainingExample(Vector.of(x, y), Vector.of(r, g, b));
                }
            }

            for (int i = 0; i < 10; i++) {
                brain.train(trainingExamples);
            }

            for (int j = 0; j < h; j++) {
                float y = (float) j / h;

                for (int i = 0; i < w; i++) {
                    float x = (float) i / w;

                    Vector output = brain.predict(x, y).mult(255.0f);
                    int r = (int) output.get(0);
                    int g = (int) output.get(1);
                    int b = (int) output.get(2);

                    int rgb = 0xFF000000 | r << 16 | g << 8 | b;
                    imageOut.setRGB(i, j, rgb);
                }
            }

            ImageIO.write(imageOut, format, new File(file.getParent() + "\\output." + format));

        } catch (IOException e) {
            e.printStackTrace();
        }
        /* */

        /* * /
        Brain domain = new Brain(2, 2, 1);
        domain.setLearningRate(3.0f);
        System.out.println(domain.predict(0, 0).get(0));
        System.out.println(domain.predict(0, 1).get(0));
        System.out.println(domain.predict(1, 0).get(0));
        System.out.println(domain.predict(1, 1).get(0));

        TrainingExample[] trainingExamples = new TrainingExample[4];

        for (int i = 0; i < trainingExamples.length; i++) {
            int a = i & 1;
            int b = i >>> 1;
            trainingExamples[i] = new TrainingExample(Vector.of(a, b), Vector.of(a ^ b));
        }

        for (int i = 0; i < 1000; i++) {
            domain.train(trainingExamples);
        }

        System.out.println();
        System.out.println(domain.predict(0, 0).get(0));
        System.out.println(domain.predict(0, 1).get(0));
        System.out.println(domain.predict(1, 0).get(0));
        System.out.println(domain.predict(1, 1).get(0));
        /* */
        System.out.println("Done");
    }

}
