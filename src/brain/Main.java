package brain;

import brain.domain.Brain;
import brain.math.ActivationFunction;
import brain.math.Vector;
import brain.misc.LayerDefinition;
import brain.misc.TrainingExample;
import processing.core.PApplet;

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
public class Main extends PApplet {

    private static final boolean WRITE_TO_FILE = false;

    private final Brain brain = new Brain(
            new LayerDefinition(8, ActivationFunction.LINEAR),
            new LayerDefinition(32, ActivationFunction.RELU),
            new LayerDefinition(128, ActivationFunction.RELU),
            new LayerDefinition(32, ActivationFunction.RELU),
            new LayerDefinition(3, ActivationFunction.SIGMOID)
    );

    private final String imageName = "itdogodown";

    private final String format = "png";

    private final String path = STR."../\{imageName}.\{format}";

    private File file;

    private BufferedImage image;

    private BufferedImage imageOut;

    private TrainingExample[] trainingExamples;

    public static void main(String[] args) {
        PApplet.main(Main.class, args);
    }

    private static float[] fourierSeries2(float x, float y) {
        x *= TWO_PI;
        y *= TWO_PI;

        return new float[]{
                sin(x),
                cos(x),
                sin(y),
                cos(y),
                sin(x * 2),
                cos(x * 2),
                sin(y * 2),
                cos(y * 2)
        };
    }

    @Override
    public void settings() {
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

        URL url = Main.class.getResource(path);
        file = new File(url.getFile().replace("%20", " "));

        try {
            image = ImageIO.read(new FileImageInputStream(file));
            imageOut = new BufferedImage(width, height, image.getType());
            size(image.getWidth(), image.getHeight());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    // TODO: domain from and to bytes (single file, instead of an entire folder)
    public void setup() {
        trainingExamples = new TrainingExample[width * height];
        brain.setLearningRate(1e-3f);
        brain.setMiniBatchSize(32);
//            brain.setMiniBatchSize(Integer.MAX_VALUE);
//        int hiddenNeurons = trainingExamples.length / (2 * (2 + 1));
//        int neuronsPerLayer = hiddenNeurons / 4;
//        Brain domain = new Brain(2, neuronsPerLayer, neuronsPerLayer, neuronsPerLayer, 1);

        for (int j = 0; j < height; j++) {
            float y = (float) j / height;

            for (int i = 0; i < width; i++) {
                float x = (float) i / width;
                int rgb = image.getRGB(i, j);
                float r = (rgb >> 16 & 0xFF) / 255.0f;
                float g = (rgb >> 8 & 0xFF) / 255.0f;
                float b = (rgb & 0xFF) / 255.0f;
                //trainingExamples[j * w + i] = new TrainingExample(Vector.of(x, y), Vector.of(b));
                trainingExamples[j * width + i] = new TrainingExample(
                        Vector.of(fourierSeries2(x, y)),
                        Vector.of(r, g, b)
                );
            }
        }

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
    }

    @Override
    public void draw() {
        brain.train(trainingExamples);
        System.out.println(STR."Finished epoch \{frameCount}");
        predictImage();

        if (WRITE_TO_FILE) {
            try {
                var pathname = STR."\{file.getParent()}\\output\{System.currentTimeMillis()}.\{format}";
                ImageIO.write(imageOut, format, new File(pathname));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private void predictImage() {
        loadPixels();

        for (int j = 0; j < height; j++) {
            float y = (float) j / height;

            for (int i = 0; i < width; i++) {
                float x = (float) i / width;
                Vector output = brain.predict(fourierSeries2(x, y)).mult(255.0f);
                int r = (int) output.get(0);
                int g = (int) output.get(1);
                int b = (int) output.get(2);
                pixels[j * width + i] = color(r, g, b);
            }
        }

        updatePixels();
    }

}
