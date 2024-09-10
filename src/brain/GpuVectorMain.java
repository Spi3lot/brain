package brain;

import brain.domain.Brain;
import brain.math.ActivationFunction;
import brain.math.CpuMatrix;
import brain.math.GpuVector;
import brain.misc.LayerDefinition;
import brain.misc.TrainingExample;
import processing.core.PApplet;

import javax.imageio.ImageIO;
import javax.imageio.stream.FileImageInputStream;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * @author : Emilio Zottel (5AHIF)
 * @since : 13.11.2023, Mo.
 **/
public class GpuVectorMain extends PApplet {

    private static final boolean WRITE_TO_FILE = false;

    private static final int FOURIER_ORDER = 8;

    private final Brain brain = new Brain(
            GpuVector::new,
            CpuMatrix::new,
            new LayerDefinition(FOURIER_ORDER * 4, ActivationFunction.LINEAR),
            new LayerDefinition(32, ActivationFunction.LRELU),
            new LayerDefinition(128, ActivationFunction.LRELU),
            new LayerDefinition(32, ActivationFunction.LRELU),
            new LayerDefinition(3, ActivationFunction.SIGMOID)
    );

    private final String imageName = "pencil";

    private final String format = "png";

    private final String path = STR."../\{imageName}.\{format}";

    private File file;

    private BufferedImage image;

    private BufferedImage imageOut;

    private TrainingExample[] trainingExamples;

    public static void main(String[] args) {
        PApplet.main(GpuVectorMain.class, args);
    }

    private static float[] fourierSeries(int order, float x, float y) {
        float[] series = new float[order * 4];
        x *= TWO_PI;
        y *= TWO_PI;

        for (int i = 0; i < order; i++) {
            series[i * 4] = sin(x * (i + 1));
            series[i * 4 + 1] = cos(x * (i + 1));
            series[i * 4 + 2] = sin(y * (i + 1));
            series[i * 4 + 3] = cos(y * (i + 1));
        }

        return series;
    }

    @Override
    public void settings() {
        var url = GpuVectorMain.class.getResource(path);
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
        brain.setLearningRate(1e-2f);
        brain.setMiniBatchSize(32);

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
                        GpuVector.of(fourierSeries(FOURIER_ORDER, x, y)),
                        GpuVector.of(r, g, b)
                );
            }
        }
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
                var output = brain.predict(GpuVector.of(fourierSeries(FOURIER_ORDER, x, y))).mult(255.0f);
                int r = (int) output.get(0);
                int g = (int) output.get(1);
                int b = (int) output.get(2);
                pixels[j * width + i] = color(r, g, b);
            }
        }

        updatePixels();
    }

}
