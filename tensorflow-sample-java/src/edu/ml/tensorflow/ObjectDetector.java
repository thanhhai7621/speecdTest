package edu.ml.tensorflow;

import static edu.ml.tensorflow.Config.GRAPH_FILE;
import static edu.ml.tensorflow.Config.LABEL_FILE;
import static edu.ml.tensorflow.Config.MEAN;
import static edu.ml.tensorflow.Config.SIZE;

import java.nio.FloatBuffer;
import java.nio.file.Paths;
import java.util.List;

import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import edu.ml.tensorflow.classifier.YOLOClassifier;
import edu.ml.tensorflow.model.Recognition;
import edu.ml.tensorflow.util.GraphBuilder;
import edu.ml.tensorflow.util.IOUtil;
import edu.ml.tensorflow.util.ImageUtil;
import edu.ml.tensorflow.util.ServiceException;

/**
 * ObjectDetector class to detect objects using pre-trained models with
 * TensorFlow Java API.
 */
public class ObjectDetector {
	private byte[] GRAPH_DEF;
	private List<String> LABELS;

	public ObjectDetector() {
		try {
			GRAPH_DEF = IOUtil.readAllBytesOrExit(Paths.get(GRAPH_FILE));
			LABELS = IOUtil.readAllLinesOrExit(LABEL_FILE);
		} catch (ServiceException ex) {
		}
	}

	/**
	 * Detect objects on the given image
	 * 
	 * @param imageLocation
	 *            the location of the image
	 */
	public void detect(final String imageLocation) {
		// byte[] image = IOUtil.readAllBytesOrExit(imageLocation);
		byte[] image = IOUtil.readAllBytesOrExit(Paths.get(imageLocation));
		try (Tensor<Float> normalizedImage = normalizeImage(image)) {
			List<Recognition> recognitions = YOLOClassifier.getInstance()
					.classifyImage(executeYOLOGraph(normalizedImage), LABELS);
			printToConsole(recognitions);
			ImageUtil.getInstance().labelImage(image, recognitions, IOUtil.getFileName(imageLocation));
		}
	}

	/**
	 * Pre-process input. It resize the image and normalize its pixels
	 * 
	 * @param imageBytes
	 *            Input image
	 * @return Tensor<Float> with shape [1][416][416][3]
	 */
	private Tensor<Float> normalizeImage(final byte[] imageBytes) {
		try (Graph graph = new Graph()) {
			GraphBuilder graphBuilder = new GraphBuilder(graph);

			final Output<Float> output = graphBuilder.div( // Divide each pixels with the MEAN
					graphBuilder.resizeBilinear( // Resize using bilinear interpolation
							graphBuilder.expandDims( // Increase the output tensors dimension
									graphBuilder.cast( // Cast the output to Float
											graphBuilder.decodeJpeg(graphBuilder.constant("input", imageBytes), 3),
											Float.class),
									graphBuilder.constant("make_batch", 0)),
							graphBuilder.constant("size", new int[] { SIZE, SIZE })),
					graphBuilder.constant("scale", MEAN));

			try (Session session = new Session(graph)) {
				return session.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
			}
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}
	}

	/**
	 * Executes graph on the given preprocessed image
	 * 
	 * @param image
	 *            preprocessed image
	 * @return output tensor returned by tensorFlow
	 */
	private float[] executeYOLOGraph(final Tensor<Float> image) {
		try (Graph graph = new Graph()) {
			graph.importGraphDef(GRAPH_DEF);
			try (Session s = new Session(graph);
					Tensor<Float> result = s.runner().feed("input", image).fetch("output").run().get(0)
							.expect(Float.class)) {
				float[] outputTensor = new float[YOLOClassifier.getInstance().getOutputSizeByShape(result)];
				FloatBuffer floatBuffer = FloatBuffer.wrap(outputTensor);
				result.writeTo(floatBuffer);
				return outputTensor;
			}
		}
	}

	/**
	 * Prints out the recognize objects and its confidence
	 * 
	 * @param recognitions
	 *            list of recognitions
	 */
	private void printToConsole(final List<Recognition> recognitions) {
		for (Recognition recognition : recognitions) {
			System.out.println("Object: " + recognition.getTitle() + " - confidence: " + recognition.getConfidence());
		}
	}
}
