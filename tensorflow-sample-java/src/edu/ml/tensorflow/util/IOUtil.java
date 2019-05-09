package edu.ml.tensorflow.util;

import org.apache.commons.io.IOUtils;

import edu.ml.tensorflow.ObjectDetector;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Util class to read image, graphDef and label files.
 */
public final class IOUtil {
    private IOUtil() {}

	public static byte[] readAllBytesOrExit(Path path) {
		try {
			return Files.readAllBytes(path);
		} catch (IOException e) {
			System.err.println("Failed to read [" + path + "]: " + e.getMessage());
			e.printStackTrace();
			System.exit(1);
		}
		return null;
	}
	
    public static byte[] readAllBytesOrExit(final String fileName) {
        try {
            return IOUtils.toByteArray(fileName);
        } catch (IOException | NullPointerException ex) {
            throw new ServiceException("Failed to read [" + fileName + "]!", ex);
        }
    }

    public static List<String> readAllLinesOrExit(final String filename) {
        try {
            File file = new File(filename);
            return Files.readAllLines(file.toPath(), Charset.forName("UTF-8"));
        } catch (IOException ex) {
            throw new ServiceException("Failed to read [" + filename + "]!", ex);
        }
    }

    public static void createDirIfNotExists(final File directory) {
        if (!directory.exists()) {
            directory.mkdir();
        }
    }

    public static String getFileName(final String path) {
        return path.substring(path.lastIndexOf("/") + 1, path.length());
    }
}
