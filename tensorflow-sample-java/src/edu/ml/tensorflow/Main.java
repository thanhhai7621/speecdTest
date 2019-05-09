package edu.ml.tensorflow;

public class Main {
    private final static String IMAGE = "C:/Users/haidt/Desktop/Yolo/eagle.jpg";

    public static void main(String[] args) {
        ObjectDetector objectDetector = new ObjectDetector();
        objectDetector.detect(IMAGE);
    }
}
