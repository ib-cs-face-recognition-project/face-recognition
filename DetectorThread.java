/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package gui;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import javax.imageio.ImageIO;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

/**
 *
 * @author aydnc
 */
public class DetectorThread extends Thread {

    boolean running;
    FaceDetection jFrame;
    int count;
    VideoCapture webSource;
    Mat frame;
    Mat grayFrame;
    MatOfByte mem;
    CascadeClassifier detector;
    MatOfRect faceDetections;
    Point current;
    Point previous;
    int idleCount;
    boolean done;
    Rect crop;
    BufferedImage face;

    public BufferedImage getFace() {
        this.start();
        return face;
    }

    @Override
    public void run() {
        while (running) {
            if (webSource.grab()) {
                try {
                    webSource.retrieve(frame);
                    grayFrame = new Mat();
                    Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
                    Imgproc.equalizeHist(grayFrame, grayFrame);
                    Graphics g = jFrame.getPanel().getGraphics();

                    detector.detectMultiScale(grayFrame, faceDetections, 1.1, 2, 1000, new Size(150, 150));
                    for (Rect rect : faceDetections.toArray()) {

                        Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                                new Scalar(0, 255, 0));
                        if (faceDetections.toArray().length == 1) {
                            if (current.x > 0 && current.y > 0) {
                                previous.x = current.x;
                                previous.y = current.y;
                            }
                            current.x = rect.x;
                            current.y = rect.y;
                            if (previous != null && Math.abs(current.x - previous.x) < 50 && Math.abs(current.y - previous.y) < 25) {
                                idleCount++;
                                crop.x += (int) current.x;
                                crop.y += (int) current.y;
                                crop.width += (int) rect.width;
                                crop.height += (int) rect.height;
                            } else {
                                idleCount = 0;
                                crop = new Rect();
                            }
                        }
                    }
                    done = false;
                    if (faceDetections.empty() || faceDetections.equals(null)) {
                        jFrame.getLabel().setText("No Face Detected!");
                    } else if (faceDetections.toArray().length > 1) {
                        jFrame.getLabel().setText("Multiple Faces Detected!");
                    } else if (idleCount < 15) {
                        jFrame.getLabel().setText("Hold Still Please!");
                    } else {
                        jFrame.getLabel().setText("Done!");
                        done = true;
                    }
                    jFrame.getLabel().setVisible(true);

                    if (done) {
                        crop.x /= idleCount;
                        crop.y /= idleCount;
                        crop.width /= idleCount;
                        crop.height /= (idleCount * 2);
                        Imgcodecs.imencode(".bmp", frame.submat(crop), mem);
                    } else {
                        Imgcodecs.imencode(".bmp", frame, mem);
                    }
                    Image im = ImageIO.read(new ByteArrayInputStream(mem.toArray()));
                    BufferedImage buff = (BufferedImage) im;
                    if (done) {
                        face = buff;
                        return;
                    } else {
                        g.drawImage(buff, 0, 0, jFrame.getWidth(), jFrame.getHeight() - 150, 0, 0, buff.getWidth(), buff.getHeight(), null);
                    }

                } catch (Exception ex) {
                    System.out.println("Error!!");
                    ex.printStackTrace();
                }
            }
        }
    }

    public void initialize(FaceDetection panel) {
        this.jFrame = panel;
        current = new Point(-1000, -1000);
        previous = new Point(-2000, -2000);
        crop = new Rect();
        detector = new CascadeClassifier(FaceDetection.class.getResource("haarcascade_frontalface_alt2.xml").getPath().substring(1));
        count = 0;
        webSource = new VideoCapture(0);
        frame = new Mat();
        mem = new MatOfByte();
        faceDetections = new MatOfRect();
    }
}
