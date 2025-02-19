#include <stdint.h>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

using std::cout;
using std::endl;

void brisk_analisis(Mat img_object_face, CascadeClassifier face_cascade, VideoCapture video_data) {
    // Parametros para BRISK
    int thresh = 30;
    int octaves = 3;
    float patternScale = 1.0f;
    Ptr<BRISK> detector_BRISK = BRISK::create(thresh, octaves, patternScale);

    // Detect y compute los descriptores
    Mat descriptors_object_BRISK;
    std::vector<KeyPoint> keypoints_object_BRISK;
    detector_BRISK->detectAndCompute(img_object_face, noArray(), keypoints_object_BRISK, descriptors_object_BRISK);

    // Brute-Force matcher
    BFMatcher matcher(NORM_HAMMING);

    Mat frame, descriptors_scene_BRISK;
    std::vector<KeyPoint> keypoints_scene_BRISK;

    // Bucle para cada frame del video
    while (video_data.read(frame)) {
        // Detectar rostros en cada frame
        std::vector<Rect> faces;
        face_cascade.detectMultiScale(frame, faces);

        // Dibujar rectangulos alrededor de los rostros detectados
        for (const Rect& face : faces) {
            rectangle(frame, face, Scalar(0, 255, 0), 2);
        }

        // Detect keypoints y compute descriptores usando BRISK
        detector_BRISK->detectAndCompute(frame, noArray(), keypoints_scene_BRISK, descriptors_scene_BRISK);

        // Match descriptores
        std::vector<std::vector<DMatch>> matches;
        matcher.knnMatch(descriptors_object_BRISK, descriptors_scene_BRISK, matches, 2);

        // Filtrar good matches
        std::vector<DMatch> good_matches;
        const float ratio_thresh = 0.75f;
        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
                good_matches.push_back(matches[i][0]);
            }
        }

        // Dibujar matches
        Mat img_matches;
        drawMatches(img_object_face, keypoints_object_BRISK, frame, keypoints_scene_BRISK, good_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Mostrar matches
        imshow("Analisis BRISK-BRISK", img_matches);

        // Verificar tecla presionada
        if (waitKey(30) == 13) {
            break;
        }
    }
}

void brisk_freak_analisis(Mat img_object_face, CascadeClassifier face_cascade, VideoCapture video_data) {
    // Parametros para BRISK
    int brisk_thresh = 30;
    int brisk_octaves = 3;
    float brisk_patternScale = 1.0f;
    Ptr<BRISK> detector_BRISK = BRISK::create(brisk_thresh, brisk_octaves, brisk_patternScale);

    // Paráametros para FREAK
    Ptr<FREAK> extractor_FREAK = FREAK::create();

    // Detect y compute los descriptores usando BRISK y FREAK
    Mat descriptors_object_BRISK, descriptors_object_FREAK;
    std::vector<KeyPoint> keypoints_object_BRISK, keypoints_object_FREAK;
    detector_BRISK->detectAndCompute(img_object_face, noArray(), keypoints_object_BRISK, descriptors_object_BRISK);
    extractor_FREAK->compute(img_object_face, keypoints_object_BRISK, descriptors_object_FREAK);

    // Brute-Force matcher con distancia Hamming
    BFMatcher matcher_BRISK(NORM_HAMMING);

    // Brute-Force matcher con distancia L2
    BFMatcher matcher_FREAK(NORM_L2);

    Mat frame;
    std::vector<KeyPoint> keypoints_scene_BRISK;

    // Bucle para cada frame del video
    while (video_data.read(frame)) {
        // Detectar rostros en cada frame
        std::vector<Rect> faces;
        face_cascade.detectMultiScale(frame, faces);

        // Dibujar rectangulos alrededor de los rostros detectados
        for (const Rect& face : faces) {
            rectangle(frame, face, Scalar(0, 255, 0), 2);
        }

        // Detecta keypoints usando BRISK
        detector_BRISK->detect(frame, keypoints_scene_BRISK);
        Mat descriptors_scene_FREAK;
        extractor_FREAK->compute(frame, keypoints_scene_BRISK, descriptors_scene_FREAK);

        // Match descriptores
        std::vector<std::vector<DMatch>> matches;
        if (!descriptors_object_FREAK.empty()) {
            matcher_FREAK.knnMatch(descriptors_object_FREAK, descriptors_scene_FREAK, matches, 2);
        } else {
            matcher_BRISK.knnMatch(descriptors_object_BRISK, descriptors_scene_FREAK, matches, 2);
        }

        // Filtra good matches
        std::vector<DMatch> good_matches;
        const float ratio_thresh = 0.75f;
        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
                good_matches.push_back(matches[i][0]);
            }
        }

        // Dibuja matches
        Mat img_matches;
        drawMatches(img_object_face, keypoints_object_BRISK, frame, keypoints_scene_BRISK, good_matches, img_matches, Scalar::all(-1),
                    Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

        // Mostrar matches
        imshow("Analisis BRISK-FREAK", img_matches);

        // Verificar tecla presionada
        if (waitKey(30) == 13) {
            break;
        }
    }
}

void face_and_eyes_analisis(Mat img_test_face, CascadeClassifier face_cascade, CascadeClassifier eye_cascade, VideoCapture video_data) {
    Mat frame;
    while (video_data.read(frame)) {
        // Converitir cada frame a escala de grises
        Mat frame_gray;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        // Detectar rostros en el frame
        std::vector<Rect> faces;
        face_cascade.detectMultiScale(frame_gray, faces);

        // Dibujar circulos alrededor de rostros y ojos
        for (const Rect& face : faces) {
            Mat resized_test_face;
            resize(img_test_face, resized_test_face, Size(face.width, face.height));

            // Calcular el centro del rostro detectada
            Point center_face(face.x + face.width / 2, face.y + face.height / 2);

            // Dibujar el circulo alrededor del rostro detectado
            int face_radius = std::min(face.width, face.height) / 2;
            circle(frame, center_face, face_radius, Scalar(0, 0, 255), 2);

            // Detectar los ojos en la region del rostro
            Mat face_roi = frame_gray(face);
            std::vector<Rect> eyes;
            eye_cascade.detectMultiScale(face_roi, eyes);

            // Dibujar los circulos alrededor de los ojos detectados
            for (const Rect& eye : eyes) {
                // Calcular el centro del ojo detectado
                Point center_eye(face.x + eye.x + eye.width / 2, face.y + eye.y + eye.height / 2);

                // Dibujar el circulo alrededor del ojo detectado
                int eye_radius = std::min(eye.width, eye.height) / 2;
                circle(frame, center_eye, eye_radius, Scalar(255, 0, 0), 2);
            }
        }

        // Mostrar matches
        imshow("Deteccion de rostro y ojos", frame);

        // Verificar tecla presionada
        if (waitKey(30) == 13) {
            break;
        }
    }
}

int main() {
    // Importar la imagen de referencia
    Mat img_object = imread("../Data/book.png", IMREAD_COLOR);
    if (img_object.empty()) {
        std::cerr << "Error: Unable to load object image." << std::endl;
        return -1;
    }

    // Cargar el face cascade classifier
    CascadeClassifier face_cascade;
    if (!face_cascade.load("../Data/haarcascades/haarcascade_frontalface_alt.xml")) {
        std::cerr << "Error: Unable to load face cascade classifier." << std::endl;
        return -1;
    }

    // Cargar el eye cascade classifier
        CascadeClassifier eye_cascade;
        if (!eye_cascade.load("../Data/haarcascades/haarcascade_eye.xml")) {
            std::cerr << "Error: Unable to load eye cascade classifier." << std::endl;
            return -1;
        }

    // Detectar rostros en la imagen de referencia
    std::vector<Rect> faces;
    face_cascade.detectMultiScale(img_object, faces);

    // Asegurar que un rostro fue detectado
    if (faces.empty()) {
        std::cerr << "Error: No face detected in the object image." << std::endl;
        return -1;
    }

    // Extraer las regiones del rostro detectado
    Rect face_roi = faces[0];
    Mat img_object_face = img_object(face_roi);

    // VideoCapture object para obtener el video
    VideoCapture video_data("../Data/blais.mp4");
    if (!video_data.isOpened()) {
        std::cerr << "Error: Unable to open video file." << std::endl;
    }

    // Metodos para visualizar
    face_and_eyes_analisis(img_object_face, face_cascade, eye_cascade, video_data);
    brisk_analisis(img_object_face, face_cascade, video_data);
    brisk_freak_analisis(img_object_face, face_cascade, video_data);
}
