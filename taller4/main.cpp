#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

//Cargar el clasificador en cascada de Haar para caras y el calsificador para ojos
String face_cascade_name ="../Data/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name ="../Data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;

Mat detectAndDysplay(Mat frame) {
    Mat faceROI;
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    std::vector<Rect> faces;
    Rect roi;
    face_cascade.detectMultiScale( frame_gray, faces );
    for ( size_t i = 0; i < faces.size(); i++ ){
        roi.x = faces[i].x-5;
        roi.y = faces[i].y;
        roi.width = faces[i].width -30;

        roi.height = faces[i].height+15;
        faceROI = frame( roi );
    }
    return faceROI;
}

int main() {
    // -------------------- Punto 1 --------------------
    // Realice un programa en C++ usando opencv que segmente por color la imagen de imagen de entrada (a) y de obtenga como salida la imagen (b).

    // Lectura de la imagen
    cv::Mat img_1 = cv::imread("../Data/entrada.png");

    // Convertir la imagen RGB a HSV
    Mat img_1_hsv;
    cvtColor(img_1, img_1_hsv, COLOR_BGR2HSV);

    // Rangos de azules en HSV
    Scalar blue_down(78,40,20);
    Scalar blue_up(125,255,255);

    // Mascara para extraer al pajaro
    Mat mascara_pajaro;
    inRange(img_1_hsv, blue_down, blue_up, mascara_pajaro);

    // Definicion de la silueta del pajaro
    Mat img_1_final;
    bitwise_and(img_1, img_1, img_1_final, mascara_pajaro = mascara_pajaro);

    // Crear una matriz grande para contener las imágenes
    int height = img_1.rows;
    int width = img_1.cols + img_1.cols;
    Mat combinedImage(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo
    // Copiar las imágenes individuales en la matriz grande
    img_1.copyTo(combinedImage(Rect(0, 0, img_1.cols, img_1.rows)));
    img_1_final.copyTo(combinedImage(Rect(img_1.cols, 0, img_1_final.cols, img_1_final.rows)));

    // Mostrar la imagen combinada en una ventana
    namedWindow("Extraccion del pajaro", WINDOW_NORMAL);
    resizeWindow("Extraccion del pajaro", 1200, 500);
    imshow("Extraccion del pajaro", combinedImage);

    // -------------------- Punto 2 --------------------
    // Realice un programa en C++ usando opencv que diga la cantidad de dinero en coronas suecas que hay en la foto (b).

    // Lectura de la imagen
    Mat img_2 = imread("../Data/koruny_black1.jpg");

    // Escalar la imagen
    double scale = 0.5;
    resize(img_2, img_2, Size(), scale, scale);

    Mat img_2_clone = img_2.clone();
    // Filtro de mediana
    cv::medianBlur(img_2_clone, img_2_clone, 5);
    // Escala de grises
    cv::cvtColor(img_2_clone, img_2_clone, cv::COLOR_BGR2GRAY);

    // Detectar círculos con la transformada de Hough
    std::vector<cv::Vec3f> circles;
    HoughCircles(img_2_clone, circles, cv::HOUGH_GRADIENT, 1, 180, 30, 30, 50, 100);
    cvtColor(img_2_clone,img_2_clone, COLOR_GRAY2BGR);

    int suma_dinero = 0;

    // Dibujar los círculos detectados
    for (size_t i = 0; i <= circles.size(); i++) {
        cv::Vec3i c = circles[i];
        cv::Point center = cv::Point(c[0], c[1]);

        // Dibujar el círculo
        cv::circle(img_2_clone, center, c[2], cv::Scalar(0,0,0), 5);
        // Dibujar el centro
        cv::circle(img_2_clone, center, 2, cv::Scalar(0,255,0), 5);
        // Extraccion del radio de cada circulo para determinar su valor
        float radio = circles[i][2];

        // Conteo de cada peso
        if (71 < radio && radio < 75) {
            suma_dinero += 1;
            }
        else if (75 < radio && radio < 83) {
            suma_dinero += 2;
            }
        else if (83 < radio && radio < 90) {
            suma_dinero += 5;
            }
        else if (90 < radio && radio < 95) {
            suma_dinero += 10;
            }
        else if (95 < radio && radio < 100) {
            suma_dinero += 20;
            }
    }

    // Crear una matriz grande para contener las imágenes
    height = img_2.rows;
    width = img_2.cols + img_2.cols;
    Mat combinedImage_2(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo
    // Copiar las imágenes individuales en la matriz grande
    img_2.copyTo(combinedImage_2(Rect(0, 0, img_2.cols, img_2.rows)));
    img_2_clone.copyTo(combinedImage_2(Rect(img_2_clone.cols, 0, img_2_clone.cols, img_2_clone.rows)));

    // Mostrar la imagen combinada en una ventana
    namedWindow("Conteo coronas suecas", WINDOW_NORMAL);
    resizeWindow("Conteo coronas suecas", 1200, 500);
    imshow("Conteo coronas suecas", combinedImage_2);
    cout << "Total de coronas suecas: $" << suma_dinero << "." << endl;

    // -------------------- Punto 3 --------------------
    // Realice un programa en C++ usando opencv que segmente la imagen de de imagen de entrada (a) y de como salida la imagen (b) y diga cuántos sparkies hay en la foto.

    // Lectura de la imagen
    Mat img_3 = imread("../Data/smarties.png");
    Mat img_3_clone = img_3.clone();

    // Filtro de mediana
    cv::medianBlur(img_3_clone, img_3_clone, 5);

    // Escala de grises
    Mat img_3_gray;
    cv::cvtColor(img_3_clone, img_3_gray, cv::COLOR_BGR2GRAY);

    // Detectar círculos con la transformada de Hough
    std::vector<cv::Vec3f> sparkies;
    cv::HoughCircles(img_3_gray, sparkies, cv::HOUGH_GRADIENT, 1, 48, 50, 30, 0, 120);

    // Dibujar los círculos detectados
    for (size_t i = 0; i <= sparkies.size(); i++) {
        cv::Vec3i c = sparkies[i];
        cv::Point center_sparkies = cv::Point(c[0], c[1]);
        // Dibujar el círculo
        cv::circle(img_3_clone, center_sparkies, c[2], cv::Scalar(255,0,255), 5);
        // Dibujar el centro
        cv::circle(img_3_clone, center_sparkies, 2, cv::Scalar(255,255,255), 5);
    }

    // Crear una matriz grande para contener las imágenes
    height = img_3.rows;
    width = img_3.cols + img_3.cols;
    Mat combinedImage_3(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo
    // Copiar las imágenes individuales en la matriz grande
    img_3.copyTo(combinedImage_3(Rect(0, 0, img_3.cols, img_3.rows)));
    img_3_clone.copyTo(combinedImage_3(Rect(img_3_clone.cols, 0, img_3_clone.cols, img_3_clone.rows)));

    // Mostrar la imagen combinada en una ventana
    namedWindow("Conteo sparkies", WINDOW_NORMAL);
    resizeWindow("Conteo sparkies", 1200, 500);
    imshow("Conteo sparkies", combinedImage_3);
    cout << "Hay un total de " << sparkies.size() << " sparkies." << endl;

    // -------------------- Punto 4 --------------------
    // Realice un programa en C++ usando opencv que segmente la imagen de imagen de entrada (a) y de como salida la imagen (b) , ©, (d) y en una imagen final cambien el color de la piel a color fucsia.

    // Lectura de la imagen
    Mat img_4 = imread("../Data/mano-in.png");
    Mat img_4_clone_1 = img_4.clone();

    // Extraer hsv
    cvtColor(img_4_clone_1, img_4_clone_1, COLOR_BGR2HSV);
    // Extraer mascara de un color (Rosado color piel)
    inRange(img_4_clone_1, Scalar(0, 1, 60), Scalar(20, 255, 255), img_4_clone_1);
    Mat img_4_clone_2 = img_4_clone_1.clone();

    Mat  kernel;
    kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(img_4_clone_2, img_4_clone_2, MORPH_CLOSE, kernel);
    // Crear un elemento estructurante para la operación de dilatación
    Mat element = getStructuringElement(MORPH_RECT, Size(7, 7));
    // Aplicar la operación de dilatación a la imagen
    dilate(img_4_clone_2, img_4_clone_2, element);
    cvtColor(img_4_clone_2, img_4_clone_2,COLOR_GRAY2BGR);

    // Mascara Rosada BGR
    Mat img_4_clone_3(img_4_clone_2.rows, img_4_clone_2.cols, CV_8UC3, Scalar(255, 0, 255));
    multiply(img_4_clone_2, img_4_clone_3, img_4_clone_3, 1.0, -1);

    // Crear una matriz grande para contener las imágenes
    height = img_4.rows;
    width = img_4.cols + img_4.cols + img_4.cols + img_4.cols;
    Mat combinedImage_4(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo
    // Copiar las imágenes individuales en la matriz grande
    img_4.copyTo(combinedImage_4(Rect(0, 0, img_4.cols, img_4.rows)));
    cvtColor(img_4_clone_1, img_4_clone_1, COLOR_GRAY2BGR);
    img_4_clone_1.copyTo(combinedImage_4(Rect(img_4.cols, 0, img_4_clone_1.cols, img_4_clone_1.rows)));
    img_4_clone_2.copyTo(combinedImage_4(Rect(img_4.cols + img_4_clone_1.cols, 0, img_4_clone_2.cols, img_4_clone_2.rows)));
    img_4_clone_3.copyTo(combinedImage_4(Rect(img_4.cols + img_4_clone_1.cols + img_4_clone_2.cols, 0, img_4_clone_3.cols, img_4_clone_3.rows)));

    // Mostrar la imagen combinada en una ventana
    namedWindow("Extraccion mano", WINDOW_NORMAL);
    resizeWindow("Extraccion mano", 2400, 500);
    imshow("Extraccion mano", combinedImage_4);

    // -------------------- Punto 5 --------------------
    // Realice un programa en C++ usando opencv que segmente la imagen de lena y la convierta en morena.

    // Verificar cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "Error loading face cascade\n";
        return -1;
    };

    // Lectura de la imagen
    Mat img_5 = cv::imread("../Data/lena.png");
    Mat img_5_original = img_5.clone();

    // Deteccion de rostros
    Mat img_5_rostro = detectAndDysplay(img_5);
    Mat img_5_rostro_clone = img_5_rostro.clone();

    // Convertir a hsv
    Mat roi_hsv;
    cvtColor(img_5_rostro, roi_hsv, COLOR_BGR2HSV);
    Mat img_5_detect, kernel_lena;
    inRange(roi_hsv, Scalar(0, 20, 70), Scalar(50, 255, 255), img_5_detect);
    kernel_lena = getStructuringElement(MORPH_RECT, Size(5,5));
    morphologyEx(img_5_detect, img_5_detect, MORPH_DILATE, kernel_lena);
    cvtColor(img_5_detect, img_5_detect,COLOR_GRAY2BGR);

    for (int i=0; i < img_5_detect.rows; i++){
        for(int j=0; j < img_5_detect.cols; j++){
            Vec3b pixel = img_5_detect.at<Vec3b>(i, j);
            uchar B=pixel[0];
            uchar G=pixel[1];
            uchar R=pixel[2];
            if(B>0){
                img_5_rostro.at<Vec3b>(i, j)[0]=87;
            }
            if(G>0){
                img_5_rostro.at<Vec3b>(i, j)[1]=122;
            }
            if(R>0){
                img_5_rostro.at<Vec3b>(i, j)[2]=185;
            }
        }
    }

    // Crear una matriz grande para contener las imágenes
    height = img_5_original.rows;
    width = img_5_original.cols + img_5_detect.cols + img_5.cols;
    Mat combinedImage_5(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo
    // Copiar las imágenes individuales en la matriz grande
    img_5_original.copyTo(combinedImage_5(Rect(0, 0, img_5_original.cols, img_5_original.rows)));
    img_5_rostro_clone.copyTo(combinedImage_5(Rect(img_5_original.cols, 100, img_5_rostro_clone.cols, img_5_rostro_clone.rows)));
    img_5_detect.copyTo(combinedImage_5(Rect(img_5_original.cols, img_5_rostro_clone.rows+100, img_5_detect.cols, img_5_detect.rows)));
    img_5.copyTo(combinedImage_5(Rect(img_5_original.cols + img_5_detect.cols, 0, img_5.cols, img_5.rows)));

    // Mostrar la imagen combinada en una ventana
    namedWindow("Lena morena", WINDOW_NORMAL);
    resizeWindow("Lena morena", 1800, 500);
    imshow("Lena morena", combinedImage_5);

    waitKey(0);
    return EXIT_SUCCESS;
}
