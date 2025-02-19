#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui_c.h"

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

int main ( )
{
   // ============================== Definición de Matrices ==============================================
    Mat img_1, img_2, img_3, img_4, img_5, img_6;

   // ============================== Lectura de Imagenes ==============================================
    img_1 = imread("../Data/barcelona.png" , IMREAD_ANYCOLOR );
    img_2 = imread("../Data/corral.png" , IMREAD_ANYCOLOR );
    img_3 = imread("../Data/elmo.png" , IMREAD_ANYCOLOR );
    img_4 = imread("../Data/lena1.jpg" , IMREAD_ANYCOLOR );
    img_5 = imread("../Data/meap.png" , IMREAD_ANYCOLOR );
    img_6 = imread("../Data/messi.png" , IMREAD_ANYCOLOR );

    // ============================== Lectura de Imagenes ==============================================

    resize(img_1, img_1, Size(225, 225), INTER_LINEAR);
    resize(img_2, img_2, Size(225, 225), INTER_LINEAR);
    resize(img_3, img_3, Size(225, 225), INTER_LINEAR);
    resize(img_4, img_4, Size(225, 225), INTER_LINEAR);
    resize(img_5, img_5, Size(225, 225), INTER_LINEAR);
    resize(img_6, img_6, Size(225, 225), INTER_LINEAR);

    // ============================== Ejercicio 1 ==============================
    // Crear un programa en c++ que presente 6 imágenes en una misma ventana

    // Crear una matriz grande para contener las 6 imágenes
    int height = img_1.rows + img_2.rows;
    int width = img_1.cols + img_2.cols + img_3.cols;
    Mat combinedImage(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo

    // Copiar las imágenes individuales en la matriz grande
    img_1.copyTo(combinedImage(Rect(0, 0, img_1.cols, img_1.rows)));
    img_2.copyTo(combinedImage(Rect(img_1.cols, 0, img_2.cols, img_2.rows)));
    img_3.copyTo(combinedImage(Rect(img_1.cols + img_2.cols, 0, img_3.cols, img_3.rows)));
    img_4.copyTo(combinedImage(Rect(0, img_1.rows, img_4.cols, img_4.rows)));
    img_5.copyTo(combinedImage(Rect(img_4.cols, img_2.rows, img_5.cols, img_5.rows)));
    img_6.copyTo(combinedImage(Rect(img_4.cols + img_5.cols, img_3.rows, img_6.cols, img_6.rows)));

    // Mostrar la imagen combinada en una ventana
    namedWindow("Display image", WINDOW_NORMAL);
    resizeWindow("Display image", 800, 600);
    imshow("Display image", combinedImage);

    // ============================== Ejercicio 2 ==============================
    // ============================== Ejercicio 2.a ==============================
    // Crear un programa en c++ que a partir de la imagen de entrada (a) de como salida la imagen (b) en una región de 100 x 100 pixeles. SIN USAR OTRA IMAGEN COMO MASCARA

    // Obtener el número de filas y columnas de la imagen

    // Crear una imagen de salida con el mismo tamaño que la imagen de entrada
    Mat img_1_clone = img_1.clone();

    // Definir las coordenadas de la región de interés (ROI)
    int x_aux = (img_1.rows - 100) / 2; // Centrar horizontalmente
    int y_aux = (img_1.cols - 100) / 2; // Centrar verticalmente

    // Verificar que las coordenadas de la ROI estén dentro de los límites de la imagen
    if (x_aux < 0 || y_aux < 0 || x_aux + 100 > img_1.rows || y_aux + 100 > img_1.cols) {
        cout << "La región de interés está fuera de los límites de la imagen." << endl;
        return -1;
    }

    // Iterar sobre la región de interés y cambiar el color de los píxeles
    for (int i = y_aux; i < y_aux + 100; i++) {
        for (int j = x_aux; j < x_aux + 100; j++) {
            // Obtener el valor promedio de intensidad de los canales de color
            Vec3b color = img_1_clone.at<Vec3b>(i, j);
            uchar gray = (color[0] + color[1] + color[2]) / 3; // Promedio de intensidad
            // Asignar el valor promedio a cada canal de color en la imagen de salida
            img_1_clone.at<Vec3b>(i, j) = Vec3b(gray, gray, gray);
        }
    }
    namedWindow("Imagen en escala de grises en region", WINDOW_NORMAL);
    resizeWindow("Imagen en escala de grises en region", 800, 600);
    imshow("Imagen en escala de grises en region", img_1_clone);

    // ============================== Ejercicio 2.b ==============================
    // Una vez obtenida la imagen (b) hacer un programa c++ que binarice la región en escala de gris con valor de umbral de 127. Sin modificar el resto de la imagen.

    Mat img_1_clone_2 = img_1_clone.clone();
    // Iterar sobre la región de interés y binarizar los píxeles en escala de grises
    for (int i = y_aux; i < y_aux + 100; i++) {
        for (int j = x_aux; j < x_aux + 100; j++) {
            // Obtener el valor de intensidad en escala de grises
            uchar gray = img_1_clone_2.at<Vec3b>(i, j)[0]; // Suponemos escala de grises
            // Binarizar el píxel usando un umbral de 127
            uchar binarizedValue = (gray > 127) ? 255 : 0;
            // Asignar el valor binarizado a cada canal de color en la imagen de salida
            img_1_clone_2.at<Vec3b>(i, j) = Vec3b(binarizedValue, binarizedValue, binarizedValue);
        }
    }
    namedWindow("Imagen region binarizada", WINDOW_NORMAL);
    resizeWindow("Imagen region binarizada", 800, 600);
    imshow("Imagen region binarizada", img_1_clone_2);

    // ============================== Ejercicio 3 ==============================
    // ============================== Ejercicio 3.a ==============================
    // Hacer un programa en c++ que a partir de las imágenes (a) y (b) detecte el triceratopos como muestra imagen(c).

    Mat img_A, img_B, img_C;
    img_A = imread("../Data/imA.bmp" , IMREAD_ANYCOLOR );
    img_B = imread("../Data/imB.png" , IMREAD_ANYCOLOR );

    /*Mat grayImageA, grayImageB;
    cvtColor(img_A, grayImageA, COLOR_BGR2GRAY);
    cvtColor(img_B, grayImageB, COLOR_BGR2GRAY);
    // Calcular la diferencia entre las dos imágenes en escala de grises
    Mat differenceImage;
    absdiff(grayImageA, grayImageB, differenceImage);

    imshow("Imagen del objeto faltante", differenceImage);*/

    //Mat differenceImage;
    //absdiff(img_A, img_B, differenceImage);
    absdiff(img_A, img_B, img_C);

    // Aplicar un umbral para resaltar las diferencias
    //double thresholdValue = 100; // Umbral de diferencia (puedes ajustarlo según sea necesario)
    //Mat thresholdedImage;
    //threshold(differenceImage, img_C, thresholdValue, 150, THRESH_BINARY);

    // Crear una matriz grande para contener las 6 imágenes
    height = img_A.rows;
    width = img_A.cols + img_B.cols + img_C.cols;
    Mat combinedImage1(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo
    // Copiar las imágenes individuales en la matriz grande
    img_A.copyTo(combinedImage1(Rect(0, 0, img_A.cols, img_A.rows)));
    img_B.copyTo(combinedImage1(Rect(img_A.cols, 0, img_B.cols, img_B.rows)));
    img_C.copyTo(combinedImage1(Rect(img_A.cols + img_B.cols, 0, img_C.cols, img_C.rows)));

    // Mostrar la imagen combinada en una ventana
    namedWindow("Deteccion triceratops", WINDOW_NORMAL);
    resizeWindow("Deteccion triceratops", 1200, 630);
    imshow("Deteccion triceratops", combinedImage1);

    // ============================== Ejercicio 3.b ==============================
    // Una vez obtenida la imagen(c) hacer un programa en c++ que muestre la posición (x,y) del centro de masa del triceratopos dentro de la imagen. Usando el método de caja englobante o boundingbox

    Mat img_C_clone;
    cvtColor(img_C, img_C_clone, COLOR_BGR2GRAY);

    // Umbralizar la imagen de diferencia (puedes ajustar el umbral según tus necesidades)
    Mat umbral;
    threshold(img_C_clone, umbral, 30, 255, THRESH_BINARY);

    // Encontrar contornos en la imagen umbralizada
    std::vector<std::vector<Point>> contornos;
    findContours(umbral, contornos, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Encontrar el contorno más grande (suponiendo que sea el objeto faltante)
    double maxArea = 0;
    int maxIdx = -1;
    for (size_t i = 0; i < contornos.size(); ++i) {
        double area = contourArea(contornos[i]);
        if (area > maxArea) {
            maxArea = area;
            maxIdx = static_cast<int>(i);
        }
    }
    // Calcular el centro de masa del contorno más grande
    if (maxIdx >= 0) {
        Moments momentos = moments(contornos[maxIdx]);
        double centroX = momentos.m10 / momentos.m00;
        double centroY = momentos.m01 / momentos.m00;
        cout << "Centro de masa del objeto faltante: (" << centroX << ", " << centroY << ")" << endl;
        // Dibujar un rectángulo alrededor del objeto faltante (caja englobante)
        Rect caja = boundingRect(contornos[maxIdx]);
        rectangle(img_C_clone, caja, Scalar(255, 0, 0), 2);

        // Mostrar la imagen con el rectángulo dibujado
        namedWindow("Imagen de diferencia con caja englobante", WINDOW_NORMAL);
        resizeWindow("Imagen de diferencia con caja englobante", 800, 600);
        imshow("Imagen de diferencia con caja englobante", img_C_clone);
    } else {
        cout << "No se encontraron objetos faltantes." << endl;
    }

    // ============================== Ejercicio 4 ==============================
    // ============================== Ejercicio 4.a ==============================
    // Hacer un programa en c++ que a partir de las imágenes (a) y (b) genere un nueva imagen (c) donde solo aparezca el triceratopos en sus colores de gris originales.

    // Convertir las imágenes a escala de grises
    Mat grayImageA, grayImageB;
    cvtColor(img_A, grayImageA, COLOR_BGR2GRAY);
    cvtColor(img_B, grayImageB, COLOR_BGR2GRAY);
    // Calcular la diferencia entre las dos imágenes en escala de grises
    Mat differenceImage1;
    absdiff(grayImageA, grayImageB, differenceImage1);
    differenceImage1 =  255 - differenceImage1;

    // Crear una matriz grande para contener las 6 imágenes
    height = img_A.rows;
    width = img_A.cols + img_B.cols + img_C.cols;
    Mat combinedImage2(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo
    // Copiar las imágenes individuales en la matriz grande
    img_A.copyTo(combinedImage2(Rect(0, 0, img_A.cols, img_A.rows)));
    img_B.copyTo(combinedImage2(Rect(img_A.cols, 0, img_B.cols, img_B.rows)));
    Mat differenceImage1Color;
    cvtColor(differenceImage1, differenceImage1Color, COLOR_GRAY2BGR);
    differenceImage1Color.copyTo(combinedImage2(Rect(img_A.cols + img_B.cols, 0, differenceImage1.cols, differenceImage1.rows)));
    // Mostrar la imagen combinada en una ventana
    namedWindow("Deteccion triceratops colores originales", WINDOW_NORMAL);
    resizeWindow("Deteccion triceratops colores originales", 1200, 630);
    imshow("Deteccion triceratops colores originales", combinedImage2);

    // ============================== Ejercicio 4.b ==============================
    // Una vez obtenida la imagen(c) hacer un programa en c++ que muestre la posición (x,y) del centro de masa del triceratopos dentro de la imagen. Usando el método de caja englobante o boundingbox en sus colores de gris originales

    Mat img_C2_clone;
    cvtColor(differenceImage1Color, img_C2_clone, COLOR_BGR2GRAY);
    // Umbralizar la imagen de diferencia (puedes ajustar el umbral según tus necesidades)
    Mat umbral2;
    adaptiveThreshold(img_C2_clone, umbral2, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 2);
    // Encontrar contornos en la imagen umbralizada
    std::vector<std::vector<Point>> contornos2;
    findContours(umbral2, contornos2, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // Encontrar el contorno más grande (suponiendo que sea el objeto faltante)
    double maxArea2 = 0;
    int maxIdx2 = -1;
    for (size_t i = 0; i < contornos2.size(); ++i) {
        double area = contourArea(contornos2[i]);
        if (area > maxArea2) {
            maxArea2 = area;
            maxIdx2 = static_cast<int>(i);
        }
    }
    // Calcular el centro de masa del contorno más grande
    if (maxIdx2 >= 0) {
        Moments momentos = moments(contornos2[maxIdx2]);
        double centroX = momentos.m10 / momentos.m00;
        double centroY = momentos.m01 / momentos.m00;
        cout << "Centro de masa del objeto faltante: (" << centroX << ", " << centroY << ")" << endl;
        // Dibujar un rectángulo alrededor del objeto faltante (caja englobante)
        Rect caja2 = boundingRect(contornos2[maxIdx2]);
        rectangle(img_C2_clone, caja2, Scalar(0, 0, 255), 2);

        // Mostrar la imagen con el rectángulo dibujado
        namedWindow("Imagen con color original con caja englobante", WINDOW_NORMAL);
        resizeWindow("Imagen con color original con caja englobante", 800, 600);
        imshow("Imagen con color original con caja englobante", img_C2_clone);
    } else {
        cout << "No se encontraron objetos faltantes." << endl;
    }

    waitKey(0);
    return EXIT_SUCCESS;
}
