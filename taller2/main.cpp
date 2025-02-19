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
    img_1 = imread("../Data/ave.jpg" , IMREAD_ANYCOLOR );
    img_2 = imread("../Data/ave2.png" , IMREAD_ANYCOLOR );
    img_3 = imread("../Data/imagenBGR.png" , IMREAD_ANYCOLOR );
    img_4 = imread("../Data/lenanoise.png" , IMREAD_ANYCOLOR );
    img_5 = imread("../Data/mes.jpg" , IMREAD_ANYCOLOR );
    img_6 = imread("../Data/tutor.png" , IMREAD_ANYCOLOR );

    // ============================== Ejercicio 1 ==============================
    // Dilatacion imagen 6

    // Crear un elemento estructurante para la operación de dilatación
    Mat element = getStructuringElement(MORPH_RECT, Size(9, 9));
    // Aplicar la operación de dilatación a la imagen
    Mat dilated_img_6;
    dilate(img_6, dilated_img_6, element);

    // Crear una matriz grande para contener las 6 imágenes
    int height = img_6.rows;
    int width = img_6.cols + img_6.cols;
    Mat combinedImage(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo
    // Copiar las imágenes individuales en la matriz grande
    img_6.copyTo(combinedImage(Rect(0, 0, img_6.cols, img_6.rows)));
    dilated_img_6.copyTo(combinedImage(Rect(img_6.cols, 0, dilated_img_6.cols, dilated_img_6.rows)));

    // Mostrar la imagen combinada en una ventana
    namedWindow("Imagen 6 Dilatada", WINDOW_NORMAL);
    resizeWindow("Imagen 6 Dilatada", 1200, 500);
    imshow("Imagen 6 Dilatada", combinedImage);

    // ============================== Ejercicio 2 ==============================
    // Erosion imagen 6

    // Crear un elemento estructurante para la operación de erosión
    element = getStructuringElement(MORPH_RECT, Size(5, 5));
    // Aplicar la operación de erosión a la imagen
    Mat eroded_img_6;
    erode(img_6, eroded_img_6, element);

    // Crear una matriz grande para contener las 6 imágenes
    height = img_6.rows;
    width = img_6.cols + img_6.cols;
    Mat combinedImage2(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo
    // Copiar las imágenes individuales en la matriz grande
    img_6.copyTo(combinedImage2(Rect(0, 0, img_6.cols, img_6.rows)));
    eroded_img_6.copyTo(combinedImage2(Rect(img_6.cols, 0, eroded_img_6.cols, eroded_img_6.rows)));

    // Mostrar la imagen combinada en una ventana
    namedWindow("Imagen 6 Erosionada", WINDOW_NORMAL);
    resizeWindow("Imagen 6 Erosionada", 1200, 500);
    imshow("Imagen 6 Erosionada", combinedImage2);

    // ============================== Ejercicio 3 ==============================
    // Erosion imagen 2 - O tambien se puede erosionar y luego dilatar imagen 2
    //Primero se Erosiona                           //solo se pueden valores impares
    element = getStructuringElement(MORPH_RECT, Size(7, 7));
    Mat eroded_img_2;
    //img original, img de destino
    erode(img_2, eroded_img_2, element);

    //Segundo se dilata
    element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat dilated_img_2;
    dilate(eroded_img_2, dilated_img_2, element);

    //Tercero se muestra el resultado
    // una matriz que compare la imagen original con la imagen de salida

    height = img_2.rows;
    width = img_2.cols + img_2.cols;
    Mat combinedImage3(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo
    // Copiar las imágenes individuales en la matriz grande
    img_2.copyTo(combinedImage3(Rect(0, 0, img_2.cols, img_2.rows)));
    dilated_img_2.copyTo(combinedImage3(Rect(img_2.cols, 0, dilated_img_2.cols, dilated_img_2.rows)));

    // Mostrar la imagen combinada en una ventana
    namedWindow("Imagen 2 Erosionada y Dilatada", WINDOW_NORMAL);
    resizeWindow("Imagen 2 Erosionada y Dilatada", 1200, 500);
    imshow("Imagen 2 Erosionada y Dilatada", combinedImage3);

    // ============================== Ejercicio 4 ==============================
    // Dilatacion imagen 1 - O tambien se puede dilatar y luego erosionar imagen 1

    //Primero se dilata
    element = getStructuringElement(MORPH_RECT, Size(7, 7));
    Mat dilated_img_1;
    dilate(img_1, dilated_img_1, element);

    //Segundo se Erosiona                           //solo se pueden valores impares
    element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat eroded_img_1;
    //img original, img de destino
    erode(dilated_img_1, eroded_img_1, element);

    //Tercero se muestra el resultado
    // una matriz que compare la imagen original con la imagen de salida
    height = img_1.rows;
    width = img_1.cols + img_1.cols;
    Mat combinedImage4(height, width, CV_8UC3, Scalar(0, 0, 0)); // Matriz negra como fondo
    // Copiar las imágenes individuales en la matriz grande
    img_1.copyTo(combinedImage4(Rect(0, 0, img_1.cols, img_1.rows)));
    eroded_img_1.copyTo(combinedImage4(Rect(img_1.cols, 0, dilated_img_1.cols, dilated_img_1.rows)));

    // Mostrar la imagen combinada en una ventana
    namedWindow("Imagen 1 Dilatada y Erosionada", WINDOW_NORMAL);
    resizeWindow("Imagen 1 Dilatada y Erosionada", 1200, 500);
    imshow("Imagen 1 Dilatada y Erosionada", combinedImage4);

    // ============================== Ejercicio 5 ==============================
    // Separacion de caracteristicas imagen 3

    Mat img_3_clone = img_3.clone();

    // Dividir la imagen en sus canales de color (BGR)
    std::vector<Mat> canales(3);
    split(img_3_clone, canales);

    // Crear imágenes solo con un canal de color activo
    Mat img_3_red = Mat::zeros(img_3_clone.size(), CV_8UC3);
    img_3_red.setTo(Scalar(0, 0, 255), canales[2]); // Rojo

    Mat img_3_green = Mat::zeros(img_3_clone.size(), CV_8UC3);
    img_3_green.setTo(Scalar(0, 255, 0), canales[1]); // Verde

    Mat img_3_blue = Mat::zeros(img_3_clone.size(), CV_8UC3);
    img_3_blue.setTo(Scalar(255, 0, 0), canales[0]); // Azul

    // Mostrar las imágenes resultantes en ventanas separadas
    namedWindow("Imagen 3 Rojo", WINDOW_NORMAL);
    resizeWindow("Imagen 3 Rojo", 800, 600);
    imshow("Imagen 3 Rojo", img_3_red);

    namedWindow("Imagen 3 Verde", WINDOW_NORMAL);
    resizeWindow("Imagen 3 Verde", 800, 600);
    imshow("Imagen 3 Verde", img_3_green);

    namedWindow("Imagen 3 Azul", WINDOW_NORMAL);
    resizeWindow("Imagen 3 Azul", 800, 600);
    imshow("Imagen 3 Azul", img_3_blue);

    // ============================== Ejercicio 6 ==============================
    // Reduccion de ruido imagen 4
    // 1. clonar la imagen original

    Mat img_4_clone = img_4.clone();
    Mat imagen_filtro_mediana;
    medianBlur(img_4_clone, imagen_filtro_mediana, 5);

    namedWindow("Imagen 4 Reducción de ruido", WINDOW_NORMAL);
    resizeWindow("Imagen 4 Reducción de ruido", 800, 600);
    imshow("Imagen 4 Reducción de ruido", imagen_filtro_mediana);


    // ============================== Ejercicio 7 ==============================
    // Separacion de canales RGB imagen 5

    Mat img_5_clone = img_5.clone();

    int h_blue = 110, r_blue = 40; // Valores para los colores azules
    int h_red = 180, r_red = 170; // Valores para los colores rojos
    int h_green = 60, r_green = 40; // Valores para los colores verdes

    // Convertir la imagen a espacio de color HSV
    Mat hsv;
    cvtColor(img_5_clone, hsv, COLOR_BGR2HSV);

    // Crear las matrices de destino para cada canal de color
    Mat dst_blue, dst_red, dst_green;
    dst_blue.create(img_5_clone.rows, img_5_clone.cols, img_5_clone.type());
    dst_red.create(img_5_clone.rows, img_5_clone.cols, img_5_clone.type());
    dst_green.create(img_5_clone.rows, img_5_clone.cols, img_5_clone.type());

    // Variables para el tamaño de la imagen
    int rows = img_5_clone.rows;
    int cols = img_5_clone.cols;

    // Rangos de colores
    uchar h1_blue = (h_blue - (r_blue / 2) + 360) % 360;
    uchar h2_blue = (h_blue + (r_blue / 2) + 360) % 360;

    uchar h1_red = (h_red - (r_red / 2) + 360) % 360;
    uchar h2_red = (h_red + (r_red / 2) + 360) % 360;

    uchar h1_green = (h_green - (r_green / 2) + 360) % 360;
    uchar h2_green = (h_green + (r_green / 2) + 360) % 360;

    for (int i = 0; i < rows; i++) {
     uchar* ptr_src = img_5_clone.ptr<uchar>(i);
     uchar* ptr_hsv = hsv.ptr<uchar>(i);
     uchar* ptr_dst_blue = dst_blue.ptr<uchar>(i);
     uchar* ptr_dst_red = dst_red.ptr<uchar>(i);
     uchar* ptr_dst_green = dst_green.ptr<uchar>(i);

     for (int j = 0; j < cols; j++) {
         uchar H = ptr_hsv[j * 3]; // Valor de hue (H)

         // Azul
         bool in_range_blue = false;
         if (h1_blue <= h2_blue) {
             if (H >= h1_blue && H <= h2_blue)
                 in_range_blue = true;
         } else if (H >= h1_blue || H <= h2_blue)
             in_range_blue = true;

         // Rojo
         bool in_range_red = false;
         if (h1_red <= h2_red) {
             if (H >= h1_red && H <= h2_red)
                 in_range_red = true;
         } else {
             if (H >= h1_red || H <= h2_red)
                 in_range_red = true;
         }

         // Verde
         bool in_range_green = false;
         if (h1_green <= h2_green) {
             if (H >= h1_green && H <= h2_green)
                 in_range_green = true;
         } else if (H >= h1_green || H <= h2_green)
             in_range_green = true;

         if (in_range_blue) {
             ptr_dst_blue[j * 3 + 0] = ptr_src[j * 3 + 0];
             ptr_dst_blue[j * 3 + 1] = ptr_src[j * 3 + 1];
             ptr_dst_blue[j * 3 + 2] = ptr_src[j * 3 + 2];
         } else {
             uchar gray = (ptr_src[j * 3] + ptr_src[j * 3 + 1] + ptr_src[j * 3 + 2]) / 3;
             ptr_dst_blue[j * 3 + 2] = ptr_dst_blue[j * 3 + 1] = ptr_dst_blue[j * 3] = gray;
         }

         if (in_range_red && !in_range_blue) { // Añadir !in_range_blue para excluir el azul
             ptr_dst_red[j * 3 + 0] = ptr_src[j * 3 + 0];
             ptr_dst_red[j * 3 + 1] = ptr_src[j * 3 + 1];
             ptr_dst_red[j * 3 + 2] = ptr_src[j * 3 + 2];
         } else {
             uchar gray = (ptr_src[j * 3] + ptr_src[j * 3 + 1] + ptr_src[j * 3 + 2]) / 3;
             ptr_dst_red[j * 3 + 2] = ptr_dst_red[j * 3 + 1] = ptr_dst_red[j * 3] = gray;
         }

         if (in_range_green) {
             ptr_dst_green[j * 3 + 0] = ptr_src[j * 3 + 0];
             ptr_dst_green[j * 3 + 1] = ptr_src[j * 3 + 1];
             ptr_dst_green[j * 3 + 2] = ptr_src[j * 3 + 2];
         } else {
             uchar gray = (ptr_src[j * 3] + ptr_src[j * 3 + 1] + ptr_src[j * 3 + 2]) / 3;
             ptr_dst_green[j * 3 + 2] = ptr_dst_green[j * 3 + 1] = ptr_dst_green[j * 3] = gray;
         }
     }
    }

    // Mostrar las imágenes resultantes en ventanas separadas
    namedWindow("Imagen 5 Blue", WINDOW_NORMAL);
    resizeWindow("Imagen 5 Blue", 800, 600);
    imshow("Imagen 5 Blue", dst_blue);

    namedWindow("Imagen 5 Red", WINDOW_NORMAL);
    resizeWindow("Imagen 5 Red", 800, 600);
    imshow("Imagen 5 Red", dst_red);

    namedWindow("Imagen 5 Green", WINDOW_NORMAL);
    resizeWindow("Imagen 5 Green", 800, 600);
    imshow("Imagen 5 Green", dst_green);



    waitKey(0);
    return EXIT_SUCCESS;
}
