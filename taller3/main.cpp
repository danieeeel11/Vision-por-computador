#include <stdint.h>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <math.h>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/opencv.hpp"
/*#include "opencv2/core.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/tracking.hpp"
#include "opencv2/videoio.hpp"*/

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

using std::cout;
using std::endl;


void sort_matches_increasing(std::vector< cv::DMatch >& matches)
{
    for (int i = 0; i < matches.size(); i++)
    {
        for (int j = 0; j < matches.size() - 1; j++)
        {
            if (matches[j].distance > matches[j + 1].distance)
            {
                auto temp = matches[j];
                matches[j] = matches[j + 1];
                matches[j + 1] = temp;
            }
        }
    }
}

std::vector<DMatch> extract_good_matches(int opc, Mat descriptors_object, Mat descriptors_scene){
    std::vector<DMatch> good_matches;
    if (opc == 1){
        // Flann
        Ptr<DescriptorMatcher> matcherFlann = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);

        // Combinar descriptores
        std::vector<std::vector<DMatch>> matches;
        matcherFlann->knnMatch(descriptors_object, descriptors_scene, matches, 2);

        // Filtrar correspondencias usando el test de razón de distancia de Lowe
        const float ratio_thresh = 0.7f;

        for (size_t i = 0; i < matches.size(); i++) {
            if (matches[i][0].distance < ratio_thresh * matches[i][1].distance) {
                good_matches.push_back(matches[i][0]);
            }
        }
    } else if (opc == 2){
        // Brute Force Matcher
        BFMatcher matcherBFM(NORM_L2);
        std::vector<DMatch> matches;
        matcherBFM.match(descriptors_object, descriptors_scene, matches);

        good_matches = matches;

        // Sort them in order of their distance. The less distance, the better.
        sort_matches_increasing(good_matches);

        if (good_matches.size() > 20)
        {
            good_matches.resize(20);
        }
    }

    return good_matches;
}

Mat process(Mat img_object, Mat img_scene, vector<KeyPoint> keypoints_object, vector<KeyPoint> keypoints_scene, Mat descriptors_object, Mat descriptors_scene, int opc){

    std::vector<DMatch> good_matches = extract_good_matches(opc, descriptors_object, descriptors_scene);

    std::cout << "  -> Cantidad de matches: " << good_matches.size() << std::endl;

    // Dibujar matches
    Mat img_matches;
    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Localizar el objeto
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for (size_t i = 0; i < good_matches.size(); i++) {
        // Obtener los keypoints de los good matches
        obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }

    // Encontrar homografia
    Mat H = findHomography(obj, scene, RANSAC);

    // Definir las esquinas del objeto
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f((float)img_object.cols, 0);
    obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
    obj_corners[3] = Point2f(0, (float)img_object.rows);

    // Transformar las esquinas del objeto a coordenadas de escena
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform(obj_corners, scene_corners, H);

    // Dibujar las lineas entre las esquinas del objeto
    line(img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
         scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
         scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
         scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);
    line(img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
         scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4);

    return img_matches;
}

int main() {
    // https://github.com/oreillymedia/Learning-OpenCV-3_examples/blob/master/example_16-02.cpp

    Mat img_object = imread("../Data/box.png", IMREAD_GRAYSCALE);
    Mat img_scene = imread("../Data/box_in_scene.png", IMREAD_GRAYSCALE);
    if (img_object.empty() || img_scene.empty()) {
        cout << "Could not open or find the image!\n" << endl;
        return -1;
    }

    // ----------------------------------- Creacion de los detectores -------------------------------

    Ptr<SIFT> sift_detector = SIFT::create();
    Ptr<SURF> surf_detector = SURF::create();

    // Parametros para BRISK
    int thresh = 30;
    int octaves = 3;
    float patternScale = 1.0f;
    Ptr<BRISK> detector_BRISK = BRISK::create(thresh, octaves, patternScale);
    //Ptr<BRISK> brisk_detector = BRISK::create();

    // Parametros para ORB
    int nfeatures2 = 1000;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold2 = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    int patchSize = 31;
    int fastThreshold = 20;
    ORB::ScoreType scoreType = ORB::HARRIS_SCORE;
    Ptr<ORB> detector_orb = ORB::create(nfeatures2,scaleFactor,nlevels,edgeThreshold2,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);

    Ptr<FREAK> detector_freak = FREAK::create();

    Ptr<FastFeatureDetector> detector_fast = FastFeatureDetector::create();

    Ptr<BriefDescriptorExtractor> detector_brief = BriefDescriptorExtractor::create();

    // ----------------------------------- Creacion de keypoints --------------------------------

    // Keypoints SIFT
    vector<KeyPoint> keypoints_object_SIFT, keypoints_scene_SIFT;
    sift_detector->detect(img_object, keypoints_object_SIFT);
    sift_detector->detect(img_scene, keypoints_scene_SIFT);

    // Keypoints SURF
    vector<KeyPoint> keypoints_object_SURF, keypoints_scene_SURF;
    surf_detector->detect(img_object, keypoints_object_SURF);
    surf_detector->detect(img_scene, keypoints_scene_SURF);

    // Keypoints BRISK
    vector<KeyPoint> keypoints_object_BRISK, keypoints_scene_BRISK;
    detector_BRISK->detect(img_object, keypoints_object_BRISK);
    detector_BRISK->detect(img_scene, keypoints_scene_BRISK);

    // Keypoints ORB
    vector<KeyPoint> keypoints_object_ORB, keypoints_scene_ORB;
    detector_orb->detect(img_object, keypoints_object_ORB);
    detector_orb->detect(img_scene, keypoints_scene_ORB);

    // Keypoints FAST
    vector<KeyPoint> keypoints_object_FAST, keypoints_scene_FAST;
    detector_fast->detect(img_object, keypoints_object_FAST);
    detector_fast->detect(img_scene, keypoints_scene_FAST);

    // ----------------------------------- Creacion de descriptores --------------------------------

    // Descriptores SIFT
    Mat descriptors_object_SIFT, descriptors_scene_SIFT;
    sift_detector->compute(img_object, keypoints_object_SIFT, descriptors_object_SIFT);
    sift_detector->compute(img_scene, keypoints_scene_SIFT, descriptors_scene_SIFT);

    // Descriptores SURF
    Mat descriptors_object_SURF, descriptors_scene_SURF;
    surf_detector->compute(img_object, keypoints_object_SURF, descriptors_object_SURF);
    surf_detector->compute(img_scene, keypoints_scene_SURF, descriptors_scene_SURF);

    // Descriptores BRISK
    Mat descriptors_object_BRISK, descriptors_scene_BRISK;
    detector_BRISK->compute(img_object, keypoints_object_BRISK, descriptors_object_BRISK);
    detector_BRISK->compute(img_scene, keypoints_scene_BRISK, descriptors_scene_BRISK);
    descriptors_object_BRISK.convertTo(descriptors_object_BRISK, CV_32F);
    descriptors_scene_BRISK.convertTo(descriptors_scene_BRISK, CV_32F);

    // Descriptores ORB
    Mat descriptors_object_ORB, descriptors_scene_ORB;
    detector_orb->compute(img_object, keypoints_object_ORB, descriptors_object_ORB);
    detector_orb->compute(img_scene, keypoints_scene_ORB, descriptors_scene_ORB);
    descriptors_object_ORB.convertTo(descriptors_object_ORB, CV_32F);
    descriptors_scene_ORB.convertTo(descriptors_scene_ORB, CV_32F);

    // ----------------------------------- COMBINACIONES --------------------------------
    // ------------------------------------ SIFT_SIFT ------------------------------------

    std::cout << "SIFT_SIFT" << std::endl;
    Mat img_matches_SIFT_SIFT = process(img_object, img_scene, keypoints_object_SIFT, keypoints_scene_SIFT, descriptors_object_SIFT, descriptors_scene_SIFT, 1);

    // Mostrar correspondencias
    namedWindow("Good Matches SIFT_SIFT", WINDOW_AUTOSIZE);
    imshow("Good Matches SIFT_SIFT", img_matches_SIFT_SIFT);

    // ------------------------------------ SURF_SURF ------------------------------------

    std::cout << "SURF_SURF" << std::endl;
    Mat img_matches_SURF_SURF = process(img_object, img_scene, keypoints_object_SURF, keypoints_scene_SURF, descriptors_object_SURF, descriptors_scene_SURF, 1);

    // Mostrar correspondencias
    namedWindow("Good Matches SURF_SURF", WINDOW_AUTOSIZE);
    imshow("Good Matches SURF_SURF", img_matches_SURF_SURF);

    // ------------------------------------ BRISK_BRISK ------------------------------------

    std::cout << "BRISK_BRISK" << std::endl;
    Mat img_matches_BRISK_BRISK = process(img_object, img_scene, keypoints_object_BRISK, keypoints_scene_BRISK, descriptors_object_BRISK, descriptors_scene_BRISK, 1);

    // Mostrar correspondencias
    namedWindow("Good Matches BRISK_BRISK", WINDOW_AUTOSIZE);
    imshow("Good Matches BRISK_BRISK", img_matches_BRISK_BRISK);

    // ------------------------------------ ORB_ORB ------------------------------------

    std::cout << "ORB_ORB" << std::endl;
    Mat img_matches_ORB_ORB = process(img_object, img_scene, keypoints_object_ORB, keypoints_scene_ORB, descriptors_object_ORB, descriptors_scene_ORB, 1);

    // Mostrar correspondencias
    namedWindow("Good Matches ORB_ORB", WINDOW_AUTOSIZE);
    imshow("Good Matches ORB_ORB", img_matches_ORB_ORB);

    // ------------------------------------ SIFT_SURF ------------------------------------

    // SIFT_SURF
    Mat descriptors_object_SIFT_SURF, descriptors_scene_SIFT_SURF;
    surf_detector->compute(img_object,keypoints_object_SIFT, descriptors_object_SIFT_SURF );
    surf_detector->compute(img_scene,keypoints_scene_SIFT, descriptors_scene_SIFT_SURF );

    std::cout << "SIFT_SURF" << std::endl;
    Mat img_matches_SIFT_SURF = process(img_object, img_scene, keypoints_object_SIFT, keypoints_scene_SIFT, descriptors_object_SIFT_SURF, descriptors_scene_SIFT_SURF, 1);

    // Mostrar correspondencias
    namedWindow("Good Matches SIFT_SURF", WINDOW_AUTOSIZE);
    imshow("Good Matches SIFT_SURF", img_matches_SIFT_SURF);

    // ------------------------------------ SURF_SIFT ------------------------------------

    // SURF_SIFT
    Mat descriptors_object_SURF_SIFT, descriptors_scene_SURF_SIFT;
    sift_detector->compute(img_object,keypoints_object_SURF, descriptors_object_SURF_SIFT );
    sift_detector->compute(img_scene,keypoints_scene_SURF, descriptors_scene_SURF_SIFT );

    std::cout << "SURF_SIFT" << std::endl;
    Mat img_matches_SURF_SIFT = process(img_object, img_scene, keypoints_object_SURF, keypoints_scene_SURF, descriptors_object_SURF_SIFT, descriptors_scene_SURF_SIFT, 1);

    // Mostrar correspondencias
    namedWindow("Good Matches SURF_SIFT", WINDOW_AUTOSIZE);
    imshow("Good Matches SURF_SIFT", img_matches_SURF_SIFT);

    // ------------------------------------ BRISK_SURF ------------------------------------

    // BRISK_SURF
    Mat descriptors_object_BRISK_SURF, descriptors_scene_BRISK_SURF;
    surf_detector->compute(img_object,keypoints_object_BRISK, descriptors_object_BRISK_SURF );
    surf_detector->compute(img_scene,keypoints_scene_BRISK, descriptors_scene_BRISK_SURF );

    std::cout << "BRISK_SURF" << std::endl;
    Mat img_matches_BRISK_SURF = process(img_object, img_scene, keypoints_object_BRISK, keypoints_scene_BRISK, descriptors_object_BRISK_SURF, descriptors_scene_BRISK_SURF, 1);

    // Mostrar correspondencias
    namedWindow("Good Matches BRISK_SURF", WINDOW_AUTOSIZE);
    imshow("Good Matches BRISK_SURF", img_matches_BRISK_SURF);

    // ------------------------------------ SURF_BRISK ------------------------------------

    // SURF_BRISK
    Mat descriptors_object_SURF_BRISK, descriptors_scene_SURF_BRISK;
    detector_BRISK->compute(img_object,keypoints_object_SURF, descriptors_object_SURF_BRISK );
    detector_BRISK->compute(img_scene,keypoints_scene_SURF, descriptors_scene_SURF_BRISK );
    descriptors_object_SURF_BRISK.convertTo(descriptors_object_SURF_BRISK, CV_32F);
    descriptors_scene_SURF_BRISK.convertTo(descriptors_scene_SURF_BRISK, CV_32F);

    std::cout << "SURF_BRISK" << std::endl;
    Mat img_matches_SURF_BRISK = process(img_object, img_scene, keypoints_object_SURF, keypoints_scene_SURF, descriptors_object_SURF_BRISK, descriptors_scene_SURF_BRISK, 1);

    // Mostrar correspondencias
    namedWindow("Good Matches SURF_BRISK", WINDOW_AUTOSIZE);
    imshow("Good Matches SURF_BRISK", img_matches_SURF_BRISK);

    // ------------------------------------ BRISK_SIFT ------------------------------------

    // BRISK_SIFT
    Mat descriptors_object_BRISK_SIFT, descriptors_scene_BRISK_SIFT;
    sift_detector->compute(img_object,keypoints_object_BRISK, descriptors_object_BRISK_SIFT );
    sift_detector->compute(img_scene,keypoints_scene_BRISK, descriptors_scene_BRISK_SIFT );

    std::cout << "BRISK_SIFT" << std::endl;
    Mat img_matches_BRISK_SIFT = process(img_object, img_scene, keypoints_object_BRISK, keypoints_scene_BRISK, descriptors_object_BRISK_SIFT, descriptors_scene_BRISK_SIFT, 1);

    // Mostrar correspondencias
    namedWindow("Good Matches BRISK_SIFT", WINDOW_AUTOSIZE);
    imshow("Good Matches BRISK_SIFT", img_matches_BRISK_SIFT);

    // ------------------------------------ SIFT_BRISK ------------------------------------

    Mat descriptors_object_SIFT_BRISK, descriptors_scene_SIFT_BRISK;
    detector_BRISK->compute(img_object,keypoints_object_SIFT, descriptors_object_SIFT_BRISK );
    detector_BRISK->compute(img_scene,keypoints_scene_SIFT, descriptors_scene_SIFT_BRISK );
    descriptors_object_SIFT_BRISK.convertTo(descriptors_object_SIFT_BRISK, CV_32F);
    descriptors_scene_SIFT_BRISK.convertTo(descriptors_scene_SIFT_BRISK, CV_32F);

    std::cout << "SIFT_BRISK" << std::endl;
    Mat img_matches_SIFT_BRISK = process(img_object, img_scene, keypoints_object_SIFT, keypoints_scene_SIFT, descriptors_object_SIFT_BRISK, descriptors_scene_SIFT_BRISK, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches SIFT_BRISK", WINDOW_AUTOSIZE);
    imshow("Good Matches SIFT_BRISK", img_matches_SIFT_BRISK);

    // ------------------------------------ ORB_SURF ------------------------------------

    // ORB_SURF
    Mat descriptors_object_ORB_SURF, descriptors_scene_ORB_SURF;
    surf_detector->compute(img_object,keypoints_object_ORB, descriptors_object_ORB_SURF );
    surf_detector->compute(img_scene,keypoints_scene_ORB, descriptors_scene_ORB_SURF );

    std::cout << "ORB_SURF" << std::endl;
    Mat img_matches_ORB_SURF = process(img_object, img_scene, keypoints_object_ORB, keypoints_scene_ORB, descriptors_object_ORB_SURF, descriptors_scene_ORB_SURF, 1);

    // Mostrar correspondencias
    namedWindow("Good Matches ORB_SURF", WINDOW_AUTOSIZE);
    imshow("Good Matches ORB_SURF", img_matches_ORB_SURF);

    // ------------------------------------ SURF_ORB ------------------------------------

    Mat descriptors_object_SURF_ORB, descriptors_scene_SURF_ORB;
    detector_orb->compute(img_object,keypoints_object_SURF, descriptors_object_SURF_ORB );
    detector_orb->compute(img_scene,keypoints_scene_SURF, descriptors_scene_SURF_ORB );
    descriptors_object_SURF_ORB.convertTo(descriptors_object_SURF_ORB, CV_32F);
    descriptors_scene_SURF_ORB.convertTo(descriptors_scene_SURF_ORB, CV_32F);

    std::cout << "SURF_ORB" << std::endl;
    Mat img_matches_SURF_ORB = process(img_object, img_scene, keypoints_object_SURF, keypoints_scene_SURF, descriptors_object_SURF_ORB, descriptors_scene_SURF_ORB, 1);
    namedWindow("Good Matches SURF_ORB", WINDOW_AUTOSIZE);
    imshow("Good Matches SURF_ORB", img_matches_SURF_ORB);

    // ------------------------------------ ORB_SIFT ------------------------------------
    // https://github.com/santosderek/Brute-Force-Matching-using-ORB-descriptors/blob/master/src/main.cpp

    keypoints_object_ORB.clear();
    keypoints_scene_ORB.clear();
    detector_orb->detect(img_object, keypoints_object_ORB);
    detector_orb->detect(img_scene, keypoints_scene_ORB);

    // ORB_SIFT
    Mat descriptors_object_ORB_SIFT, descriptors_scene_ORB_SIFT;
    sift_detector->compute(img_object,keypoints_object_ORB, descriptors_object_ORB_SIFT );
    sift_detector->compute(img_scene,keypoints_scene_ORB, descriptors_scene_ORB_SIFT );
    descriptors_object_ORB_SIFT.convertTo(descriptors_object_ORB_SIFT, CV_32F);
    descriptors_scene_ORB_SIFT.convertTo(descriptors_scene_ORB_SIFT, CV_32F);

    std::cout << "ORB_SIFT" << std::endl;
    Mat img_matches_ORB_SIFT = process(img_object, img_scene, keypoints_object_ORB, keypoints_scene_ORB, descriptors_object_ORB_SIFT, descriptors_scene_ORB_SIFT, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches ORB_SIFT", WINDOW_AUTOSIZE);
    imshow("Good Matches ORB_SIFT", img_matches_ORB_SIFT);

    // ------------------------------------ SIFT_ORB ------------------------------------
    // Esta combinacion no es posible, al hacer la siguiente combinacion se rebienta el programa: what():  OpenCV(4.3.0-pre) /home/lab/Downloads/opencv/modules/core/src/alloc.cpp:73: error: (-4:Insufficient memory) Failed to allocate 71026336400 bytes in function 'OutOfMemoryError'

    /*Mat descriptors_object_SIFT_ORB, descriptors_scene_SIFT_ORB;
    detector_orb->compute(img_object,keypoints_object_SIFT, descriptors_object_SIFT_ORB );
    detector_orb->compute(img_scene,keypoints_scene_SIFT, descriptors_scene_SIFT_ORB );

    Mat img_matches_SIFT_ORB = process(img_object, img_scene, keypoints_object_SIFT, keypoints_scene_SIFT, descriptors_object_SIFT_ORB, descriptors_scene_SIFT_ORB);

    // Mostrar correspondencias
    namedWindow("Good Matches SIFT_ORB", WINDOW_AUTOSIZE);
    imshow("Good Matches SIFT_ORB", img_matches_SIFT_ORB);*/

    // ------------------------------------ ORB_BRISK ------------------------------------

    Mat descriptors_object_ORB_BRISK, descriptors_scene_ORB_BRISK;
    detector_BRISK->compute(img_object,keypoints_object_ORB, descriptors_object_ORB_BRISK );
    detector_BRISK->compute(img_scene,keypoints_scene_ORB, descriptors_scene_ORB_BRISK );

    std::cout << "ORB_BRISK" << std::endl;
    Mat img_matches_ORB_BRISK = process(img_object, img_scene, keypoints_object_ORB, keypoints_scene_ORB, descriptors_object_ORB_BRISK, descriptors_scene_ORB_BRISK, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches ORB_BRISK", WINDOW_AUTOSIZE);
    imshow("Good Matches ORB_BRISK", img_matches_ORB_BRISK);

    // ------------------------------------ BRISK_ORB ------------------------------------

    Mat descriptors_object_BRISK_ORB, descriptors_scene_BRISK_ORB;
    detector_orb->compute(img_object,keypoints_object_BRISK, descriptors_object_BRISK_ORB );
    detector_orb->compute(img_scene,keypoints_scene_BRISK, descriptors_scene_BRISK_ORB );

    std::cout << "BRISK_ORB" << std::endl;
    Mat img_matches_BRISK_ORB = process(img_object, img_scene, keypoints_object_BRISK, keypoints_scene_BRISK, descriptors_object_BRISK_ORB, descriptors_scene_BRISK_ORB, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches BRISK_ORB", WINDOW_AUTOSIZE);
    imshow("Good Matches BRISK_ORB", img_matches_BRISK_ORB);

    // ------------------------------------ SURF_FREAK ------------------------------------
    // https://github.com/kikohs/freak/blob/master/demo/freak_demo.cpp

    Mat descriptors_object_SURF_FREAK, descriptors_scene_SURF_FREAK;
    detector_freak->compute(img_object, keypoints_object_SURF, descriptors_object_SURF_FREAK);
    detector_freak->compute(img_scene, keypoints_scene_SURF, descriptors_scene_SURF_FREAK);

    std::cout << "SURF_FREAK" << std::endl;
    Mat img_matches_SURF_FREAK = process(img_object, img_scene, keypoints_object_SURF, keypoints_scene_SURF, descriptors_object_SURF_FREAK, descriptors_scene_SURF_FREAK, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches SURF_FREAK", WINDOW_AUTOSIZE);
    imshow("Good Matches SURF_FREAK", img_matches_SURF_FREAK);

    // ------------------------------------ SIFT_FREAK ------------------------------------

    Mat descriptors_object_SIFT_FREAK, descriptors_scene_SIFT_FREAK;
    detector_freak->compute(img_object, keypoints_object_SIFT, descriptors_object_SIFT_FREAK);
    detector_freak->compute(img_scene, keypoints_scene_SIFT, descriptors_scene_SIFT_FREAK);

    std::cout << "SIFT_FREAK" << std::endl;
    Mat img_matches_SIFT_FREAK = process(img_object, img_scene, keypoints_object_SIFT, keypoints_scene_SIFT, descriptors_object_SIFT_FREAK, descriptors_scene_SIFT_FREAK, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches SIFT_FREAK", WINDOW_AUTOSIZE);
    imshow("Good Matches SIFT_FREAK", img_matches_SIFT_FREAK);

    // ------------------------------------ BRISK_FREAK ------------------------------------

    Mat descriptors_object_BRISK_FREAK, descriptors_scene_BRISK_FREAK;
    detector_freak->compute(img_object, keypoints_object_BRISK, descriptors_object_BRISK_FREAK);
    detector_freak->compute(img_scene, keypoints_scene_BRISK, descriptors_scene_BRISK_FREAK);

    std::cout << "BRISK_FREAK" << std::endl;
    Mat img_matches_BRISK_FREAK = process(img_object, img_scene, keypoints_object_BRISK, keypoints_scene_BRISK, descriptors_object_BRISK_FREAK, descriptors_scene_BRISK_FREAK, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches BRISK_FREAK", WINDOW_AUTOSIZE);
    imshow("Good Matches BRISK_FREAK", img_matches_BRISK_FREAK);

    // ------------------------------------ ORB_FREAK ------------------------------------

    Mat descriptors_object_ORB_FREAK, descriptors_scene_ORB_FREAK;
    detector_freak->compute(img_object, keypoints_object_ORB, descriptors_object_ORB_FREAK);
    detector_freak->compute(img_scene, keypoints_scene_ORB, descriptors_scene_ORB_FREAK);

    std::cout << "ORB_FREAK" << std::endl;
    Mat img_matches_ORB_FREAK = process(img_object, img_scene, keypoints_object_ORB, keypoints_scene_ORB, descriptors_object_ORB_FREAK, descriptors_scene_ORB_FREAK, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches ORB_FREAK", WINDOW_AUTOSIZE);
    imshow("Good Matches ORB_FREAK", img_matches_ORB_FREAK);

    // ------------------------------------ FAST_SURF ------------------------------------

    Mat descriptors_object_FAST_SURF, descriptors_scene_FAST_SURF;
    surf_detector->compute(img_object, keypoints_object_FAST, descriptors_object_FAST_SURF);
    surf_detector->compute(img_scene, keypoints_scene_FAST, descriptors_scene_FAST_SURF);

    std::cout << "FAST_SURF" << std::endl;
    Mat img_matches_FAST_SURF = process(img_object, img_scene, keypoints_object_FAST, keypoints_scene_FAST, descriptors_object_FAST_SURF, descriptors_scene_FAST_SURF, 1);

    // Mostrar correspondencias
    namedWindow("Good Matches FAST_SURF", WINDOW_AUTOSIZE);
    imshow("Good Matches FAST_SURF", img_matches_FAST_SURF);

    // ------------------------------------ FAST_SIFT ------------------------------------

    Mat descriptors_object_FAST_SIFT, descriptors_scene_FAST_SIFT;
    sift_detector->compute(img_object, keypoints_object_FAST, descriptors_object_FAST_SIFT);
    sift_detector->compute(img_scene, keypoints_scene_FAST, descriptors_scene_FAST_SIFT);

    std::cout << "FAST_SIFT" << std::endl;
    Mat img_matches_FAST_SIFT = process(img_object, img_scene, keypoints_object_FAST, keypoints_scene_FAST, descriptors_object_FAST_SIFT, descriptors_scene_FAST_SIFT, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches FAST_SIFT", WINDOW_AUTOSIZE);
    imshow("Good Matches FAST_SIFT", img_matches_FAST_SIFT);

    // ------------------------------------ FAST_BRISK ------------------------------------

    Mat descriptors_object_FAST_BRISK, descriptors_scene_FAST_BRISK;
    detector_BRISK->compute(img_object, keypoints_object_FAST, descriptors_object_FAST_BRISK);
    detector_BRISK->compute(img_scene, keypoints_scene_FAST, descriptors_scene_FAST_BRISK);

    std::cout << "FAST_BRISK" << std::endl;
    Mat img_matches_FAST_BRISK = process(img_object, img_scene, keypoints_object_FAST, keypoints_scene_FAST, descriptors_object_FAST_BRISK, descriptors_scene_FAST_BRISK, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches FAST_BRISK", WINDOW_AUTOSIZE);
    imshow("Good Matches FAST_BRISK", img_matches_FAST_BRISK);

    // ------------------------------------ FAST_ORB ------------------------------------

    Mat descriptors_object_FAST_ORB, descriptors_scene_FAST_ORB;
    detector_orb->compute(img_object, keypoints_object_FAST, descriptors_object_FAST_ORB);
    detector_orb->compute(img_scene, keypoints_scene_FAST, descriptors_scene_FAST_ORB);

    std::cout << "FAST_ORB" << std::endl;
    Mat img_matches_FAST_ORB = process(img_object, img_scene, keypoints_object_FAST, keypoints_scene_FAST, descriptors_object_FAST_ORB, descriptors_scene_FAST_ORB, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches FAST_ORB", WINDOW_AUTOSIZE);
    imshow("Good Matches FAST_ORB", img_matches_FAST_ORB);

    // ------------------------------------ FAST_FREAK ------------------------------------

    Mat descriptors_object_FAST_FREAK, descriptors_scene_FAST_FREAK;
    detector_freak->compute(img_object, keypoints_object_FAST, descriptors_object_FAST_FREAK);
    detector_freak->compute(img_scene, keypoints_scene_FAST, descriptors_scene_FAST_FREAK);

    std::cout << "FAST_FREAK" << std::endl;
    Mat img_matches_FAST_FREAK = process(img_object, img_scene, keypoints_object_FAST, keypoints_scene_FAST, descriptors_object_FAST_FREAK, descriptors_scene_FAST_FREAK, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches FAST_FREAK", WINDOW_AUTOSIZE);
    imshow("Good Matches FAST_FREAK", img_matches_FAST_FREAK);

    // ------------------------------------ SURF_BRIEF ------------------------------------
    //https://github.com/oreillymedia/Learning-OpenCV-3_examples/blob/master/example_16-02.cpp

    Mat descriptors_object_SURF_BRIEF, descriptors_scene_SURF_BRIEF;
    detector_brief->compute(img_object, keypoints_object_SURF, descriptors_object_SURF_BRIEF);
    detector_brief->compute(img_scene, keypoints_scene_SURF, descriptors_scene_SURF_BRIEF);

    std::cout << "SURF_BRIEF" << std::endl;
    Mat img_matches_SURF_BRIEF = process(img_object, img_scene, keypoints_object_SURF, keypoints_scene_SURF, descriptors_object_SURF_BRIEF, descriptors_scene_SURF_BRIEF, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches SURF_BRIEF", WINDOW_AUTOSIZE);
    imshow("Good Matches SURF_BRIEF", img_matches_SURF_BRIEF);

    // ------------------------------------ SIFT_BRIEF ------------------------------------

    Mat descriptors_object_SIFT_BRIEF, descriptors_scene_SIFT_BRIEF;
    detector_brief->compute(img_object, keypoints_object_SIFT, descriptors_object_SIFT_BRIEF);
    detector_brief->compute(img_scene, keypoints_scene_SIFT, descriptors_scene_SIFT_BRIEF);

    std::cout << "SIFT_BRIEF" << std::endl;
    Mat img_matches_SIFT_BRIEF = process(img_object, img_scene, keypoints_object_SIFT, keypoints_scene_SIFT, descriptors_object_SIFT_BRIEF, descriptors_scene_SIFT_BRIEF, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches SIFT_BRIEF", WINDOW_AUTOSIZE);
    imshow("Good Matches SIFT_BRIEF", img_matches_SIFT_BRIEF);

    // ------------------------------------ BRISK_BRIEF ------------------------------------

    Mat descriptors_object_BRISK_BRIEF, descriptors_scene_BRISK_BRIEF;
    detector_brief->compute(img_object, keypoints_object_BRISK, descriptors_object_BRISK_BRIEF);
    detector_brief->compute(img_scene, keypoints_scene_BRISK, descriptors_scene_BRISK_BRIEF);

    std::cout << "BRISK_BRIEF" << std::endl;
    Mat img_matches_BRISK_BRIEF = process(img_object, img_scene, keypoints_object_BRISK, keypoints_scene_BRISK, descriptors_object_BRISK_BRIEF, descriptors_scene_BRISK_BRIEF, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches BRISK_BRIEF", WINDOW_AUTOSIZE);
    imshow("Good Matches BRISK_BRIEF", img_matches_BRISK_BRIEF);

    // ------------------------------------ ORB_BRIEF ------------------------------------

    Mat descriptors_object_ORB_BRIEF, descriptors_scene_ORB_BRIEF;
    detector_brief->compute(img_object, keypoints_object_ORB, descriptors_object_ORB_BRIEF);
    detector_brief->compute(img_scene, keypoints_scene_ORB, descriptors_scene_ORB_BRIEF);

    std::cout << "ORB_BRIEF" << std::endl;
    Mat img_matches_ORB_BRIEF = process(img_object, img_scene, keypoints_object_ORB, keypoints_scene_ORB, descriptors_object_ORB_BRIEF, descriptors_scene_ORB_BRIEF, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches ORB_BRIEF", WINDOW_AUTOSIZE);
    imshow("Good Matches ORB_BRIEF", img_matches_ORB_BRIEF);

    // ------------------------------------ FAST_BRIEF ------------------------------------

    Mat descriptors_object_FAST_BRIEF, descriptors_scene_FAST_BRIEF;
    detector_brief->compute(img_object, keypoints_object_FAST, descriptors_object_FAST_BRIEF);
    detector_brief->compute(img_scene, keypoints_scene_FAST, descriptors_scene_FAST_BRIEF);

    std::cout << "FAST_BRIEF" << std::endl;
    Mat img_matches_FAST_BRIEF = process(img_object, img_scene, keypoints_object_FAST, keypoints_scene_FAST, descriptors_object_FAST_BRIEF, descriptors_scene_FAST_BRIEF, 2);

    // Mostrar correspondencias
    namedWindow("Good Matches FAST_BRIEF", WINDOW_AUTOSIZE);
    imshow("Good Matches FAST_BRIEF", img_matches_FAST_BRIEF);

    waitKey(0);
    return 0;
}
