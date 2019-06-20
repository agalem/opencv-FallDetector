#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/optflow.hpp>
#include <opencv2/optflow/motempl.hpp>

#include <iostream>
#include <vector>
#include <sys/time.h>
#include <time.h>

#define ESC 27

using namespace std;
using namespace cv;

double getStddev(vector<double>* vec);
double getMovementCoefficient(Mat* foreground,Mat* history);
void analyzePosition(Mat* frame, vector<double>* thetaRatio, vector<double>* aRatio, vector<double>* bRatio, vector<double>* xPos, vector<double>* yPos, vector<Point> largestContour);
void checkIfStaysInPlace(time_t start, bool* isChecking, bool* isFall, vector<double> xPos, vector<double> yPos);
void checkMovementAfterFall(bool* toBeChecked,  bool *isFall, vector<double> xPos, vector<double> yPos);

int main() {
    VideoCapture cap("/Users/agalempaszek/Desktop/opencv-lab/falls.avi");
    //VideoCapture cap(0);

    Mat frame, eroded, dilated;
    Mat backgroundImg, foregroundImg;
    Mat fgMaskMOG2;
    Mat history;

    Ptr<BackgroundSubtractorMOG2> pMOG2 = createBackgroundSubtractorMOG2();

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<Point> largestContour;
    vector<Moments> center(1);

    vector<double> thetaRatio;
    vector<double> aRatio;
    vector<double> bRatio;
    vector<double> xPos;
    vector<double> yPos;

    double movementCoefficientValue;
    double aRatioValue;
    double bRatioValue;
    double thetaRatioValue;

    bool toBeChecked = false;
    bool isFall = false;

    namedWindow("Original", WINDOW_AUTOSIZE);
    namedWindow("Mask", WINDOW_AUTOSIZE);

    Mat firstFrame;

    if(cap.isOpened()) {
        cap >> firstFrame;
        history = Mat::zeros(firstFrame.size().height, firstFrame.size().width, CV_32FC1);
    }

    //setup licznika czasu dla sprawdzenia ruchu po upadku
    time_t start;

    while(true) {

        cap >> frame;

        if(frame.empty()) {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            cap >> frame;
        }

        pMOG2->setHistory(20);
        pMOG2->setNMixtures(10);
        pMOG2->setDetectShadows(false);
        pMOG2->apply(frame, fgMaskMOG2);
        pMOG2->getBackgroundImage(backgroundImg);

        findContours(fgMaskMOG2, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));

        if(!toBeChecked) {
            largestContour.clear();
        }

        double maxArea = 0.0;

        for(size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i], false);
            if(area > maxArea && area > 500) {
                maxArea = area;
                largestContour = contours[i];
            }
        }


        if(!largestContour.empty()) {
            analyzePosition(&frame, &thetaRatio, &aRatio, &bRatio, &xPos, &yPos, largestContour);

        }

        double timestamp = (double)clock() / CLOCKS_PER_SEC;
        motempl::updateMotionHistory(fgMaskMOG2, history, timestamp, 0.5);

        movementCoefficientValue = getMovementCoefficient(&fgMaskMOG2, &history);
        thetaRatioValue = getStddev(&thetaRatio);
        aRatioValue = getStddev(&aRatio);
        bRatioValue = getStddev(&bRatio);

        cout << "Coeff: " << movementCoefficientValue << ",  Theta: " << thetaRatioValue << ", A: " << aRatioValue << ", B: " << bRatioValue << endl;


        if(!toBeChecked) {
            if(movementCoefficientValue > 80 && thetaRatioValue > 20 && (aRatioValue / bRatioValue) > 0.9 ) {
                cout << "Check" << endl;
                toBeChecked = true;
                start = time(0);
            }
        }

        if(toBeChecked) {
            checkIfStaysInPlace(start, &toBeChecked, &isFall, xPos, yPos);
        }

        if(isFall) {
            cout << "UPADEK" << endl;
            checkMovementAfterFall(&toBeChecked, &isFall, xPos, yPos);
        }


        Point textOrg(10, cap.get(CAP_PROP_FRAME_HEIGHT) - 15);

        string text;
        if(isFall && toBeChecked) {
            text = "Fall";
        } else if(!isFall && toBeChecked) {
            text = "Warning";
        }
        putText(frame, text, textOrg, FONT_ITALIC, 1, Scalar::all(255), 2, 8);


        imshow("Original", frame);
        imshow("Mask", fgMaskMOG2);
        imshow("History", history);

        if(waitKey(30) == ESC) break;
    }

    cap.release();
    frame.release();
    destroyAllWindows();

    return 0;
}


double getStddev(vector<double>* vec) {
    Scalar mean, stddev;
    if(!vec->empty()) {
        meanStdDev(*vec, mean, stddev);
    }
    return stddev[0];
}


double getMovementCoefficient(Mat* foreground,Mat* history) {
    double sumForeground = sum(*foreground)[0];
    double sumHistory = sum(*history)[0];
    return (sumHistory / sumForeground) * 100.0;
}

void checkIfStaysInPlace(time_t start, bool* isChecking, bool* isFall, vector<double> xPos, vector<double> yPos) {
    double secondsSinceStart = difftime( time(0), start);
    double xDevValue;
    double yDevValue;

//        cout << "x: [";
//        for(size_t i = 0 ; i < xPos.size(); i++) {
//            cout << xPos.at(i) << "  ";
//        }
//        cout << "]\n";
//
//        cout << "y: [";
//        for(size_t i = 0 ; i < yPos.size(); i++) {
//            cout << yPos.at(i) << "  ";
//        }
//        cout << "]\n";

    xDevValue = getStddev(&xPos);
    yDevValue = getStddev(&yPos);

//    cout << endl << "XDev: " << xDevValue << "\n";
//    cout << "YDev: " << yDevValue << "\n\n";

    if(xDevValue < 2 && yDevValue < 2) {
        *isFall = true;
    }

    if (!isFall && secondsSinceStart > 2) {
        *isChecking = false;
    }
}

void analyzePosition(Mat* frame, vector<double>* thetaRatio, vector<double>* aRatio, vector<double>* bRatio, vector<double>* xPos, vector<double>* yPos, vector<Point> largestContour) {
    Rect boundingRectangle = boundingRect(largestContour);
    rectangle(*frame, boundingRectangle, Scalar(0, 255, 0), 2);

    if(largestContour.size() > 5) {

        RotatedRect e = fitEllipse(largestContour);
        ellipse( *frame, e, Scalar(255, 0, 0), 2 );

        thetaRatio->push_back(e.angle);

        double a = (double)e.size.width / 2.0;
        double b = (double)e.size.height / 2.0;
        aRatio->push_back(a);
        bRatio->push_back(b);

        double x = e.center.x;
        double y = e.center.y;
        xPos->push_back(x);
        yPos->push_back(y);

        if(thetaRatio->size() > 10) {
            thetaRatio->erase(thetaRatio->begin());
        }

        if(aRatio->size() > 10) {
            aRatio->erase(aRatio->begin());
        }
        if(bRatio->size() > 10) {
            bRatio->erase(bRatio->begin());
        }
        if(xPos->size() > 10) {
            xPos->erase(xPos->begin());
        }
        if(yPos->size() > 10) {
            yPos->erase(yPos->begin());
        }
    }
}

void checkMovementAfterFall(bool* toBeChecked, bool *isFall, vector<double> xPos, vector<double> yPos) {
    double xDevValue = getStddev(&xPos);
    double yDevValue = getStddev(&yPos);

    cout << "X pos: " << xDevValue;
    cout << ", Y pos: " << yDevValue << "\n";

    if(xDevValue > 2 && yDevValue > 2) {
        *isFall = false;
        *toBeChecked = false;
    }
}