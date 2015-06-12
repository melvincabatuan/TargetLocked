#include "com_cabatuan_targetlocked_MainActivity.h"
#include <android/log.h>
#include <android/bitmap.h>
#include <math.h>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

#define  LOG_TAG    "TargetLocked"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  DEBUG 0


Mat *pSrcColor = NULL;
Mat *pResult = NULL;
Mat *pHsv = NULL;
Mat *plower_red_hue = NULL;
Mat *pupper_red_hue = NULL;
Mat *pred_hue = NULL;
Mat *pred_hue_gray =NULL;
Mat *pred_hue_binary =NULL;


void message(Mat &img, string text){
  int fontFace = FONT_HERSHEY_TRIPLEX; //  FONT_HERSHEY_SCRIPT_SIMPLEX
  double fontScale = 0.65;
  int thickness = 1.2;
  int baseline=0;
  Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);

  // center the text
  Point textOrg((img.cols - textSize.width)/2,
              (img.rows + textSize.height)/2);

   // put the text 
   putText(img, text, textOrg, fontFace, fontScale,
        Scalar::all(255), thickness, 8);
}


int rount(double num)
{
  return std::ceil(num - 0.5);
}

/*
 * Class:     com_cabatuan_targetlocked_MainActivity
 * Method:    detect
 * Signature: (Landroid/graphics/Bitmap;[BI)V
 */
JNIEXPORT void JNICALL Java_com_cabatuan_targetlocked_MainActivity_detect
  (JNIEnv *pEnv, jobject pClazz, jobject pTarget, jbyteArray pSource, jint pFilter){

AndroidBitmapInfo bitmapInfo;
   uint32_t* bitmapContent;

   if(AndroidBitmap_getInfo(pEnv, pTarget, &bitmapInfo) < 0) abort();
   if(bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) abort();
   if(AndroidBitmap_lockPixels(pEnv, pTarget, (void**)&bitmapContent) < 0) abort();

   /// Access source array data... OK
   jbyte* source = (jbyte*)pEnv->GetPrimitiveArrayCritical(pSource, 0);
   if (source == NULL) abort();

   /// cv::Mat for YUV420sp source
    Mat src(bitmapInfo.height + bitmapInfo.height/2, bitmapInfo.width, CV_8UC1, (unsigned char *)source);
    //Mat srcGray(bitmapInfo.height, bitmapInfo.width, CV_8UC1, (unsigned char *)source);

    /// Destination image
    Mat mbgra(bitmapInfo.height, bitmapInfo.width, CV_8UC4, (unsigned char *)bitmapContent);

    if(pSrcColor == NULL)
       pSrcColor = new Mat(bitmapInfo.height, bitmapInfo.width, CV_8UC3);
 
    Mat srcColor = *pSrcColor;
    cvtColor(src, srcColor, CV_YUV2RGB_NV21); // Correct colors  

    /// Result image
    if(pResult == NULL)
       pResult = new Mat(srcColor.size(), srcColor.type());
    Mat result = *pResult;   

    /// Sanity Check: PASSED
    //cvtColor(srcColor, mbgra, CV_BGR2BGRA);

/**************************************************************************************************/
    /// Native Image Processing HERE...  

    if(DEBUG){
      LOGI("Starting native image processing...");
    }

 
    /// Median Blur
    medianBlur(srcColor, srcColor, 3);

    /// Convert to HSV
    if(pHsv == NULL)
       pHsv = new Mat(srcColor.size(), srcColor.type());
    Mat hsv = *pHsv;
    //cvtColor(srcColor, hsv, CV_BGR2HSV); // Tracking BLUE
    cvtColor(srcColor, hsv, CV_RGB2HSV);   // Tracking RED
 


    /// Threshold HSV to keep Red pixels
    if(plower_red_hue == NULL)
       plower_red_hue = new Mat(srcColor.size(), CV_8UC1);
     Mat lower_red_hue = *plower_red_hue;

     if(pupper_red_hue == NULL)
       pupper_red_hue = new Mat(srcColor.size(), CV_8UC1);
     Mat upper_red_hue = *pupper_red_hue;
	
     inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), lower_red_hue); // Output: CV_8UC1
     inRange(hsv, Scalar(160, 100, 100), Scalar(179, 255, 255), upper_red_hue); // Output: CV_8UC1


     
     // Combine the lower and upper hue ranges
     if(pred_hue == NULL)
        pred_hue = new Mat(srcColor.size(), CV_8UC1);
     Mat red_hue = *pred_hue;
     addWeighted(lower_red_hue, 1.0, upper_red_hue, 1.0, 0.0, red_hue); // Output: CV_8UC1
 
 
  
    /// Output the red_hue
    if (pFilter == 1){
         message(red_hue, "red hue");
         cvtColor(red_hue, mbgra, CV_GRAY2BGRA); 
     }
 
 
    // Use the Hough transform to detect circles in the combined threshold image
    std::vector<cv::Vec3f> circles;

    // HoughCircles(red_hue, circles, CV_HOUGH_GRADIENT, 1, red_hue.rows/8, 50, 10, 0, 0); 
    // High False positives

    HoughCircles(red_hue, circles, CV_HOUGH_GRADIENT, 1, red_hue.rows/8, 80, 15, 0, 0);

      // Loop over all detected circles and outline them on the original image
      if(circles.size() != 0) {     

          for(size_t current_circle = 0; current_circle < circles.size(); ++current_circle) {
	    Point center(round(circles[current_circle][0]), round(circles[current_circle][1]));
            int radius = round(circles[current_circle][2]);
	    cv::circle(srcColor, center, radius, Scalar(0, 255, 0), 5);
          }
      }

 
    /// output the Tracked Circle    
     if (pFilter == 2){
            
            cvtColor(srcColor, mbgra, CV_BGR2BGRA);
    }
 
  
    /// output the Red Circle 
     if (pFilter == 3){
            result=Mat::zeros(srcColor.size(), srcColor.type());
            srcColor.copyTo(result, red_hue);
            message(result, "Target");
            cvtColor(result, mbgra, CV_BGR2BGRA);
    }
 


    if(DEBUG){
      LOGI("Successfully finished native image processing...");
    }
/*************************************************************************************************/


    /// Release Java byte buffer and unlock backing bitmap
    pEnv-> ReleasePrimitiveArrayCritical(pSource,source,0);
   if (AndroidBitmap_unlockPixels(pEnv, pTarget) < 0) abort();
}    
