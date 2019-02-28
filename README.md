Mogra Flower Classification using Python and Tensorflow 

EXPORT QUALITY MOGRA CLASSIFICATION USING :
1. Calculating size of each Mogra bud using Contours and Euclidean Distance.
- This method was implemented first for calculating size of entire bud.
- Later changed it to calculate only size of bud(white portion)

2.Template matching
- Simple template matching and Multiscale matching
- This method is not effective as for simple matching requires same size image
  always

3. Deep Learning using Tensorflow object-detection API
- Specifically trained model to classify Mogra in two classes as
   mogra-gradeI(export quality)
   mogra-gradeII(not good quality)
-  ssd_mobilenet model used here.
