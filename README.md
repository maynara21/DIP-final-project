# DIP Final Project - Cartooning face images

Author: Maynara Natalia  <br>

## Description and main objective 

Cartooning images is a subarea of the Non-Photorealistic Rendering (NPR) field, which is widely explored in computer graphics. The topic is relevant to the digital art area and it is adopted for entertainment purposes as well, considering the popularity of photo effects in social media such as Instagram or Snapshot. Also, cartoon images are a manner of conveying human-interpretable information at the same level of photorealistic imagery with less amount of data. This project consists of a face cartoon stylization algorithm, the result was achieved through the exploration of edge detection and color quantization techniques. The main challenge was applying a proper technique to diminish the false edge detection, which is likely to happen in face portrait images due to existence of small shadows and line expressions that are not generally represented by human artists. Similarly, finding smooth continuous contours and suppressing noise was also challenging.


## Input images description

The input images was human faces portraits obtained from:

https://unsplash.com/s/photos/face-portraits?orientation=portrait

The selected images were the ones with less cluttered background and resolution of roughly 640x960.


## Steps

The steps to reach the cartoon effect was:

<ul>
    <li>Convert the image to grayscale</li>
    <li>Apply the  Extended Difference of Gaussians (XDoG) to obtain the edges</li>
    <li>Apply color quantization and bilateral filter to the original image</li>
    <li>Combining the color and the edge images to obtain the final result</li>
<ul



The XDoG technique is detailed in the paper: <b> XDoG: An eXtended difference-of-Gaussians compendium including advanced image stylization </b>

It can be accessed through https://www.kyprianidis.com/p/cag2012/winnemoeller-cag2012.pdf
