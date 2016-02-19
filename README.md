# Impressionist Image Rendering
Uses Canny edge detection to produce impressionist/painterly renderings of images. 

First, canny edgels are obtained for the input image(s) (complete with non-maximal suppresion, hysteresis, etc.); then, paint strokes are made perpendicular to the edge directions. Additionally, a certain amount of random perturbations in paintstroke-angles was implemented, to give the results a more natural, impressionist feel. Various radii of paint strokes were tested before settling on radius=1 as a good choice.

## Results
Below are two images (both the originals, and the final impressionist renderings) that were used to test the renderer. They are provided here just to showcase a couple of sample results:

### Original Images
<img src="/img/RV.jpg" width="500px" height="300px"/>

<img src="/img/orchid.jpg" width="300px" height="250px"/>

### Impressionist Renderings
<img src="/img/part6_RVoutput_rad1.png" width="500px" height="300px"/>

<img src="/img/part6_output_rad1.png" width="300px" height="250px"/>