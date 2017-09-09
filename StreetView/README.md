# GoogleSVLibrary
## Introduction
This is python package to query Google Street View Image and Depth map.

### Panorama
Image Panorama is a 3328 x 6656 image.
<center><img src="src/pano.jpg"></center>

### Depth Map
Depth Map is a 256 x 512 numpy array.
<center><img src="src/image.jpg"></center>
<center><img src="src/depth.png"></center>

### Usage
Panorama is 3328 x 6656 image. <br>
Depth is 256 x 512 numpy array.

```python
from GoogleSVLibrary import StreetViewInfo
from GoogleSVLibrary import StreetViewImage
from GoogleSVLibrary import StreetViewDepth

# Input is [lat, lon]
panoid = StreetViewInfo.GetIDByLoc([24.7881509,121.0104312]) 

# If there is no pano here, panoid will be None
if panoid is None: 
  print 'No pano'
  exit()

# it will store pano as './pano_%s.jpg'%panoid
StreetViewImage.GetPanoByID_full(panoid, './')

# depth_map is our depth data, which is 256 x 512
depth_map = StreetViewDepth.GetDepthMap(panoid = panoid) 
```
