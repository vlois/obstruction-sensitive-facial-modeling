# Obstruction Sensitive Facial Attention Modeling

## Overview

Certain regions of an image may attract more attention than others. The concept of visual salience is defined as "the distinct subjective perceptual quality which makes some items in the world stand out from their neighbors and immediately grab our attention" (Itti & Koch, 2001; Scholarpedia).

Visual perception in every day like rarely occurs under perfect conditions. Faces can be obstructed by shadows, hair, scarves, glasses, or other objects, obscuring significant features and likely altering attention distribution.

Occlusion is prevalent in the real world, yet existing facial perception systems may overlook it. This project investigates how attention to facial features changes when key features like eyes, nose, mouth become obstructed.

## Dataset
The Wider Facial Landmarks in the Wild (WLFW) dataset captures facial variations under real world conditions.
- 10000 images
- 98 facial landmark annotations per image
- Metadata labels: pose, illumination, expression, makeup, occlusion, blur

### Download the dataset manually:
WFLW dataset: http://wywu.github.io/projects/LAB/WFLW.html

After downloading, place the files in:
data/wflw/

Expected structure:
data/
  wflw/
    WFLW_images/
    WFLW_annotations/

## Research Questions

- How can attention be represented via landmark-defined facial regions?
- How does occlusion change visual salience?

### Step 1:
The first step is to verify that the annotations provided by the WFLW dataset align accurately with the facial geometry present in the images.

## References:
http://www.scholarpedia.org/article/Visual_salience
https://wywu.github.io/projects/LAB/WFLW.html

