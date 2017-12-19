# LBP-for-Aging-Face-Recognition

Hello everyone, this is my project for EL9123 "Introduction to Machine Learning" @ NYU Tandon.

# Project Summary

Face recognition became a popular issue in the late 20th century and gained huge progress during the 1990s. But, aging issues still remain as a huge problem for face recognition. 

In this project, I am trying to find a good way to solve this problem. The idea is to use Local Binary Patterns (LBP) to effectively extract facial features that are invariant to aging process and conduct aging face recognition based on selected aging database from MORPH Album 2, one of the most famous public available aging face database in the world.

For detailed description, please refer to the project report. 

# About Dataset

The MORPH Album 2 database is the largest aging database containing about 78000 longitudinal face images of more than 13000 people at different ages. For each subject in the MORPH Album 2 database, the maximum age gap should be 1 to 5. 

In this project, 700 images of 100 subjects were randomly selected from the database. Each selected subject has 6 training images at younger age and 1 testing image at older age. These selected images have relatively good quality with similar lighting conditions, facial expressions, poses, and the like. Therefore, this dataset was suitable for the aging-related experiments in this project.

The dataset was preprocessed using my image preprocessing program. It is provided as a zip file named MORPH_PROC.

Please download the dataset and unzip it to the same folder of the codes.  

# User Guide

IDE: Microsoft Visual Studio 2015 Community

Language: C++

In this program, an external library OpenCV 2.4.13 was applied in our design. Please download and configure it before running the program.
The link for OpenCV is provided here. https://opencv.org/releases.html
