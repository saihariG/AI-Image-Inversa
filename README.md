## Image Inversa

This is an end-to-end implementation of image search engine. The image search engines allow us to retrive similar images based on the query one

NOTE : This project is built based on this [course](https://www.udemy.com/course/practical-deep-learning-image-search-engine)

### Demo

![Image Inversa Demo](https://user-images.githubusercontent.com/52252342/125833288-f2c9b93c-a815-4dda-9a16-8838ddb1c9a8.gif)

## Project Instructions

### Getting Started

1. Clone the repository, and navigate to the downloaded folder.
```
git clone https://github.com/saihariG/AI-Image-Inversa
```

2. Install TensorFlow.
	
	
```
pip install tensorflow
```

3. Install a few pip packages.
```
pip install -r requirements.txt
```

4. Download CIFAR-10 dataset.

	  - [Darknet mirror](https://pjreddie.com/projects/cifar-10-dataset-mirror/)
    
    1. Put downloaded data into dataset folder
    2. Put all images from `dataset/train` to `static/images`
    
  
5. Run throuh [image-search project.ipynb](https://github.com/saihariG/AI-Image/blob/master/image-search-engine.ipynb) and generate all necessary files

6. Start the app.py file using the following command

```
python app.py
```