# Part 2 - Image Classification #
## Prerequisites ##

- Java
- Maven

## Installation ##

To get started, install all dependencies with Maven

```bash
$ mvn install
```


## FeatureExtractionDemo ##

To run the feature extraction demo on your local computer, do the following:

* Ensure there are no spaces in your project's root folder path __(IMPORTANT!)__
    - For example, if your project path is _/user/me/Folder with spaces/myproject/_, the demo will fail!

* Install maven dependencies

```bash
$ mvn install
```

* Build jar package

```bash
$ mvn clean package assembly:single
```

* Download the [sample classified images](https://inclass.kaggle.com/c/image-classification2/data) and place them
    in the root folder of this project. Your folder structure should look like this:
    
    - [Root folder]
        - imagecl/
            - train/
                - bicycle/
                    - [images]
                - car/
                    - [images]
                - motorbike/
                    - [images]

* Run the demo

```bash
$ java -cp target/image-recognition-0.0.1-jar-with-dependencies.jar org.seng.image_recognition.demos.FeatureExtractionDemo
```
