# bag of words for image retrieval

Python Implementation of Bag of Words for Image Retrieval using OpenCV and sklearn 

## Training the codebook and quantization
```
python findFeatures.py -t dataset/training/
```

* Query a single image(with inverted file index)
```
python search.py -i dataset/testing/all_souls_000000.jpg
```

- Query a single image(without/with RANSAC)

```
python search_RANSAC.py -i dataset/testing/all_souls_000000.jpg
```

- Query a single image(with alternative feedback)

```
python search_FEEDBACK.py -i dataset/testing/all_souls_000000.jpg
```

## Training and testing with hierarchical clustering

```
python Voc_Tree.py
```
## Dataset URL: https://pan.baidu.com/s/1HL812XJFevG_38Dw0e81yw
password: uvf6
