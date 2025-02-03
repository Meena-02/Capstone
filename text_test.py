from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
from PIL import Image
import os

ocr = PaddleOCR(lang='en')

img = os.path.join('Dataset/Text/TM001/1.jpg')

result = ocr.ocr(img)

# [
#     [
#         [[[304.0, 59.0], [557.0, 59.0], [557.0, 164.0], [304.0, 164.0]], ('PORK', 0.9959486126899719)], 
#         [[[46.0, 90.0], [233.0, 75.0], [236.0, 113.0], [48.0, 128.0]], ('PREMIUM', 0.9945175051689148)], 
#         [[[57.0, 180.0], [808.0, 175.0], [809.0, 308.0], [58.0, 314.0]], ('LUNCHEON MEAT', 0.9788351655006409)]
#     ]
# ]
# result[0] -> returns entire list of detected text
# result[0][0] -> returns the first detected text
# result[0][0][0] -> returns the coordinates of the detected text
# result[0][0][1] -> returns the text detected and the confidence score

for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)


result = result[0]
image = Image.open('Dataset/Text/TM001/1.jpg').convert('RGB')
boxes = [line[0] for line in result]
texts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]

im_show = draw_ocr(image, boxes, texts, scores, font_path='PaddleOCR/doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.show()