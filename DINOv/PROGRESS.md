# Testing of Visual Prompt Model
## Test 1

**Filename: test_1.py**
<br />
Using DinoV model only
<br />
Reference image given: Datasets/VP001/ref_img/can_1.jpg
<br />
Target image given: Datasets/VP001/1.jpg

**Observation**
1. Unable to segment the entire can fully
2. Segmented different parts of the can

**Result stored: test_result/test_1.jpg**

## Test 2
**Filename: test_2.py**
<br />
Using DinoV model only
<br />
Reference image given: Datasets/VP001/ref_img/can_1.jpg, Datasets/VP001/ref_img/can_2.jpg, Datasets/VP001/ref_img/can_3.jpg
<br />
Target image given: Datasets/VP001/1.jpg

**Observation**
1. Unable to segment the entire can fully
2. Segmented different parts of the can

**Result stored: test_result/test_2.jpg**

## Test 3
**Filename: test_3.py**
<br />
Using DinoV and sam model together
<br />
Reference image given: Datasets/VP001/ref_img/can_1.jpg
<br />
Target image given: Datasets/VP001/1.jpg

**Observation**
1. Unable to segment the entire can fully
2. Segmented different parts of the can

**Result stored: test_result/test_3.jpg**

## Test 4
**Filename: test_4.py**
<br />
Using DinoV and sam model together
<br />
Reference image given: Datasets/VP001/ref_img/can_1.jpg, Datasets/VP001/ref_img/can_2.jpg, Datasets/VP001/ref_img/can_3.jpg
<br />
Target image given: Datasets/VP001/1.jpg

**Observation**
1. Unable to segment the entire can fully
2. Segmented different parts of the can

**Result stored: test_result/test_4.jpg**

## Conclusion
Dinov model unable to segment the whole object.