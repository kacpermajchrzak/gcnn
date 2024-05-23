## Studio Porjektowe 2 - [Granulated deep learning and Z-numbers in motion detection and object recognition](https://link.springer.com/article/10.1007/s00521-019-04200-1)

# TODO

Priorytetem jest ogarnięcie jak zrobić sieć GCNN. Generalnie w folderze helpers jest model (SSD trained on COCO) który dobrze radzi sobie z detekcją ale nie da się go uzyc do transfer learningu. w folderze model znajduje się model SSD ale wytrenowany na czyms innym. Da się go wykorzystać do transfer learninu ale mozna tylko dolaczyc warstwy koncowe. Probowalem dolaczyc warstwe na poczatek ale wystepuja takie bledy ze sie poddalem.

### Brief description of the concept

1. **Granular Computing**:
    - Granular computing aims to handle complex information by dividing it into smaller, more manageable pieces called “granules.”
    - In this context, granules represent regions of interest within an image or scene.
2. **Z-numbers**:
    - Z-numbers are used to quantify the abstraction of semantic information.
    - They provide a way to express certainty or subjectivity in interpreting a scene.
    - By using Z-numbers, we can describe objects in a more natural and nuanced manner.
3. **Integration of Deep Learning and Granular Computing**:
    - Deep learning is computationally intensive, while granular computing offers computational gains.
    - The article proposes a methodology that combines the strengths of both approaches.
    - Instead of scanning the entire image pixel by pixel during deep learning, only representative pixels of each granule are scanned. This significantly speeds up computation time.
4. **Object Recognition and Scene Understanding**:
    - The system developed in the article recognizes both static objects in the background and moving objects in the foreground.
    - Rough set theoretic granular computing is used to define object and background models.
    - The method of tracking efficiently handles challenging cases, such as partially overlapped objects and sudden appearances.
5. **Linguistic Description of Scenes**:
    - The unique aspect lies in using Z-numbers to provide a granulated linguistic description of a scene.
    - This approach offers a more natural interpretation of object recognition, considering certainty and subjectivity.

### Problems:

- Calculation of T:
  
   - [**paper**] We made the parameters as adaptable as possible. The parameter values are dependent on the nature of the input data. For example, the number of previous frames n is dependent on the speed of the video. It is normally chosen as 7 for the sequences with speed of 15 frames per sec. The object-background threshold value T is initially chosen to be 30 as it was found to be experimentally suitable for most of the datasets. If the set OT = fi , then reduce T by 5.
   - [**our solution**] We are computing the threshold by obtaining the minimum sum of ROB and RBT in range(30, 70) and step 5.

- Network appearance
  
   - The major difference with that in [13] is that, we implemented it in granulated fashion, where the first convolution layer is granulated (as described before). The input image (granulated) is fed into the granulated convolution layer, followed by max pooling and rectified linear unit (ReLU). Then finally a fully connected layer of the neural network pro- duces the output classification. Use of this network for object recognition and tracking is described in Sect. 3.3 where the granulated convolution layer takes the object model and background model as input.

- Calculation of T in the quad tree decomposition:
  
   - A threshold T, where T is the average value of the first and third quartile of the image gray level distribution