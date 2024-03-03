# EDI_OCT_line_segmentation

This repository contains the code for **Enhanced depth imaging in spectral-domain optical coherence tomography (EDI SD-OCT)** line segmentation. It is based on the paper [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf%EF%BC%89) by **Olaf Ronneberger, Philipp Fischer, and Thomas Brox**.
The model is used to segment the lines of layers of the retina in the EDI SD-OCT images.

## Installation

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To clean initial dataset, use the following command:

```bash
python data_cleansing.py
```

To convert dataset to `.npy`, use the following command:

```bash
python data2npy.py
```

To preprocess dataset, use the following command:

```bash
python data_preprocess.py
```

To run the code, use the following command:

```bash
python main.py
```

## Results

The following images show the original, labeled, and segmented EDI OCT images.

<div>
<img src="https://github.com/rebedy/EDI_OCT_line_segmentation/blob/main/imgs/original_L.png" width="30%">
<img src="https://github.com/rebedy/EDI_OCT_line_segmentation/blob/main/imgs/marked_L.png" width="30%">
<img src="https://github.com/rebedy/EDI_OCT_line_segmentation/blob/main/imgs/segmented_L.png" width="30%">
</div>

**[Original left EDI OCT]** **[Labeled left EDI OCT]** **[Segmented left EDI OCT]**

