# messAIh
#### A dataset for Speech Emotion Recognition

---
license: mit

task_categories:
- audio-classification
language:
- en
tags:
- SER
- Speech Emotion Recognition
- Speech Emotion Classification
- Audio Classification
- Audio
- Emotion
- Emo
- Speech
- Mosei
  
pretty_name: messiah

size_categories:
- 10K<n<100K
---


DATASET DESCRIPTION

The MESSAIH dataset is a fork of [CMU MOSEI](http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/).

Unlike its parent, MESSAIH is indended for unimodal model development and focusses exclusively on audio classification, more specifically, Speech Emotion Recognition (SER).

Of course, it can be used for bimodal classification by transcribing each audio track.

MESSAIH currently contains 13,234 speech samples annotated according to the [CMU MOSEI](https://aclanthology.org/P18-1208/) scheme:

> Each sentence is annotated for sentiment on a [-3,3] Likert scale of:
> [−3: highly negative, −2 negative, −1 weakly negative, 0 neutral, +1 weakly positive, +2 positive, +3 highly positive].
> Ekman emotions of {happiness, sadness, anger, fear, disgust, surprise}
> are annotated on a [0,3] Likert scale for presence of emotion
> x: [0: no evidence of x, 1: weakly x, 2: x, 3: highly x].

The dataset is provided as a [parquet file](https://drive.google.com/file/d/17qOa2cFDNCH2j2mL5gCNUOwLxpgnzPmB/view?usp=drive_link). 

Provisionally, the file is stored on a [cloud drive](https://drive.google.com/file/d/17qOa2cFDNCH2j2mL5gCNUOwLxpgnzPmB/view?usp=drive_link) as it is too big for GitHub. Note that the original parquet file from August 10th 2023 was buggy and so was the Python script. 

To facilitate inspection, a truncated csv sample file is also provided, but it does not contain the audio arrays.

If you train a model on this dataset, you would make us very happy by letting us know.


UNPACKING THE DATASET

A sample Python script (check the top of the script for the requirements) is also provided for illustrative purposes.

The script reads the parquet file and produces the following:

1. A csv file with file names and MOSEI values (columns names are self-explanatory).
   
2. A folder named "wavs" containing the audio samples.


LEGAL CONSIDERATIONS

Note that producing the wav files might (or might not) constitute copyright infringement as well as a violation of Google's Terms of Service.

Instead, researchers are encouraged to use the numpy arrays contained in the last column of the dataset ("wav2numpy") directly, without actually extracting any playable audio.

That, I believe, may keep us in the grey zone.


CAVEATS

As one can appreciate from the charts contained in the "charts" folder, the dataset is biased towards "positive" emotions, namely happiness.

Certain emotions such as fear may be underrepresented, not only in terms of number of occurences, but, more problematically, in terms of "intensity".

MOSEI is considered a natural or spontaneous emotion dataset (as opposed to an actored or scripted one) showcasing "genuine" emotions.

However, keep in mind that MOSEI was curated from a popular social network and social networks are notoriously abundant in fake emotions.

Moreover, certain emotions may be intrinsically more difficult to detect than others, even from a human perspective.

Yet, MOSEI is possibly one of the best datasets of its kind currently in the public domain.

Also note that the original [MOSEI](http://immortal.multicomp.cs.cmu.edu/CMU-MOSEI/labels/) contains nearly twice as many entries as MESSAIH does.
