# Fake-Rapper-Helper
A Dual Convolutional Neural Network for Specific Style Music Generation from an Insipid Speech Clip


[1.Data Preparation](#prepare-data)

[2.Style Encoder Branch](#style-encoder)

[3.Content Preserver Branch](#content-encoder)


### Prepare Data
- The content encoder is pretrained on LibriSpeech dataset which is often used in ASR. It contains 1000 hours of English speech with a unversal sampling rate of 16k.
- For training  style the style encoder, we adopt the MagnaTagATune (MATA) dataser, which is created from online music pieces and labels. We use a subset of 6000 music pieces for training. To ensure music diversity, the data is first grouped into 10 broad music genres including Hip-pop and a subset of each category is selected.  We also examine the quality of the data and further include more Hip-hop music samples. Each music piece is clipped into 10-second segments before further analysis.


### Style Encoder
The aim of our style transformer is to learn a special encoding of the input audio. And this encoding should act as a latent feature representation of the music style (especially for the hip-hop style) of theinput.


### Content Encoder
