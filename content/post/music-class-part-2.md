---
author: author
date: 2017-06-04T14:08:34+02:00
description: 
draft: false
keywords:
- machine learning
- deep learning
- music
- classification
tags:
- machine learning
- music
- computer vision
title: Deep learning music classifier part 2. Computer says no!
topics:
- topic 1
type: post
---



## Recap

Last post I described what my motivations were to start building a music classifier, or at least attempt to build one. The post also described how I collected a dataset, extracted important features and clustered the data based on their variance. [You can read the previous post here.]({{< ref "post/music-class-part-1.md" >}})

This post describes how I got my feet wet with classifying music. Spotify kind of sorted the data I've downloaded by genre. However when I started looking at the data, I found that a lot of songs were interchangeable between the genres. If I am to use data that do not belong solely to one class but should be labeled as both soul and blues music, how can a neural network learn to distinguish between them?

Take the genres blues and rock as an example. In the data set both genres contained the song *Ironic* made by *Alanis Morissette*. In my opinion this songs doesn't belong to any of these genres. To address this problem I chose to train the models on the more distinguishable genres, namely:

| Genre          | No. of songs  |
| ---------------|---------------|
| Hiphop         | 1212          |
| EDM            | 2496          |
| Classical music| 2678		 |
| Metal          | 1551		 |
| Jazz           | 417     	 |


I believe that these genres don't overlap too much. For instance a metal song can have multiple labels, but I am pretty sure classical and jazz won't be one.

Now the genres are chosen, the data set is still far from perfect. As you can see in the table above the data is not evenly distributed. So I expect that the trained models will find it harder to generalise for music that is less represented in the data. Jazz music for instance will be harder to train than classical music as the latter is is abundant in the data set. 

Another example of the flaws in the data set is that there are songs that aren't even songs at all! During training I realised that the data set contained songs that were labeled as metal. When actually listening to the audio they seemed to be just recordings of interviews with the band members of Metallica. I really didn't feel like manually checking every song for the correct label, thus I've accepted that the models will learn some flawed data. This luckely gives me an excuse when some predictions are really off :).

</br>

## Multilayer perceptron
The first models I trained were multilayer perceptrons also called feed forward neural networks. These nets consist of an input layer, one or multiple hidden layers and an output layer. The output layer should have as many nodes as classes you want to predict. In our case this would be five nodes. The figure below shows an example of a multilayer perceptron.

{{< figure src="/img/post-8-dl-music/mlp.png" title="Multilayer perceptron" >}}

</br>

### data augmentation
In the last post I described that the feature extraction has converted the raw audio data into images. Every pixel of these images is an input to the neural network. As we can see in the figure above. Every input is connected to all the nodes of the next layer. Lets compute the number of parameters this would take for a neural net with one hidden layer for one song's image.

As a rule of thumb we can choose the number of hidden nodes somewhere between the input nodes and the output nodes.

The input image has a size of <span>\\(10 x 2548 = 25,480 \\)</span> pixels. The number of output nodes is equal to the number of classes = 5 nodes. If we interpolate the amount of hidden layer connections:

<div>$$hidden \ nodes = \frac{25480 + 5}{2}= 12743$$</div> 

The total connections of the neural net would become:


<div>$$connections = 25480 \cdot 12743 + 12743 \cdot 5 = 3.24 \cdot 10^8$$</div> 

This sheer number of connections for a simple neural net with one hidden layer was way too much for my humble laptop to deal with. Therefore I started to reduce the amount of input data. 

Computers see images as 2D arrays of numbers. Below is an example shown of a random grayscale image of 9x6 pixels. The pictures I used to classify information contain MFCC spectra. Every column of these images is a discrete time step t<sub>i</sub> and contains time information. The rows contain frequency coefficient information.

{{< figure src="/img/post-8-dl-music/img_grayscale.png" title="Grayscale picture (as a computer sees it)" >}}

</br>

I reduced the amount of information by only feeding every 20th column of the image. This way the frequency information of a timestep t<sub>i</sub> remains intact however the amount of time steps is divided by 20. This is of course a huge loss of information, but the upside of this rough downsampling was that I could expand my data set with 20 times the amount of data. As I could just change the offset of counting every 20th column. The amount of input nodes reduced to <span>\\(\frac{25480}{20} = 1274 \\)</span> nodes. Which is a number of input nodes my laptop can handle and gives me the possibility expand the network deeper.

### model results

The best results were yielded from a model with one hidden layer and a model with three hidden layers. The amount of hidden nodes were divided as following:

* Model 1; hidden layer 1; 600 nodes
* Model 2; hidden layer 1; 600 nodes
* Model 2; hidden layer 2; 400 nodes
* Model 2; hidden layer 1; 100 nodes

To prevent the models from overfitting I used both l2 regularisation and dropout. 20% of the total data set was set aside as validation data.

Below are the results plotted. The dashed lines show the result the accuracy of the models on the training data. The solid lines are the validation data results. This is an indication of how accurate the models are on real world data. As in music they haven't heard before, or in this case music they haven't **seen** before.


{{< figure src="/img/post-8-dl-music/mlp-results.png" title="Accuracy of the feed forward models" >}}

The maximum accuracy on the validation set was **82%**. I am quite pleased with this result as I threw away a lot of information with the rigorous dropping of the columns. In a next model I wanted to be able to feed all the information stored in a song's image to the model. Furthermore it would also be nice if the model could learn something about the order of frequencies. In other words that the model learns something about time information, maybe we can even call it rhythm. 

I don't believe a feed forward neural net does this. For a feed forward network I need the rescale the 2D image matrix into a 1D vector which probably is also some (spatial) information loss. 

</br>

## Convolutional
I believe the solution to some of the wishes I named in the last section are convolutional neural networks (CNN). </br>
Convolutional neural networks work great with images as they can deal with spatial information. An image contains 2D information. The input nodes of a CNN are not fully connected to every layer so the model can deal with large input data without blowing up the amount of nodes required in the model. Images can have a high number of megapixels, thus a lot of data. I think these properties of CNN are also convenient for our problem of classifying music images.

{{< figure src="/img/post-8-dl-music/cnn.jpg" title="Convolutional neural network in action" >}}

</br>

### shared nodes
In a convolutional neural net the weights and biases are shared instead of fully connected to the layers. For an image this means that a cat in the left corner of an image leads to the same output as the same cat in the right corner of an image. For a song this may be that a certain rythm at the beginning of song will fire the same neurons as that particular rhythm at the end of the song.

### spatial information
The images I have created in previous post contain spatial information. As said before the x direction shows time information, the y directions shows frequency coefficients. With CNN's I don't have to reshape the image matrix in a 1D vector. The spatial information stays preserved. 


### data set
Both the shared nodes and the spatial information preservation does intuitively seem to make sense. The only drawback compared to the previous model is the amount of data. By the drastic downsampling in columns I was able to created a twentyfold of the original number of songs. Now I was feeding the whole image to the model. So the amount of data remains equal to the amount of downloaded songs. 

In image classification the data sets are often augmented by rotating and flipping the images horizontally. This augmentation seems reasonable as a rotated cat is still a cat. [The proof of this statement is provided!](https://www.youtube.com/watch?v=STY1Ut9JhtY) I am not certain this same logic is true for music. Doesn't a rotated songs image hustle some time and frequency information? I am not sure about the previous statement, thus I chose to leave the songs images intact and work with the amount of songs I've got.

### results

The convolutional models yielded far better results than the feed forward neural nets. The best predicting model contained 3 convolutional layers with respectively 16, 32 and 32 filters. After the convolutional layers there was added a 50% dropout layer followed by a fully connected layer of 250 neurons. For who is interested in all the technicalities a graph of the model is [available here.](https://github.com/ritchie46/music-classification/blob/master/cnn_16_32_32_d0_0.5_250.png)

The accuracy gained during training of the two best predicting models are shown in the plot below.

{{< figure src="/img/post-8-dl-music/cnn-results.png" title="Accuracy of the convolutional neural nets" >}}

</br>

The nummerical results in terms of the prediction accuracy are shown in the table below. Note that the accuracy results was computed with all the data, training and validation data. On real world data the results would probably be a bit lower.


|            | precision  |  recall | n songs
|------------|------------|---------|----------
|   Hiphop   |      0.87  |    0.94 |     1208
|   EDM      |      0.95  |    0.91 |     2493
|  Classical |      0.97  |    0.98 |     2678
|  Metal     |      0.96  |    0.97 |    1571
|   Jazz     |      0.86  |    0.78 |    416

The table shows the precision and the recall of the labels. The precision is the amount of good classifications divided by the total amount times that label was predicted. Recall is the amount of good classifications divided by the total occurrences of that specific label in a data set. So let's say I've trained my model very badly and it only is able to predict jazz as a label. Such a hypothetical model would have 100% recall on the jazz label. 

* Total number of songs: 11,366 
* Total number of jazz songs: 416

<div>$$recall = \frac{416}{416} \cdot 100= 100 \%$$</div> 

The precision of the model would however be low as the model has classified all the songs as jazz.

<div>$$precision = \frac{416}{11366} \cdot 100= 3.66 \%$$</div> 

So back to the table we can see that EDM, metal and classical music score best on both precision and recall. Hiphop has a good recall score. It scores a bit lower on precision meaning that it mostly correct in labeling a hiphop song given that it is a hiphop song, but the model makes more mistakes by labeling a song as hiphop that should not be classified as such. </br>
Jazz scores worst. Which is as expected as the amount of jazz songs in the data set is substantially lower than the other labels.


## tl;dr
This post focusses on the classification of the songs. I've described the results I yielded with feed forward and convolutional neural networks that classified music based on MFCC images.

The convolutional model is used in the small widget below. You can try to search by an artist and a track name. If a match is found in the spotify library an mp3 is downloaded and converted to a MFCC image. When the model has finished 'viewing' the song it will make his best prediction of the known labels.

Before a prediction can be made the mp3 needs to be downloaded, converted to a .wav audio file, converted to a MFFC image and finally fed to the model. I only have got a small web server, so please be patient. A prediction is coming your way. :)

Oh and remember... it only knows:

* Hiphop
* EDM
* Classical music
* Metal
* Jazz

It is of course also interesting to see how the model classifies genres it hasn't learned. But I cannot promise you it would be a sane prediction though.

<div class="base-input">
<label>Artist:</label> <input type="text" class="input-style-1" value="coolio" id="artist"></input> </br>
<label>Song:</label> <input type="text" class="input-style-1" value="gangsta's paradise" id="track"></input> </br>
<input type="submit" value="Predict!" id="predict">
<input type="submit" value="Pause song" id="stop">

<div id="result"></div>


<div id="mfcc-wrap"></div>
</div>

</br>


<style>
label {
    width: 30%;
    display: inline-block;
}

input {
    padding: 0.2em 0.2em 0.2em 0.2em;
    box-sizing: border-box;
    margin: 0.2em 0.2em 0.2em 0.2em;
    width: 22rem;
}

.input-style-1 {
    padding: 10px;
    border: None;
    border-bottom: solid 2px #c9c9c9;
    transition: border 0.3s;
}
.input-style-1:focus,
.input-style-1.focus {
    border-bottom: solid 2px #969696;
}

.input-container {
    margin: 1em 1em 1em 1em;
    text-align: left;
    display: block;
}

.base-input {
    margin: 1em 1em 1em 1em;
    padding: 1em 1em;
    background: #efefef;
    width: 45rem;
    border: solid 2px #969696;

}

input[type=submit] {

margin: 0.5em 0.5em 1em 1em;
width: 15rem

}
</style>


<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
<script>
var audio = new Audio('')
$("#predict").click(function(){
$("#result").html("Searching for <i>" + $("#artist").val() + " - " + $("#track").val() + "</i>...</br>");
      $.ajax({
        url: "/d/music-prediction",
        data: {
            'artist': $("#artist").val(),
            'track': $("#track").val()
        },
        dataType: "json",
        success: function (data) {
	    if (data.response) {
		    $("#result").html("The best prediction of the model:"+
" <ol>"+
" <li>" + data.first[0] + ", certainty: " + Math.round(data.first[1] * 100) + "%</li>"+
" <li>" + data.second[0] + ", certainty: " + Math.round(data.second[1] * 100) + "%</li>" +
" </ol>"
);


		    audio.pause();
		    var img = document.createElement('img');
		    img.src = 'data:image/jpeg;base64,' + data.mfcc;
		    $("#mfcc-wrap").html(img);
		    audio = new Audio(data.mp3);
		    audio.play();
	}
	else {
$("#result").html("The song you are searching could not be found or has no available preview.");
}
        }
    });
});

document.getElementById('stop').onclick = function() {
	audio.pause()
};
</script>

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_CHTML">
</script>


