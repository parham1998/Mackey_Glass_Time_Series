# Mackey_Glass_Time_Series
Implementation of a two-layer perceptron (from scratch) with four back-propagation methods in Python

## Mackey-Glass Time Series
For generating the Mackey-Glass time series, I used the formula below:
![1](https://user-images.githubusercontent.com/85555218/145380897-e3df4347-ac66-4857-b56f-7dced6258856.png)

You can also see the plot of data below: <br />
![data_series](https://user-images.githubusercontent.com/85555218/145381345-e920a80a-293a-4386-a1e4-a9016f652b77.png)

The goal is to predict x(t + 1) from x(t - 2), x(t - 1) and x(t). Thus, our neural network has three-dimensional inputs: <br />
input: [x(t - 2), x(t - 1), x(t)] <br />
output: x(t + 1) <br />
I've considered 70 percent of the data as training data, 25 percent as validation data, and 5 percent as my test data. And for more stable training, I have normalized the data by the min-max normalization method. <br />
As a result of normalizing the data, the data range changes to [0, 1], allowing us to use the unipolar sigmoid function as our activation function.

## The architecture of two-layer perceptron
As I mentioned above, the input dimension of the network is three. I've considered five neurons and one neuron for the hidden and output layers, respectively. This is the network architecture:
![Screenshot (545)](https://user-images.githubusercontent.com/85555218/145389231-936fbc9e-779b-4a68-ab09-42bc778a58e5.png)

### Feed-forward:
![Screenshot (548)](https://user-images.githubusercontent.com/85555218/145398103-b2bc23fb-0f05-404a-bffb-6f300e7e752f.png)

### Back-propagation:
At first, A uniform distribution is used to randomly initialize the network's weights (W1, W2), then I've used different methods to train the network, which can be seen below:

*1: stochastic gradient descent* <br />
This method updates the network parameters (weights) at every time step, making it sensitive to noisy data.
![Screenshot (551)](https://user-images.githubusercontent.com/85555218/145407592-94caf2ec-f3ef-4619-abfe-713a9aafc81b.png)

You can also see the results below:
![ezgif com-gif-maker](https://user-images.githubusercontent.com/85555218/145400228-bc0ac7e2-4a0e-4871-992a-e5a08ee215fb.gif)

*2: emotional learning* <br />
This method is very similar to the SGD method but uses the previous step error to make the network learn faster and more accurately.
![Screenshot (552)](https://user-images.githubusercontent.com/85555218/145407619-0cf84761-8597-4807-b729-3d222ad995f8.png)

*3: adaptive learning rate* <br />
In this method, we assign a different learning rate to each of the trainable parameters, which are called adaptive learning rates. For each element of the weights matrix, we consider a different learning rate, which is also trained during the learning process. (As I said before, this method has more learning parameters (twice as much as before), and MSE might fluctuate and need more time for training.)
![Screenshot (553)](https://user-images.githubusercontent.com/85555218/145407636-28acfd04-20b2-47f4-b72c-19b7c3385a7f.png)

*4: levenberg marquardt* <br />
