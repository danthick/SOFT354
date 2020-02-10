# SOFT354 - Parallel Computation and Distributed Systems
## Description
This project is designed to compare the difference between serial and parallel run-times for a neural network using a parallel solution for matrix multiplication. CUDA has been used to implement the parallel solution.

## Neural Network
When the program is run it will train using a data set of 150 different [iris flowers](https://github.com/danthick/SOFT354/blob/master/iris-original.csv). The training time is what we are aiming to speed up as the back-propagation algorithm contains various matrix multiplication calculations. Below is a diagram of the network implemented:

![Neural Network Diagram](https://raw.githubusercontent.com/danthick/SOFT354/master/Exported%20Files/Neural%20Network%20-%20Updated.png?token=AGB2MEZDLZKGYNEM5EHY23S6JKSYO)

If you would like to know more about the network used in this project or any other algorithms then please read the report that is included in the repository.

## Running
The project contains both a serial and parallel project folders. You are able to modify the number of hidden nodes as well as the number of iterations the network trains for within the source code to be able to compare the run-times of each one.

## Results