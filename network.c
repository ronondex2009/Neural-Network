/*
Multilayer Perceptron Neural Network (MLP) by Ronan Kai Guin Adkins:
    You are free to use, copy, and modify this program.
    If used, please credit me (but you don't have to!)
Details:
    This is an MLP network (a feed-forward neural network with multiple
    layers and non-linearity) that has the following:
        - Leaky ReLU for all but last layer
        - Output layer uses Sigmoid
        - regularization (may be disabled) ...
            > Not a conventional type (L1 or L2);
            > All parameters except the output
            weights and biases, go through a nonlinear 
            tanh function that has been sized up to
            10,000. The main purpose of this is to
            solve the "exploding gradients" problem
            that is common with ReLU.
        - Gradient Descent Backpropogation
        - Check how much memory is allocated to the network:
            > "struct network net; net.bytesInUse;"
            > this value is for the user. program does
            not use it.
    
    1) initialize new network struct POINTER    "struct net *network;"
    2) use function "setupNetwork" to build     "int err = setupNetwork(&network, int hiddenLayerLength, int inputLayerWidth, int hiddenWidth, int outputWidth);"
        * returns 1 if it is unable to alloc    "if (err != 0) //HANDLE ERROR HERE//"
    3) train..                                       --go to BACKPROP training--
    4) propogate with inputs                    "propogate(&network, double inputs[]);"
        * you can access output through struct  "outputs[i] = net.outputN[i];"
    ---BACKPROP---
    1) make input and output training set       "double inputs[input_layer_width × training_examples] = { ... }"
        * they may be different sizes.          "double outputs[output_layer_width × training_examples] = { ... }"
    2) backprop                                 "int err = gradientDescent(&network, double inputs[], double outputs[], int training_examples, double training_speed);"
        * training speed should be around 0.01 - 0.001.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

int UseRegularization = 1; //1 on, 0 off (please remember that this is used to prevent exploding gradients)
// gradients becomimg NaN, Ind, or Inf is caused by exploding gradients.
// these issues may not occur for simpler networks. May also be used for overfitting prevention.

struct network
{
    int inputWidth;
    int outputWidth;
    int hiddenWidth;
    int hiddenLength;
    unsigned int bytesInUse;
    double error;
    double *inputN; //input neurons 1D
    double *outputN; //output neurons 1D
    double *outputB; //output biases 1D
    double *inputR; //input raw neurons 1D
    double *outputR; //output ... 1D
    double *inputW; //input weights 2D
    double *hiddenN; // 2D
    double *hiddenB; // 2D
    double *hiddenR; // 2D
    double *hiddenW; // 3D
    double *toOutputW; //2D
};

void* callocTrack(unsigned int size, unsigned int size_t, int* errorCode, unsigned int* bytesUsedTotal)
{
    void* result;
    result = calloc(size, size_t);
    if(result)
    {
        *errorCode = 0;
        *bytesUsedTotal += size*size_t;
        return result;
    }
    *errorCode = 1;
    return NULL;
}
void freeTrack(unsigned int size, unsigned int size_t, void* value, unsigned int* bytesUsedTotal)
{
    free(value);
    *bytesUsedTotal -= size*size_t;
}

//free data
void freeNetwork(struct network *networkstruct)
{
    free(networkstruct->inputN);
    free(networkstruct->hiddenN);
    free(networkstruct->outputN);
    free(networkstruct->inputR);
    free(networkstruct->hiddenR);
    free(networkstruct->outputR);
    free(networkstruct->inputW);
    free(networkstruct->hiddenW);
    free(networkstruct->outputB);
    free(networkstruct->hiddenB);
    free(networkstruct->toOutputW);
    free(networkstruct);
}

//setup a new network array
int setupNetwork(struct network *networkstruct, int length, int widthIn, int widthHidden, int widthOut)
{
    int errorcode = 0;
    networkstruct->bytesInUse = 0;
    networkstruct->inputWidth = widthIn;
    networkstruct->outputWidth = widthOut;
    networkstruct->hiddenWidth = widthHidden;
    networkstruct->hiddenLength = length;
    networkstruct->inputN = (double*)callocTrack(widthIn, sizeof(double), &errorcode, &networkstruct->bytesInUse);
    networkstruct->outputN = (double*)callocTrack(widthOut, sizeof(double), &errorcode, &networkstruct->bytesInUse);
    networkstruct->hiddenN = (double*)callocTrack(widthHidden*length, sizeof(double), &errorcode, &networkstruct->bytesInUse);
    networkstruct->outputB = (double*)callocTrack(widthOut, sizeof(double), &errorcode, &networkstruct->bytesInUse);
    networkstruct->hiddenB = (double*)callocTrack(widthHidden*length, sizeof(double), &errorcode, &networkstruct->bytesInUse);
    networkstruct->inputR = (double*)callocTrack(widthIn, sizeof(double), &errorcode, &networkstruct->bytesInUse);
    networkstruct->outputR = (double*)callocTrack(widthOut, sizeof(double), &errorcode, &networkstruct->bytesInUse);
    networkstruct->hiddenR = (double*)callocTrack(widthHidden*length, sizeof(double), &errorcode, &networkstruct->bytesInUse);
    networkstruct->inputW = (double*)callocTrack(widthIn*widthHidden, sizeof(double), &errorcode, &networkstruct->bytesInUse);
    networkstruct->hiddenW = (double*)callocTrack(widthHidden*widthHidden*length, sizeof(double), &errorcode, &networkstruct->bytesInUse);
    networkstruct->toOutputW = (double*)callocTrack(widthHidden*widthOut, sizeof(double), &errorcode, &networkstruct->bytesInUse);

    if(!errorcode==0)
    {
        freeNetwork(sizeof(networkstruct));
        return 1;
    }
    
    for(int j = 0; j < widthIn; j++)
    {
        for(int k = 0; k < widthHidden; k++)
        {
            networkstruct->inputW[j*widthHidden+k] = (double)(rand() % 1000)/500-1*2;
        }
    }
    for(int i = 0; i < length; i++)
    {
        for(int j = 0; j < widthHidden; j++)
        {
            for(int k = 0; k < widthHidden; k++)
            {
                networkstruct->hiddenW[(i*widthHidden*widthHidden)+(j*widthHidden)+k] = (double)(rand() % 1000)/500-1*2;
            }
            networkstruct->hiddenB[i*widthHidden+j] = (double)(rand() % 1000)/500-1*2;
        }
    }
    for(int j = 0; j < widthOut; j++)
    {
        networkstruct->outputB[j] = (double)(rand() % 1000)/500-1*2;
        for(int i=0; i < widthHidden; i++)
        {
            networkstruct->toOutputW[i*widthOut+j] = (double)(rand() % 1000)/500-1*2;
        }
    }
    return 0;
}

double sigmoid(double x) //my second favorite function
{
    double result = 1/(1+exp(-x));
    if(!isfinite(result))
        return 0.5;
    return result;
}
double sigmoidDer(double x) //my first favority function
{
    double result = sigmoid(x)*(1-sigmoid(x));
    if(!isfinite(result))
        return 1;
    return result;
}
double ReLU(double x) //why..?
{
    double result = (double)(x>=0) ? x: x*0.1; //"leaky" ReLU
    //if(!isfinite(result))
    //    return 1;
    return result;
}
double ReLUDer(double x) //why!?
{
    double result = (double)(x>=0) ? 1: 0.1;
    return result;
}
double ParameterSquish(double x) //My own take on regularlization!
{ //this is basically a tanh function sized up by ten thousand. Used to prevent weights becoming NaN, Inf, or Ind.
    double result;
    result = 10000* ( (exp(x/10000)-exp(-x/10000))/(exp(x/10000)+exp(-x/10000)) );
    return result;
}
double ParameterSquishDer(double x) //why did I need this again..?
{
    double result;
    result = 1-pow(ParameterSquish(x)/10000, 2);
    return result;
}
//propogate the network
void propogate(struct network *net, double inputs[])
{    
    memset(net->inputR, 0, net->inputWidth*sizeof(double));
    memset(net->outputR, 0, net->outputWidth*sizeof(double));
    memset(net->hiddenR, 0, net->hiddenWidth*net->hiddenLength*sizeof(double));
    memset(net->inputN, 0, net->inputWidth*sizeof(double));
    memset(net->outputN, 0, net->outputWidth*sizeof(double));
    memset(net->hiddenN, 0, net->hiddenWidth*net->hiddenLength*sizeof(double));
    
    for(int i=0; i<net->inputWidth; i++){
        net->inputN[i] = inputs[i];
        for(int j=0; j<net->hiddenWidth; j++){
            net->hiddenR[j] += net->inputN[i]*net->inputW[i*net->inputWidth+j];
        }
    }
    for(int j=0; j<net->hiddenWidth; j++){
        net->hiddenR[j] += net->hiddenB[j];
        net->hiddenN[j] = ReLU(net->hiddenR[j]);
    }
    for(int i=1; i<net->hiddenLength; i++){
        for(int j=0; j<net->hiddenWidth; j++){
            for(int k=0; k<net->hiddenWidth;k++){
                net->hiddenR[i*net->hiddenWidth+j] += net->hiddenN[(i-1)*net->hiddenWidth+k]*net->hiddenW[(i-1)*net->hiddenWidth*net->hiddenWidth+(j*net->hiddenWidth)+(k)]; //good luck trying to decode this lmao
            }    
        }
        for(int j=0; j<net->hiddenWidth; j++){
            net->hiddenR[i*net->hiddenWidth+j] += net->hiddenB[i*net->hiddenWidth+j];
            net->hiddenN[i*net->hiddenWidth+j] = ReLU(net->hiddenR[i*net->hiddenWidth+j]);
        }
    }
    for(int i=0; i<net->hiddenWidth; i++){
        for(int j=0; j<net->outputWidth; j++){
            net->outputR[j] += net->hiddenN[(net->hiddenLength-1)*net->hiddenWidth+i]*net->toOutputW[i*net->outputWidth+j];
        }
    }
    for(int i=0; i<net->outputWidth; i++){
        net->outputR[i] += net->outputB[i];
        net->outputN[i] = sigmoid(net->outputR[i]);
    }
}

//backpropogate over examples
int gradientDescent(struct network *net, double inputs[], double outputs[], int examples, double trainingSpeed)
{
    int errorCode = 0;
    //allocate memory to gradient arrays
    double *inputWGradient = (double*)callocTrack(net->inputWidth*net->hiddenWidth, sizeof(double), &errorCode, &net->bytesInUse);
    double *hiddenWGradient = (double*)callocTrack(net->hiddenWidth*net->hiddenWidth*net->hiddenLength, sizeof(double), &errorCode, &net->bytesInUse);
    double *toWGradient = (double*)callocTrack(net->hiddenWidth*net->outputWidth, sizeof(double), &errorCode, &net->bytesInUse);
    double *hiddenBGradient = (double*)callocTrack(net->hiddenWidth*net->hiddenLength, sizeof(double), &errorCode, &net->bytesInUse);
    double *outputBGradient = (double*)callocTrack(net->outputWidth, sizeof(double), &errorCode, &net->bytesInUse);
    double *costHidden = (double*)callocTrack(net->hiddenWidth*net->hiddenLength, sizeof(double), &errorCode, &net->bytesInUse);
    double *costOut = (double*)callocTrack(net->outputWidth, sizeof(double), &errorCode, &net->bytesInUse);
    //return code 1 if allocation fails.
    net->error = 0;
    for(int example = 0; example < examples; example++)
    {
        double currentInputs[net->inputWidth];
        for(int i=0; i<net->inputWidth; i++){
            currentInputs[i] = inputs[example*net->inputWidth+i];
        }
        propogate(net, currentInputs);
        //calculate loss
        for(int i=0; i<net->outputWidth; i++){
            costOut[i] = (outputs[example*net->outputWidth+i]-net->outputN[i]);
            net->error += fabs(costOut[i])/examples/net->outputWidth;
        }
        for(int i=(net->hiddenLength-1); i>=0; i--){
            for(int j=0; j<net->hiddenWidth; j++){
                if(i<net->hiddenLength-2)
                    for(int k=0; k<net->hiddenWidth; k++)
                        costHidden[i*net->hiddenWidth+j] += costHidden[(i+1)*net->hiddenWidth+k] * net->hiddenW[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k];
                else
                    for(int k=0; k<net->outputWidth; k++)
                        costHidden[i*net->hiddenWidth+j] += costOut[k] * net->toOutputW[j*net->outputWidth+k];
            }
        }
        for(int i=0; i<net->inputWidth; i++){
            for(int j=0; j<net->hiddenWidth; j++){
            }
        }
        
        //update biases and weights n stuff

        for(int i=0; i<net->inputWidth; i++){
            for(int j=0; j<net->hiddenWidth; j++){
                inputWGradient[i] += ( net->inputN[i] * (costHidden[j]) );
            }
        }
        for(int i=0; i<net->hiddenLength; i++){
            for(int j=0; j<net->hiddenWidth; j++){
                if(i < net->hiddenLength-1){
                    for(int k=0; k<net->hiddenWidth; k++)
                        hiddenWGradient[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k] += ( net->hiddenN[i*net->hiddenWidth+j] * costHidden[(i+1)*net->hiddenWidth+k] * ReLUDer(net->hiddenR[(i+1)*net->hiddenWidth+k])); 
                }else{
                    for(int k=0; k<net->outputWidth; k++)
                        toWGradient[(j*net->outputWidth)+k] += ( net->hiddenN[i*net->hiddenWidth+j] * costOut[k] * sigmoidDer(net->outputR[k]) ); 
                }
                hiddenBGradient[i*net->hiddenWidth+j] += ( ReLUDer(net->hiddenR[i*net->hiddenWidth+j]) * costHidden[i*net->hiddenWidth+j] );

            }
        }
        for(int i=0; i<net->outputWidth; i++){
            outputBGradient[i] += ( sigmoidDer(net->outputR[i]) * costOut[i] );
        }
        memset(costOut, 0, net->outputWidth*sizeof(double));
        memset(costHidden, 0, net->hiddenWidth*net->hiddenLength*sizeof(double));
    }
    
    for(int i=0; i<net->inputWidth; i++){
        for(int j=0; j<net->hiddenWidth; j++){
            net->inputW[i*net->hiddenWidth+j] += trainingSpeed * inputWGradient[i*net->hiddenWidth+j];
            if(UseRegularization) net->inputW[i*net->hiddenWidth+j] = ParameterSquish(net->inputW[i*net->hiddenWidth+j]);
        }
    }
    for(int i=0; i<net->hiddenLength; i++){
        for(int j=0; j<net->hiddenWidth; j++){
            if(i < net->hiddenLength-1)
                for(int k=0; k<net->hiddenWidth; k++){
                        net->hiddenW[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k] += trainingSpeed * hiddenWGradient[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k];
                        if(UseRegularization) net->hiddenW[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k] = ParameterSquish(net->hiddenW[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k]);
                }else{
                    for(int k=0; k<net->outputWidth; k++){
                        net->toOutputW[(j*net->outputWidth)+k] += trainingSpeed * toWGradient[(j*net->outputWidth)+k]; 
                    }
                }
                net->hiddenB[i*net->hiddenWidth+j] += trainingSpeed * hiddenBGradient[i*net->hiddenWidth+j];
                if(UseRegularization) net->hiddenB[i*net->hiddenWidth+j] = ParameterSquish(net->hiddenB[i*net->hiddenWidth+j]);
        }
    }
    for(int i=0; i<net->outputWidth; i++){
        net->outputB[i] += trainingSpeed * outputBGradient[i];
    }
    freeTrack(net->hiddenWidth*net->hiddenLength, sizeof(double), hiddenBGradient, &net->bytesInUse);
    freeTrack(net->hiddenWidth*net->hiddenWidth*net->hiddenLength, sizeof(double), hiddenWGradient, &net->bytesInUse);
    freeTrack(net->inputWidth*net->hiddenWidth, sizeof(double), inputWGradient, &net->bytesInUse);
    freeTrack(net->outputWidth, sizeof(double), outputBGradient, &net->bytesInUse);
    freeTrack(net->hiddenWidth*net->outputWidth, sizeof(double), toWGradient, &net->bytesInUse);
    freeTrack(net->hiddenWidth*net->hiddenLength, sizeof(double), costHidden, &net->bytesInUse);
    freeTrack(net->outputWidth, sizeof(double), costOut, &net->bytesInUse);
    return 0;
}