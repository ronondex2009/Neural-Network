#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

struct network
{
    int inputWidth;
    int outputWidth;
    int hiddenWidth;
    int hiddenLength;
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

//setup a new network array
int setupNetwork(struct network *networkstruct, int length, int widthIn, int widthHidden, int widthOut)
{
    networkstruct->inputWidth = widthIn;
    networkstruct->outputWidth = widthOut;
    networkstruct->hiddenWidth = widthHidden;
    networkstruct->hiddenLength = length;
    networkstruct->inputN = (double*)calloc(widthIn, sizeof(double));
    networkstruct->outputN = (double*)calloc(widthOut, sizeof(double));
    networkstruct->hiddenN = (double*)calloc(widthHidden*length, sizeof(double));
    networkstruct->outputB = (double*)calloc(widthOut, sizeof(double));
    networkstruct->hiddenB = (double*)calloc(widthHidden*length, sizeof(double));
    networkstruct->inputR = (double*)calloc(widthIn, sizeof(double));
    networkstruct->outputR = (double*)calloc(widthOut, sizeof(double));
    networkstruct->hiddenR = (double*)calloc(widthHidden*length, sizeof(double));
    networkstruct->inputW = (double*)calloc(widthIn*widthHidden, sizeof(double));
    networkstruct->hiddenW = (double*)calloc(widthHidden*widthHidden*length, sizeof(double));
    networkstruct->toOutputW = (double*)calloc(widthHidden*widthOut, sizeof(double));
    if( !networkstruct->inputN ||
        !networkstruct->outputN ||
        !networkstruct->hiddenN ||
        !networkstruct->inputR ||
        !networkstruct->outputR ||
        !networkstruct->hiddenR ||
        !networkstruct->hiddenB ||
        !networkstruct->outputB ||
        !networkstruct->inputW ||
        !networkstruct->hiddenW ||
        !networkstruct->toOutputW )
            return 1;
    
    for(int j = 0; j < widthIn; j++)
    {
        for(int k = 0; k < widthHidden; k++)
        {
            networkstruct->inputW[j*widthHidden+k] = (double)(rand() % 100)/100-0.5;
        }
    }
    for(int i = 0; i < length; i++)
    {
        for(int j = 0; j < widthHidden; j++)
        {
            for(int k = 0; k < widthHidden; k++)
            {
                networkstruct->hiddenW[(i*widthHidden*widthHidden)+(j*widthHidden)+k] = (double)(rand() % 100)/100-0.5;
            }
            networkstruct->hiddenB[i*widthHidden+j] = (double)(rand() % 100)/100-0.5;
        }
    }
    for(int j = 0; j < widthOut; j++)
    {
        networkstruct->outputB[j] = (double)(rand() % 100)/100-0.5;
        for(int i=0; i < widthHidden; i++)
        {
            networkstruct->toOutputW[i*widthOut+j] = (double)(rand() % 100)/100-0.5;
        }
    }
    return 0;
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
}

double sigmoid(double x)
{
    double result = 1/(1+exp(-x));
    if(!isfinite(result))
        return 0;
    return result;
}
double sigmoidDer(double x)
{
    double result = sigmoid(x)*(1-sigmoid(x));
    if(!isfinite(result))
        return 1;
    return result;
}
double ReLU(double x)
{
    double result = (double)(x>=0) ? x: x;
    if(!isfinite(result)) //ReLU is not used in the code because it causes exploding gradients.
        return 1;           //feel free to use ReLU in a fork if you wish; but I have not included it in my own network.
    return result;
}
double ReLUDer(double x)
{
    double result = (double)(x>=0) ? 1: 1;
    return result;
}

//propogate the network
void propogate(struct network net, double inputs[])
{    
    memset(net.inputR, 0, net.inputWidth*sizeof(double));
    memset(net.outputR, 0, net.outputWidth*sizeof(double));
    memset(net.hiddenR, 0, net.hiddenWidth*net.hiddenLength*sizeof(double));
    memset(net.inputN, 0, net.inputWidth*sizeof(double));
    memset(net.outputN, 0, net.outputWidth*sizeof(double));
    memset(net.hiddenN, 0, net.hiddenWidth*net.hiddenLength*sizeof(double));
    
    for(int i=0; i<net.inputWidth; i++){
        net.inputN[i] = inputs[i];
        for(int j=0; j<net.hiddenWidth; j++){
            net.hiddenR[j] += net.inputN[i]*net.inputW[i*net.inputWidth+j];
        }
    }
    for(int j=0; j<net.hiddenWidth; j++){
        net.hiddenR[j] += net.hiddenB[j];
        net.hiddenN[j] = sigmoid(net.hiddenR[j]);
    }
    for(int i=1; i<net.hiddenLength; i++){
        for(int j=0; j<net.hiddenWidth; j++){
            for(int k=0; k<net.hiddenWidth;k++){
                net.hiddenR[i*net.hiddenWidth+j] += net.hiddenN[(i-1)*net.hiddenWidth+k]*net.hiddenW[(i-1)*net.hiddenWidth*net.hiddenWidth+(j*net.hiddenWidth)+(k)]; //good luck trying to decode this lmao
            }    
        }
        for(int j=0; j<net.hiddenWidth; j++){
            net.hiddenR[i*net.hiddenWidth+j] += net.hiddenB[i*net.hiddenWidth+j];
            net.hiddenN[i*net.hiddenWidth+j] = sigmoid(net.hiddenR[i*net.hiddenWidth+j]);
        }
    }
    for(int i=0; i<net.hiddenWidth; i++){
        for(int j=0; j<net.outputWidth; j++){
            net.outputR[j] += net.hiddenN[(net.hiddenLength-1)*net.hiddenWidth+i]*net.toOutputW[i*net.outputWidth+j];
        }
    }
    for(int i=0; i<net.outputWidth; i++){
        net.outputR[i] += net.outputB[i];
        net.outputN[i] = sigmoid(net.outputR[i]);
    }
}

//backpropogate over examples
double gradientDescent(struct network net, double inputs[], double outputs[], int examples, double trainingSpeed)
{
    double *inputWGradient = (double*)calloc(net.inputWidth, sizeof(double));
    double *hiddenWGradient = (double*)calloc(net.hiddenWidth*net.hiddenWidth*net.hiddenLength, sizeof(double));
    double *toWGradient = (double*)calloc(net.hiddenWidth*net.outputWidth, sizeof(double));
    double *hiddenBGradient = (double*)calloc(net.hiddenWidth*net.hiddenLength, sizeof(double));
    double *outputBGradient = (double*)calloc(net.outputWidth, sizeof(double));
    double *costHidden = (double*)calloc(net.hiddenWidth*net.hiddenLength, sizeof(double));
    double *costOut = (double*)calloc(net.outputWidth, sizeof(double));
    net.error = 0;
    for(int example = 0; example < examples; example++)
    {
        double currentInputs[net.inputWidth];
        for(int i=0; i<net.inputWidth; i++){
            currentInputs[i] = inputs[example*net.inputWidth+i];
        }
        propogate(net, currentInputs);
        //calculate loss
        for(int i=0; i<net.outputWidth; i++){
            costOut[i] = (outputs[example*net.outputWidth+i]-net.outputN[i]);
            net.error += fabs(costOut[i])/examples/net.outputWidth;
            printf("%lf\n", net.error);
        }
        for(int i=(net.hiddenLength-1); i>=0; i--){
            for(int j=0; j<net.hiddenWidth; j++){
                if(i<net.hiddenLength-2)
                    for(int k=0; k<net.hiddenWidth; k++)
                        costHidden[i*net.hiddenWidth+j] += costHidden[(i+1)*net.hiddenWidth+k] * net.hiddenW[(i*net.hiddenWidth*net.hiddenWidth)+(j*net.hiddenWidth)+k];
                else
                    for(int k=0; k<net.outputWidth; k++)
                        costHidden[i*net.hiddenWidth+j] += costOut[k] * net.toOutputW[j*net.outputWidth+k];
            }
        }
        for(int i=0; i<net.inputWidth; i++){
            for(int j=0; j<net.hiddenWidth; j++){
            }
        }
        
        //update biases and weights n stuff

        for(int i=0; i<net.inputWidth; i++){
            for(int j=0; j<net.hiddenWidth; j++){
                inputWGradient[i] += ( net.inputN[i] * (costHidden[j]) );
            }
        }
        for(int i=0; i<net.hiddenLength; i++){
            for(int j=0; j<net.hiddenWidth; j++){
                if(i < net.hiddenLength-1){
                    for(int k=0; k<net.hiddenWidth; k++)
                        hiddenWGradient[(i*net.hiddenWidth*net.hiddenWidth)+(j*net.hiddenWidth)+k] += ( net.hiddenN[i*net.hiddenWidth+j] * costHidden[(i+1)*net.hiddenWidth+k] * sigmoidDer(net.hiddenR[(i+1)*net.hiddenWidth+k])); 
                }else{
                    for(int k=0; k<net.outputWidth; k++)
                        toWGradient[(j*net.outputWidth)+k] += ( net.hiddenN[i*net.hiddenWidth+j] * costOut[k] * sigmoidDer(net.outputR[k]) ); 
                }
                hiddenBGradient[i*net.hiddenWidth+j] += ( sigmoidDer(net.hiddenR[i*net.hiddenWidth+j]) * costHidden[i*net.hiddenWidth+j] );

            }
        }
        for(int i=0; i<net.outputWidth; i++){
            outputBGradient[i] += ( sigmoidDer(net.outputR[i]) * costOut[i] );
        }
        memset(costOut, 0, net.outputWidth*sizeof(double));
        memset(costHidden, 0, net.hiddenWidth*net.hiddenLength*sizeof(double));
    }
    
    for(int i=0; i<net.inputWidth; i++){
        for(int j=0; j<net.hiddenWidth; j++){
            net.inputW[i*net.hiddenWidth+j] += trainingSpeed * inputWGradient[i*net.hiddenWidth+j];
        }
    }
    for(int i=0; i<net.hiddenLength; i++){
        for(int j=0; j<net.hiddenWidth; j++){
            if(i < net.hiddenLength-1)
                for(int k=0; k<net.hiddenWidth; k++){
                        net.hiddenW[(i*net.hiddenWidth*net.hiddenWidth)+(j*net.hiddenWidth)+k] += trainingSpeed * hiddenWGradient[(i*net.hiddenWidth*net.hiddenWidth)+(j*net.hiddenWidth)+k];
                }else{
                    for(int k=0; k<net.outputWidth; k++)
                        net.toOutputW[(j*net.outputWidth)+k] += trainingSpeed * toWGradient[(j*net.outputWidth)+k]; 
                }
                net.hiddenB[i*net.hiddenWidth+j] += trainingSpeed * hiddenBGradient[i*net.hiddenWidth+j];
        }
    }
    for(int i=0; i<net.outputWidth; i++){
        net.outputB[i] += trainingSpeed * outputBGradient[i];
    }
    free(hiddenBGradient);
    free(hiddenWGradient);
    free(inputWGradient);
    free(outputBGradient);
    free(toWGradient);
    free(costHidden);
    free(costOut);
    return net.error;
}


