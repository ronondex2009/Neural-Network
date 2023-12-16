// THIS VERSION IS BROKEN AND DOES NOT FUNCTION PROPERLY !!!
// ONLY KEPT FOR ARCHIVE PURPOSES

#include "networkOld.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#pragma pack(1)

static const double max = 1000; //unused
static int DEBUG_MODE = 0.0;

double GaussianRandom(double mu, double sigma)
{
    double U1, U2, W, mult;
    static double X1, X2;
    static int call;
    if(call==1)
    {
        call = !call;
        if(DEBUG_MODE)
            printf("[GAUSSIAN_CALL]: %d\n[GAUSSIAN_SIGMA]: %0.50lf\n[GAUSSIAN_RESULT]: %0.50lf\n", !call, sigma, (mu + (sigma * X2)));
        return mu + (sigma * X2);
    }
    do
    {
        U1 = -1 + ((double) rand () / RAND_MAX) * 2;
        U2 = -1 + ((double) rand () / RAND_MAX) * 2;
        W = pow(U1, 2) + pow(U2, 2);
    } while(W >= 1 || W == 0);
    mult = sqrt((-2*log(W)) / W);
    X1 = U1 * mult;
    X2 = U2 * mult;
    call = !call;
    if(DEBUG_MODE)
        printf("[GAUSSIAN_CALL]: %d\n[GAUSSIAN_SIGMA]: %0.50lf\n[GAUSSIAN_RESULT]: %0.50lf\n", !call, sigma, (mu + (sigma * X1)));
    return mu + (sigma * X1);
}

enum NETWORK_PARAMETERS{
    dynamic_regularize, dynamic_trainspeed, backprop_dont_thread, setup_dont_initialize_parameters, backprop_dont_modify_parameters, propogate_dont_thread, never_thread
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

    double result = 0;
    if(!errorcode==0)
    {
        freeNetwork(networkstruct);
        return 1;
    }
    
    //standard initialization
    for(int j = 0; j < widthIn; j++)
    {
        for(int k = 0; k < widthHidden; k++)
        {
            result = GaussianRandom(0, sqrt(1.0/widthIn));
            networkstruct->inputW[j*widthHidden+k] = result;
        }
    }
    for(int i = 0; i < length; i++)
    {
        for(int j = 0; j < widthHidden; j++)
        {
            for(int k = 0; k < widthHidden; k++)
            {
                result = GaussianRandom(0, sqrt(1.0/widthIn));
                networkstruct->hiddenW[(i*widthHidden*widthHidden)+(j*widthHidden)+k] = result;
            }
            networkstruct->hiddenB[i*widthHidden+j] = 0.0;
        }
    }
    for(int j = 0; j < widthOut; j++)
    {
        networkstruct->outputB[j] = 0.0;
        for(int i=0; i < widthHidden; i++)
        {
            result = GaussianRandom(0, sqrt(2.0/widthIn));
            networkstruct->toOutputW[i*widthOut+j] = result; //this is not a mistake
        }
    }
    return 0;
}

double sigmoid(double x) //my second favorite function
{
    double result;
    result = 1/(1+exp(-x));
    return result;
}
double sigmoidDer(double x) //my first favorite function
{
    double result = sigmoid(x)*(1-sigmoid(x));
    if(!isfinite(result))
        return 0;
    return result;
}
double ReLU(double x) //why do people just not use linearity?
{
    double result = (double)(x>=0) ? x: x*0.01; //"leaky" ReLU
    if(!isfinite(result))
        return 1;
    return result;
}
double ReLUDer(double x) //why!?
{
    double result = (double)(x>=0) ? 1: 0.01;
    return result;
}

/*double paramClipping(double x) //DEPRECATED: not an effective solution to exploding gradients
{                              // was originally intended to fix NaN and Inf by clipping parameters.
    double result;             // this only broke training and didn't even fix the problem.
    result = max*((exp(x/max)-exp(-x/max))/(exp(x/max)+exp(-x/max)));
    if(isnan(result))          // cautionary tale: don't do this
        return (x>0) ? max: -max;
    if(DEBUG_MODE)
        printf("[PARAM_CLIPPING_OUTPUT]: %0.50lf\n", result);
    return result;
}*/
double sumOfWeightsIntoNeuron(struct network* net, int i, int j)
{
    double result = 0;
    if(i==0) //weights that go to first hidden layer.
        for(int k=0; k<net->inputWidth; k++)
            result += fabs(net->inputW[k*net->hiddenWidth+j]);
    if(i==-1) //THIS DENOTES THAT IT IS toOutputW WEIGHTS!!!!
        for(int k=0; k<net->hiddenWidth; k++)
            result += fabs(net->toOutputW[k*net->outputWidth+j]);
    if(i > 0)
        for(int k=0; k<net->hiddenWidth; k++)//{
            result += fabs(net->hiddenW[((i-1)*net->hiddenWidth*net->hiddenWidth)+(k*net->hiddenWidth)+j]);
    if(i==net->hiddenLength-1)
        for(int k=0; k<net->hiddenWidth; k++)//{
            result += fabs(net->toOutputW[j*net->outputWidth+k]);
    if(DEBUG_MODE)
        printf("[L2_REG_OUTPUT]: %0.50lf\n", result*result);
    return result;
}
double L2(struct network* net) //I think this is how it works..
{
    double result = 0;
    for(int i=0; i<net->hiddenLength; i++){
        for(int j=0; j<net->hiddenWidth; j++){
            result += sumOfWeightsIntoNeuron(net, i, j);
        }
    }
    for(int i=0; i<net->outputWidth; i++){
        result += sumOfWeightsIntoNeuron(net, net->hiddenLength, i);
    }
    return result*result;
}

//propogate the network
void propogate(struct network* net, double inputs[])
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
        if(DEBUG_MODE)
            printf("[PROPOGATE_HIDDEN_RAW_(FROMINPUT)] %d: %0.50lf\n[PROPOGATE_HIDDEN_NEURON_(FROMINPUT)] %d: %0.50lf\n", j, net->hiddenR[j], j, net->hiddenN[j]);
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
            if(DEBUG_MODE)
                printf("[PROPOGATE_HIDDEN_RAW] %d: %0.50lf\n[PROPOGATE_HIDDEN_NEURON] %d: %0.50lf\n", i*net->hiddenWidth+j, net->hiddenR[i*net->hiddenWidth+j], i*net->hiddenWidth+j, net->hiddenN[i*net->hiddenWidth+j]);
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
        if(DEBUG_MODE)
            printf("[PROPOGATE_OUTPUT_RAW] %d: %0.50lf\n[PROPOGATE_OUTPUT_NEURON] %d: %0.50lf\n", i, net->outputR[i], i, net->outputN[i]);
    }
}

//backpropogate over examples
int gradientDescent(struct network *net, double inputs[], double outputs[], int examples, double trainingSpeed, double regularization)
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
    double instances;
    instances = ((net->hiddenWidth*net->inputWidth)+(pow(net->hiddenWidth,2))+(net->hiddenWidth*net->outputWidth));
    double L2Val;
    L2Val = ((regularization/instances)*L2(net));
    double L2Der;
    L2Der = 2.0*sqrt(L2Val);
    //net->error += ((regularization/instances)*L2(net));
    //BACKPROPOGATE EXAMPLES
    for(int example = 0; example < examples; example++)
    {
        //FEEDFORWARD
        double currentInputs[net->inputWidth];
        for(int i=0; i<net->inputWidth; i++){
            currentInputs[i] = inputs[example*net->inputWidth+i];
            if(DEBUG_MODE)
                printf("[BACKPROP_SETINPUTS] %d: %0.50lf\n", i, currentInputs[i]);
        }
        propogate(net, currentInputs);
        
        //CALCULATE TOTAL GENERAL LOSS
        for(int i=0; i<net->outputWidth; i++){
            costOut[i] = 2*(outputs[example*net->outputWidth+i]-net->outputN[i]);
            net->error += pow(fabs(costOut[i]/2)/examples/net->outputWidth,2);
            if(DEBUG_MODE)
                printf("[BACKPROP_LOSS_OUTPUT_GENERAL] %d: %0.50lf\n", i, costOut[i]);
        }
        for(int i=(net->hiddenLength-1); i>=0; i--){
            for(int j=0; j<net->hiddenWidth; j++){
                if(i<net->hiddenLength-2){
                    for(int k=0; k<net->hiddenWidth; k++)
                        costHidden[i*net->hiddenWidth+j] += costHidden[(i+1)*net->hiddenWidth+k] * net->hiddenW[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k];
                    if(DEBUG_MODE)
                        printf("[BACKPROP_LOSS_HIDDEN_GENERAL] %d: %0.50lf\n", i*net->hiddenWidth+j, costHidden[i*net->hiddenWidth+j]);
                }else{
                    for(int k=0; k<net->outputWidth; k++)
                        costHidden[i*net->hiddenWidth+j] += costOut[k] * net->toOutputW[j*net->outputWidth+k];
                    if(DEBUG_MODE)
                        printf("[BACKPROP_LOSS_HIDDEN_GENERAL_(FROMOUTPUT)] %d: %0.50lf\n", i*net->hiddenWidth+j, costHidden[i*net->hiddenWidth+j]);
                }
            }
        }

        //CALCULATE TOTAL WEIGHT AND BIAS LOSS
        for(int i=0; i<net->inputWidth; i++){
            for(int j=0; j<net->hiddenWidth; j++){
                inputWGradient[i*net->hiddenWidth+j] += ( net->inputN[i] * (costHidden[j]) );
                inputWGradient[i*net->hiddenWidth+j] += (net->inputW[i*net->hiddenWidth+j]>0) 
                    ? net->inputW[i*net->hiddenWidth+j]*L2Der*(regularization/instances): net->inputW[i*net->hiddenWidth+j]*L2Der*(regularization/instances); //implementation of L2 Regularization
                if(DEBUG_MODE)
                    printf("[BACKPROP_LOSS_INPUT_WEIGHT] %d: %0.50lf\n", i, inputWGradient[i]);
            }
        }
        for(int i=0; i<net->hiddenLength; i++){
            for(int j=0; j<net->hiddenWidth; j++){
                if(i < net->hiddenLength-1){
                    for(int k=0; k<net->hiddenWidth; k++){
                        hiddenWGradient[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k] += ( net->hiddenN[i*net->hiddenWidth+j] * costHidden[(i+1)*net->hiddenWidth+k] 
                            * ReLUDer(net->hiddenR[(i+1)*net->hiddenWidth+k]));
                        if(DEBUG_MODE)
                            printf("[BACKPROP_LOSS_HIDDEN_WEIGHT] %d: %0.50lf\n", (i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k, hiddenWGradient[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k]);
                        hiddenWGradient[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k] += (net->hiddenW[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k]>0) 
                            ? net->hiddenW[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k]*L2Der*(regularization/instances):
                            -net->hiddenW[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k]*L2Der*(regularization/instances); //implementation of L2 Regularization
                    }
                }else{
                    for(int k=0; k<net->outputWidth; k++){
                        toWGradient[(j*net->outputWidth)+k] += ( net->hiddenN[i*net->hiddenWidth+j] * costOut[k] * sigmoidDer(net->outputR[k]) );
                        if(DEBUG_MODE)
                            printf("[BACKPROP_LOSS_TOOUTPUT_WEIGHT] %d: %0.50lf\n", (j*net->outputWidth)+k, toWGradient[(j*net->outputWidth)+k]);
                        toWGradient[(j*net->outputWidth)+k] = (net->toOutputW[(j*net->outputWidth)+k]>0) 
                            ? net->toOutputW[(j*net->outputWidth)+k]*L2Der*(regularization/instances): -net->toOutputW[(j*net->outputWidth)+k]*L2Der*(regularization/instances);
                    }
                }
                hiddenBGradient[i*net->hiddenWidth+j] += ( ReLUDer(net->hiddenR[i*net->hiddenWidth+j]) * costHidden[i*net->hiddenWidth+j] );
                if(DEBUG_MODE)
                    printf("[BACKPROP_LOSS_HIDDEN_BIAS] %d: %0.50lf\n", i*net->hiddenWidth+j, hiddenBGradient[i*net->hiddenWidth+j]);
            }
        }
        for(int i=0; i<net->outputWidth; i++){
            outputBGradient[i] += ( sigmoidDer(net->outputR[i]) * costOut[i] );
            if(DEBUG_MODE)
                printf("[BACKPROP_LOSS_OUTPUT_BIAS] %d: %0.50lf\n", i, outputBGradient[i]);
        }
        //CLEAR COST FOR NEXT EXAMPLE; WEIGHT AND BIAS GRADIENT PRESERVED FOR UPDATING NETWORK
        memset(costOut, 0, net->outputWidth*sizeof(double));
        memset(costHidden, 0, net->hiddenWidth*net->hiddenLength*sizeof(double));
    }
    
    double normalization_value = 1;

    //NORMALIZATION VAL FOR GRADIENT CLIPPING
    for(int i=0; i<net->inputWidth; i++){
        for(int j=0; j<net->hiddenWidth; j++){
            normalization_value = (fabs(inputWGradient[i])>normalization_value) ? fabs(inputWGradient[i]): normalization_value;
        }
    }
    for(int i=0; i<net->hiddenLength; i++){
        for(int j=0; j<net->hiddenWidth; j++){
            if(i < (net->hiddenLength-1)){
                for(int k=0; k<net->hiddenWidth; k++)
                    normalization_value = (fabs(hiddenWGradient[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k])>normalization_value) 
                        ? fabs(hiddenWGradient[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k]): normalization_value;
            }else{
                for(int k=0; k<net->outputWidth; k++)
                    normalization_value = (fabs(toWGradient[(j*net->outputWidth)+k])>normalization_value) ? fabs(toWGradient[(j*net->outputWidth)+k]): normalization_value;
            }
            normalization_value = (fabs(hiddenBGradient[i*net->hiddenWidth+j])>normalization_value) ? fabs(hiddenBGradient[i*net->hiddenWidth+j]): normalization_value;
        }
    }
    for(int i=0; i<net->outputWidth; i++){
        normalization_value = (fabs(outputBGradient[i])>normalization_value) ? fabs(outputBGradient[i]): normalization_value;
    }
    if(DEBUG_MODE)
        printf("[BACKPROP_LOSS_CLIPPING]: %0.50lf\n", normalization_value);
    //GRADIENT CLIPPING BY NORM
    for(int i=0; i<net->inputWidth; i++){
        for(int j=0; j<net->hiddenWidth; j++){
            inputWGradient[i*net->hiddenWidth+j] /= normalization_value;
        }
    }
    for(int i=0; i<net->hiddenLength; i++){
        for(int j=0; j<net->hiddenWidth; j++){
            if(i < (net->hiddenLength-1)){
                for(int k=0; k<net->hiddenWidth; k++)
                    hiddenWGradient[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k] /= normalization_value;
            }else{
                for(int k=0; k<net->outputWidth; k++)
                    toWGradient[(j*net->outputWidth)+k] /= normalization_value;
            }
            hiddenBGradient[i*net->hiddenWidth+j] /= normalization_value;
        }
    }
    for(int i=0; i<net->outputWidth; i++){
        outputBGradient[i] /= normalization_value;
    }

    //UPDATE THE NETWORK
    for(int i=0; i<net->inputWidth; i++){
        for(int j=0; j<net->hiddenWidth; j++){
            net->inputW[i*net->hiddenWidth+j] += trainingSpeed * inputWGradient[i*net->hiddenWidth+j];
            //net->inputW[i*net->hiddenWidth+j] = paramClipping(net->inputW[i*net->hiddenWidth+j]);
        }
    }
    for(int i=0; i<net->hiddenLength; i++){
        for(int j=0; j<net->hiddenWidth; j++){
            if(i < (net->hiddenLength-1)){
                for(int k=0; k<net->hiddenWidth; k++){
                    net->hiddenW[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k] += trainingSpeed * hiddenWGradient[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k];
                    //net->hiddenW[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k] = paramClipping(net->hiddenW[(i*net->hiddenWidth*net->hiddenWidth)+(j*net->hiddenWidth)+k]);
                }
            }else{
                for(int k=0; k<net->outputWidth; k++){
                        net->toOutputW[(j*net->outputWidth)+k] += trainingSpeed * toWGradient[(j*net->outputWidth)+k]; 
                       //net->toOutputW[(j*net->outputWidth)+k] = paramClipping(net->toOutputW[(j*net->outputWidth)+k]);
                    }
            }
            net->hiddenB[i*net->hiddenWidth+j] += trainingSpeed * hiddenBGradient[i*net->hiddenWidth+j];
            //net->hiddenB[i*net->hiddenWidth+j] = paramClipping(net->hiddenB[i*net->hiddenWidth+j]);
        }
    }
    for(int i=0; i<net->outputWidth; i++){
        net->outputB[i] += trainingSpeed * outputBGradient[i];
        //net->outputB[i] = paramClipping(net->outputB[i]);
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