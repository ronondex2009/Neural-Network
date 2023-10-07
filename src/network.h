#ifndef MLP_NEURAL_NETWORK_RONO_H
#define MLP_NEURAL_NETWORK_RONO_H

//static const double max; <------ this value was used for paramClipping(); deprecated
double DEBUG_MODE;
double GaussianRandom(double, double);
enum NETWORK_PARAMETERS;
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
void* callocTrack(unsigned int, unsigned int, int*, unsigned int*);
void freeTrack(unsigned int, unsigned int, void*, unsigned int*);
void freeNetwork(struct network*);
int setupNetwork(struct network*, int, int, int, int);
double sigmoid(double);
double sigmoidDer(double);
double ReLU(double);
double ReLUDer(double);
//double paramClipping(double); DEPRECATED!!!
double sumOfWeightsIntoNeuron(struct network*, int, int);
double L2(struct network*);
void propogate(struct network*, double[]);
int gradientDescent(struct network*, double[], double[], int, double, double);

#endif