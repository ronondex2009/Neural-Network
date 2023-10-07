#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "network.h"

//Example program; was used for debugging the network!

int main()
{
    struct network net;
    int length = 7;
    int width1 = 2;
    int width2 = 10;
    int width3 = 2;
    printf("enter desired input layer width (SKIPPED: fixed training set)\n");
    //scanf("%d", &width1); We dont want to change this because of the training set input size
    printf("enter desired number of hidden layers\n");
    scanf("%d", &length);
    printf("enter desired hidden layer width\n");
    scanf("%d", &width2);
    printf("enter desired output layer width (SKIPPED: fixed training set)\n");
    //scanf("%d", &width3); We dont want to change this because of the training set output size
    srand(1255642); //Seed for the network
    if(setupNetwork(&net, length, width1, width2, width3)==1){
        printf("Network memory allocation was unsuccesful.\npress enter to exit...\n");
        getchar();
        return 1;
    }
    printf("Network memory allocation was successful and setup..\n");
    printf("How small do you want the error to be?\n");
    double errorMargin;
    scanf("%lf", &errorMargin);
    errorMargin=0.0001;
    double trainingSpeed = 0.001;
    printf("How fast do you want your training speed to be? (recommended: 0.01 - 0.001)");
    scanf("%lf", &trainingSpeed);
    printf("iterating...\n");
    printf("//BACKPROP////////\n\n");
    double inputs[] = { 1, 1, 0, 0, 1, 0 };
    double outputs[] = { 0, 1, 1, 1, 0, 0 };
    int i = 0;
    clock_t t;
    t = clock();
    while(i<=1000001){
        int error;
        error = gradientDescent(&net, inputs, outputs, 3, trainingSpeed, 0.1);
        if(i==1000000)
            DEBUG_MODE = 1;
        printf("error: %lf at iteration %d using %g bytes\n", net.error, i, (double)net.bytesInUse);
        if(error==0)
            i++;
        if(net.error < errorMargin)
            break;
    }
    t = clock() - t;
    printf("error final: %lf\n", net.error);
    freeNetwork(&net);
    printf("TIME: %.0lf milliseconds\n", (double)((double)t/CLOCKS_PER_SEC*1000));
    printf("press enter to exit...\n");
    scanf("%lf");
    return 0;
}
