#include <stdio.h>
#include <time.h>
#include "network.h"
#pragma pack(1) 

int main()
{
    DEBUG_MODE = 0;
    struct network net;
    int length = 5;
    int width1 = 2;
    int width2 = 5;
    int width3 = 2;
    printf("enter desired input layer width\n");
    //scanf("%d", &width1);
    printf("enter desired number of hidden layers\n");
    //scanf("%d", &length);
    printf("enter desired hidden layer width\n");
    //scanf("%d", &width2);
    printf("enter desired output layer width\n");
    //scanf("%d", &width3);
    if(setupNetwork(&net, length, width1, width2, width3)==1){
        printf("Network memory allocation was unsuccesful.\npress enter to exit...\n");
        getchar();
        return 1;
    }
    printf("Network memory allocation was successful and setup..\n");
    printf("How small do you want the error to be?\n");
    double errorMargin;
    //scanf("%lf", &errorMargin);
    errorMargin=0.01;
    double trainingSpeed = 0.01;
    printf("How fast do you want your training speed to be? (recommended: 0.01 - 0.001)");
    //scanf("%lf", &trainingSpeed);
    printf("iterating...\n");
    printf("//BACKPROP////////\n\n");
    double inputs[] = { 1, 1, 0, 0, 1, 0 };
    double outputs[] = { 0, 1, 1, 1, 0, 0 };
    int i = 0;
    clock_t t;
    t = clock();
    while(i<=100000){
        int error;
        error = gradientDescent(&net, inputs, outputs, 3, trainingSpeed, 1);
        //if(i==100000)
            //DEBUG_MODE = 1;
        //if(i%100==0)
            //printf("error: %lf at iteration %d using %g bytes\n", net.error, i, (double)net.bytesInUse);
        if(error==0)
            i++;
        if(net.error < errorMargin)
            break;
    }
    t = clock() - t;
    printf("error final: %lf\n", net.error);
    freeNetwork(&net);
    printf("TIME: %lf\n", (double)((double)t/CLOCKS_PER_SEC*1000));
    printf("press enter to exit...\n");
    getchar();
    return 0;
}
