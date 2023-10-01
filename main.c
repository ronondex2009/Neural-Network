#include <stdio.h>
#include "network.c"
#pragma pack(1) 

int main()
{

    struct network net;
    int length;
    int width1;
    int width2;
    int width3;
    /*printf("enter desired input layer width\n");
    scanf("%d", &width1);
    printf("enter desired number of hidden layers\n");
    scanf("%d", &length);
    printf("enter desired hidden layer width\n");
    scanf("%d", &width2);
    printf("enter desired output layer width\n");
    scanf("%d", &width3);
    if(setupNetwork(&net, length, width1, width2, width3)==1){*/
    if(setupNetwork(&net, 1, 2, 10, 2)==1){
        printf("Network memory allocation was unsuccesful.\npress enter to exit...\n");
        getchar();
        return 1;
    }
    printf("Network memory allocation was successful and setup..\n");
    printf("How small do you want the error to be?\n");
    double errorMargin;
    //scanf("%lf", &errorMargin);
    errorMargin=0.01;
    printf("iterating...\n");
    printf("//BACKPROP////////\n\n");
    double inputs[] = { 1, 1, 0, 0, 1, 0 };
    double outputs[] = { 0, 1, 1, 1, 0, 0 };
    int i = 0;
    while(i<=100000){
        int error;
        error = gradientDescent(&net, inputs, outputs, 3, 0.01);
        if(i==1536)
            printf("hello\n");
        if(i%100==99)
            printf("error: %lf at iteration %d\n", net.error, i+1);
        if(error==0)
            i++;
        if(net.error < errorMargin)
            break;
    }
    printf("error final: %lf\n", net.error);
    freeNetwork(&net);
    printf("press enter to exit...\n");
    //getchar();
    return 0;
    
}