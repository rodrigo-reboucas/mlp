//#include <stdio.h>
//#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h> //isnan
#include "mlp.h"
//#include "genetic.h"

using namespace std;

void loadTXTFile(vector<vector<float> > &samples,string path){

    ifstream file;
    file.open(path);
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path<<endl;
        return;
    }

    int height, width;
    file>>height; file>>width;

    samples.resize(height);
    for(int i = 0; i < (int)samples.size();i++){
        samples[i].resize(width);
        for(int j = 0; j < (int)samples[i].size(); j++){
            file>>samples[i][j];
            if(isnan(samples[i][j]))
                samples[i][j] = 0.0;
        }

    }
    file.close();

}

void normalize(vector<vector<float> >&samples){


    vector<float>max(samples[0].size(),-11),min(samples[0].size(),233);

    for(int j = 0; j < (int)min.size(); j++){

        for(int i = 0; i < (int)samples.size(); i++){
            if(max[j] == -11 || max[j] < samples[i][j])
                max[j] = samples[i][j];

            if(min[j] == 233 || min[j] > samples[i][j])
                min[j] = samples[i][j];
        }
    }

    for(int j = 0; j < (int)min.size(); j++){

        if(min[j] > 0)
            min[j] =0;

        if(max[j] == 0)
            max[j] =1;

        for(int i = 0; i < (int)samples.size(); i++)
            samples[i][j] = (samples[i][j]-min[j])/(max[j]-min[j]);

    }
}

void saveTXTFile(vector<vector<float> > samples, string path, bool save_to_c){

    ofstream file;
    file.open(path);
    if(!file.is_open()){
        cout<<"Erro ao abrir arquivo: "<<path+".txt"<<endl;
        return;
    }
    if(save_to_c ==1)
        file<<samples.size()<<" "<<samples[0].size()<<endl;

    for(int i = 0; i < (int)samples.size(); i++){
        for(int j =0; j < (int)samples[i].size();j++){

            file<<samples[i][j]<<" ";
        }file<<endl;

    }
    file.close();
}

int main(){
    vector<vector<float> > inputTraining, outputTraining, inputValidate, outputValidate, inputTest, outputTest;
    vector<int>hidden(2);
    hidden[0] = 6;
    hidden[1] = 3;

    //Dados de Vinho
    loadTXTFile(inputTraining, "wine/trainingIn.txt");
    loadTXTFile(outputTraining,"wine/trainingOut.txt");
    loadTXTFile(inputValidate, "wine/validateIn.txt");
    loadTXTFile(outputValidate,"wine/validateOut.txt");
    loadTXTFile(inputTest, "wine/testIn.txt");

/*
    //Dados de Imóveis
    loadTXTFile(inputTraining, "data base/trainingIn.txt");
    loadTXTFile(outputTraining,"data base/trainingOut.txt");

    loadTXTFile(inputValidate, "data base/validateIn.txt");
    loadTXTFile(outputValidate,"data base/validateOut.txt");

    loadTXTFile(inputTest, "data base/testIn.txt");
*/

    MLP *mlp = new MLP(inputTraining[0].size(), outputTraining[0].size(), hidden, 0.9, 0.6);

    mlp->training(inputTraining, outputTraining, 1000, inputValidate,outputValidate);

    mlp->test(inputTest, outputTest);

    mlp->printStructure();

    //Dados de Imóveis
    saveTXTFile(outputTest, "wine/TestOutMLP.txt",1);

/*
    //Algoritmo Genetico

    //Inicia a população
    int startPopulation = 500;
    int winner = 1;
    int iWinner = 0;
    //float learningRate, momentum;
    vector <Chromossome> ag(startPopulation);

    srand((unsigned int)time((time_t*)NULL)); //mudar a semente aleatoria
    for(int i = 0; i < startPopulation; i++){

        cout << endl << " População " << i << endl << endl;

        ag[i].hidden.resize(rand() % 5 + 1);
        for(int j = 0; j < ag[i].hidden.size(); j++)
            ag[i].hidden[j]=rand() % 8 + 1;

        ag[i].learningRate = (float)rand() / (float)RAND_MAX;
        ag[i].momentum = (float)rand() / (float)RAND_MAX;

        MLP *mlp = new MLP(inputTraining[0].size(), outputTraining[0].size(), ag[i].hidden, ag[i].learningRate , ag[i].momentum);
        mlp->training(inputTraining, outputTraining, 1000, inputValidate,outputValidate);
        ag[i].cost = mlp->errorV;

        //Vencedor
        if(ag[i].cost < winner){
            winner = ag[i].cost;
            iWinner = i;
        }
        delete mlp;
    }

    MLP *mlp = new MLP(inputTraining[0].size(), outputTraining[0].size(), ag[iWinner].hidden, ag[iWinner].learningRate , ag[iWinner].momentum);
    mlp->training(inputTraining, outputTraining, 10000, inputValidate,outputValidate);

    mlp->test(inputTest, outputTest);
    mlp->printStructure();

    cout << "População " << iWinner << endl;
*/
    saveTXTFile(outputTest, "wine/TestOutMLP.txt",1);

    return 0;

}
