
#include "mlp.h"
#include <iostream>
#include <stdlib.h>
#include <math.h>

using namespace std;

Neuron::Neuron(int connections){

    weight.resize(connections);
    input.resize(connections);
    delta.resize(connections);
    oldDelta.resize(connections);

    for(int i = 0; i < connections; i++){
        weight[i] = (float)rand() / (float)RAND_MAX;// rand
    }
    bias = (float)rand() / (float)RAND_MAX; //static_cast <float> (rand()) / static_cast <float> (RAND_MAX);//rand
    //output = 0;
}
Neuron::~Neuron(){
}
//*******************************************************************************

Layer::Layer(int nNeuron,int nConnections){

    for(int i = 0; i < nNeuron; i++){
        neuron.push_back(Neuron(nConnections));
    }

}
Layer::~Layer(){
}
//*******************************************************************************

MLP::MLP(int input, int output, vector<int> hidden, float learningRate, float momentum){

    this->learningRate = learningRate;
    this->momentum = momentum;

    layer.push_back(Layer(hidden[0],input));

    for(int i =1; i < hidden.size(); i++)
        layer.push_back(Layer(hidden[i],hidden[i-1]));

    layer.push_back(Layer(output,hidden[hidden.size()-1]));
}
MLP::~MLP(){
}
//*******************************************************************************

void MLP::training(vector<vector<float> > inputT, vector<vector<float> > outputT, int epocas, vector<vector<float> > inputV, vector<vector<float> > outputV){

    float errorT, errorAntT, sumT, errorAntV, sumV;
    float errorUp = 0;

    for(int i = 0; i < epocas; i++){

        //Para cada entrada
        sumT = 0;
        sumV = 0;
        for(int i = 0; i < inputT.size(); i++){

            for(int j = 0; j < layer[0].neuron.size(); j++)
                for(int k = 0; k < layer[0].neuron[j].input.size(); k++)
                    layer[0].neuron[j].input[k] = inputT[i][k];

            feedForward(i, outputT);
            errorT = layer[layer.size()-1].neuron[0].error;
            sumT += pow(errorT,2);
            feedBack(i, outputT);
        }
        errorT = sumT / inputT.size();


        //Validação
        for(int i = 0; i < inputV.size(); i++){

            for(int j = 0; j < layer[0].neuron.size(); j++)
                for(int k = 0; k < layer[0].neuron[j].input.size(); k++)
                    layer[0].neuron[j].input[k] = inputV[i][k];

            feedForward(i, outputV);
            errorV = layer[layer.size()-1].neuron[0].error;
            sumV += pow(errorV,2);
        }
        errorV = sumV / inputV.size();

        if(i <= 10){
            errorAntV = errorV;
            errorAntT = errorT;
        }
        if(errorAntV < errorV || errorAntT < errorT){
            errorUp += 1;
            if(errorUp >= 100){
                cout <<"Epocas: " << i << " Erro Treinamento: " << errorT << " Erro Validação: " << errorV << " Maior Erro" <<endl;
                //salvar rede
                break;
            }
        }else
            errorUp = 0;

        errorAntV = errorV;
        errorAntT = errorT;

        cout <<"Epocas: " << i << " Erro Treinamento: " << errorT << " Erro Validação: " << errorV << " Erro Acima " << errorUp << endl;
    }
}
//*******************************************************************************

void MLP::test(vector<vector<float> > input, vector<vector<float> > &output){

    output.resize(input.size());
    for(int i = 0; i < output.size(); i++)
        output[i].resize(layer[layer.size()-1].neuron.size());

    //Para cada entrada
    for(int i = 0; i < input.size(); i++){

        for(int j = 0; j < layer[0].neuron.size(); j++)
            for(int k = 0; k < layer[0].neuron[j].input.size(); k++){
                layer[0].neuron[j].input[k] = input[i][k];
            }
        feedForward(i, output);
        //output[i].resize(layer[layer.size()-1].neuron.size());
        for(int j = 0 ; j < layer[layer.size()-1].neuron.size();j++){
            output[i][j] = layer[layer.size()-1].neuron[j].output;
        }
    }
}

//*******************************************************************************
void MLP::feedForward(int input, vector<vector<float> > output){

    //local_field_induced = vj

    //Primeira Camada Oculta
    for(int j = 0; j < layer[0].neuron.size(); j++){
        //layer[0].neuron[j].input = input[j];
        layer[0].neuron[j].vj = 0;
        for(int k = 0; k < layer[0].neuron[j].weight.size(); k++){
            layer[0].neuron[j].vj+= layer[0].neuron[j].weight[k]*layer[0].neuron[j].input[k];//input[i][k];
        }
        layer[0].neuron[j].vj += layer[0].neuron[j].bias;
        layer[0].neuron[j].output = sigmoidal(layer[0].neuron[j].vj);
    }

    //Outras Camadas
    for(int l = 1;l < layer.size();l++){
        for(int j = 0;j < layer[l].neuron.size();j++){
            layer[l].neuron[j].vj = 0;
            for(int k = 0; k < layer[l].neuron[j].weight.size(); k++){
                layer[l].neuron[j].vj+= layer[l].neuron[j].weight[k]*layer[l-1].neuron[k].output;
            }
            layer[l].neuron[j].vj += layer[l].neuron[j].bias;
            layer[l].neuron[j].output = sigmoidal(layer[l].neuron[j].vj);
        }

    }
    //Calcular o erro da camada de saida
    int lastLayer = layer.size()-1;
    for(int j = layer[lastLayer].neuron.size()-1; j >=0 ; j--){
        //layer[lastLayer].neuron[j].error = layer[lastLayer].neuron[j].output * (1-layer[lastLayer].neuron[j].output) * (output[input][j]-layer[lastLayer].neuron[j].output);
        layer[lastLayer].neuron[j].error = output[input][j] - layer[lastLayer].neuron[j].output;

        //Gradiente
        layer[lastLayer].neuron[j].gradient = layer[lastLayer].neuron[j].error * derivateSigmoidal(layer[lastLayer].neuron[j].output);
    }

    //Calcular o erro das camadas ocultas
    for(int i = lastLayer-1; i >= 0; i--){
        for(int j = layer[i].neuron.size()-1; j >=0; j--){
                layer[i].neuron[j].error = layer[i].neuron[j].output * (1-layer[i].neuron[j].output) * layer[lastLayer].neuron[0].output * layer[i].neuron[0].weight[j];
                //layer[i].neuron[j].error = layer[i].neuron[j].output * (1-layer[i].neuron[j].output) * layer[i+1].neuron[0].error * layer[i].neuron[0].weight[j];//layer[i+1].neuron[?].error

                //Gradiente
                float s = 0;
                for(int k = 0; k < (int)layer[i+1].neuron.size();k++){
                    s += layer[i+1].neuron[k].weight[j] * layer[i+1].neuron[k].gradient;
                }
                layer[i].neuron[j].gradient = s * derivateSigmoidal(layer[i].neuron[j].output);
        }
    }
}

//*******************************************************************************
void MLP::feedBack(int input, vector<vector<float> > output){

    int lastLayer = layer.size()-1;
    //recalcular os pesos e bias
    for(int i = lastLayer; i >= 1; i--){
        for(int j = 0; j < layer[i].neuron.size(); j++){
            for(int k = 0; k < layer[i].neuron[j].weight.size(); k++){
                //layer[i].neuron[j].weight[k] += learningRate * layer[i].neuron[j].error * layer[i-1].neuron[k].output * layer[i].neuron[j].gradient;

                //Livro Simon
                //layer[i].neuron[j].weight[k] += learningRate * (layer[i-1].neuron[k].output * layer[i].neuron[j].gradient * momentum);

                //Jefferson
                layer[i].neuron[j].delta[k] =  learningRate * (layer[i].neuron[j].gradient * layer[i-1].neuron[j].output);
                layer[i].neuron[j].weight[k] += layer[i].neuron[j].delta[k] + (layer[i].neuron[j].oldDelta[k] * momentum);
                layer[i].neuron[j].oldDelta[k] = layer[i].neuron[j].delta[k];

            }
            layer[i].neuron[j].bias += (learningRate * layer[i].neuron[j].gradient);// * layer[i].neuron[j].error;
        }
    }
    //recalcular os pesos e bias da entrada
    for(int j = 0; j < layer[0].neuron.size(); j++){
        //cout << layer[0].neuron[j].weight.size() <<endl;
        for(int k = 0; k < layer[0].neuron[j].weight.size(); k++){
            //layer[0].neuron[j].weight[k] += learningRate * layer[0].neuron[j].error * layer[0].neuron[j].input[k];//input[0][k];

            //Livro Simon
            //layer[0].neuron[j].weight[k] += learningRate * (layer[0].neuron[j].input[k] * layer[0].neuron[j].gradient* momentum);

            //Jefferson
            layer[0].neuron[j].delta[k] =  learningRate * (layer[0].neuron[j].gradient * layer[0].neuron[j].input[k]);
            layer[0].neuron[j].weight[k] += layer[0].neuron[j].delta[k] + (layer[0].neuron[j].oldDelta[k] * momentum);
            layer[0].neuron[j].oldDelta[k] = layer[0].neuron[j].delta[k];

        }
        layer[0].neuron[j].bias += (learningRate * layer[0].neuron[j].gradient);//layer[0].neuron[j].error;
    }
}

//*******************************************************************************
float MLP::sigmoidal(float vj){
    return 1/(1 + exp(-vj));
}

float MLP::derivateSigmoidal(float vj){
    return vj * (1-vj);
}

//*******************************************************************************
float MLP::radial(float vj){

    //https://pt.wikipedia.org/wiki/Fun%C3%A7%C3%A3o_de_base_radial

    /*
     * Gaussiana
     * Multiquadrática
     * Quadrática inversa
     * Multiquadrática inversa
     * Spline poli-harmônica
     * Spline "chapa" fina
     */


    return 1/(1+(-vj));
}

//*******************************************************************************
void MLP::testMlp(){

    vector<float>in(3),out(1);

    vector<vector<float> > input(1);
    in[0] = 1;
    in[1] = 0;
    in[2] = 1;
    input[0] = in;

    vector<vector<float> > output(1);
    out[0] = 1;
    output[0] = out;

    vector<int>hidden(1);
    hidden[0] = 2;

    MLP mlp(3,1,hidden);

    mlp.layer[0].neuron[0].weight[0] = 0.2;
    mlp.layer[0].neuron[0].weight[1] = 0.4;
    mlp.layer[0].neuron[0].weight[2] = -0.5;
    mlp.layer[0].neuron[0].bias = -0.4;

    mlp.layer[0].neuron[1].weight[0] = -0.3;
    mlp.layer[0].neuron[1].weight[1] = 0.1;
    mlp.layer[0].neuron[1].weight[2] = 0.2;
    mlp.layer[0].neuron[1].bias = 0.2;

    mlp.layer[1].neuron[0].weight[0] = -0.3;
    mlp.layer[1].neuron[0].weight[1] = -0.2;
    mlp.layer[1].neuron[0].bias = 0.1;

    //mlp.feedForward(input, output);
    //mlp.feedBack(0, output);

    mlp.printStructure();


}
//*******************************************************************************
void MLP::printStructure(){

    cout << "\t\n Extrutura \n" << endl;
    cout << "Entrada: " << layer[0].neuron[0].weight.size()<< endl;
    cout << "Camadas Ocultas: " << layer.size()-1 <<endl;
    for(int i = 0; i < layer.size()-1; i++)
        cout << "Camada [" << i << "] " << " neuronios: " << layer[i].neuron.size() << endl;
    cout << "Saida: " << layer[layer.size()-1].neuron.size() << endl << endl;


    for(int i = 0; i < layer.size(); i++){
        cout << "Camada: " << i << " nº neuronio: " << layer[i].neuron.size() << endl;
        for(int j = 0; j < layer[i].neuron.size(); j++){
            cout << " Neuronio: " << j <<endl;
            for(int k = 0; k < layer[i].neuron[j].weight.size(); k++){
                cout << "  Peso [" << k << "]: " << layer[i].neuron[j].weight[k] << endl;
            }
            cout << "  bias: " << layer[i].neuron[j].bias << endl;
            cout << "  Campo Local Induzido: " << layer[i].neuron[j].vj << endl;
            cout << "  Gradiente: " << layer[i].neuron[j].gradient << endl;
            cout << "  saida: " << layer[i].neuron[j].output << endl;
            cout << "  Erro: " << layer[i].neuron[j].error << endl << endl;
        }
    }

}


//*******************************************************************************
