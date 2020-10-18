#include <string>
#include <vector>


using namespace std;

#pragma once
class NeuroNet
{
public:
	NeuroNet() {};
	~NeuroNet() {};
	// производная сигмоида
	double sigmoid_derivative(double x);
	//устанавливает веса без файла
	void setLayers(int n, std::vector<int> &p);
	//ставит эти значения на вход нейронной сети
	void set_input(std::vector<double> &p);
	double ForwardFeed();
	void BackPropogation(double prediction, double rresult, double lr);
	// сохранение весов
	bool SaveWeights();
	bool ReadWeights();

public:
	int amount_layers; //количество слоев
	double **neurons_values; //нейроны
	double **neurons_errors;
	double ***weights; //веса
	int *size; //количество нейронов в каждом слое
};

