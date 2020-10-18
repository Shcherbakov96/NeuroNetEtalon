#include <fstream>
#include <iostream>
#include <random>
#include <time.h>
#include <thread>

#include "NeuroNet.h"

double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}
double NeuroNet::sigmoid_derivative(double x)
{
	return ((fabs(x - 1) < 1e-4) || (fabs(x) < 1e-4)) ? 0.0 : x*(1.0 - x);
}
//устанавливает веса без файла
void NeuroNet::setLayers(int n, std::vector<int> &p)
{
	srand(time(0));
	amount_layers = n;
	neurons_values = new double *[n];
	neurons_errors = new double *[n];
	weights = new double**[n - 1];
	size = new int[n];
	for (int i = 0; i < n; i++) {
		size[i] = p[i];
		neurons_values[i] = new double[p[i]];
		neurons_errors[i] = new double[p[i]];

		if (i < n - 1) {
			weights[i] = new double*[p[i]];
			for (int j = 0; j < p[i]; j++) {
				weights[i][j] = new double[p[i + 1]];
				for (int k = 0; k < p[i + 1]; k++) {
					weights[i][j][k] = ((rand() % 100)) * 0.01 / size[i];
				}
			}
		}
	}
}
//ставит эти значения на вход нейронной сети
void NeuroNet::set_input(std::vector<double> &p)
{
	for (int i = 0; i < size[0]; i++)
	{
		neurons_values[0][i] = p[i];
	}
}

double NeuroNet::ForwardFeed()
{
	for (int ilay = 1; ilay < amount_layers; ilay++)
	{
		for (int j = 0; j < size[ilay]; j++)
		{
			double sum = 0.0;

			for (int k = 0; k < size[ilay - 1]; k++)
			{
				sum += neurons_values[ilay - 1][k] * weights[ilay - 1][k][j];
			}

			neurons_values[ilay][j] = 1.0 / (1.0 + exp(-sum));

		}
	}
	if (size[amount_layers - 1] == 1)
	{
		return neurons_values[amount_layers - 1][0];
	}
	else
	{
		// проходит по последнему слою и находит максимальное значение 
		double max = 0;
		double prediction = 0;
		for (int i = 0; i < size[amount_layers - 1]; i++)
		{
			if (neurons_values[amount_layers - 1][i] > max)
			{
				max = neurons_values[amount_layers - 1][i];
				prediction = i;
			}
		}
		//узнаем какой нейрон выдал максимальное значение
		return max;
	}
}

void NeuroNet::BackPropogation(double prediction, double rresult, double lr) {

	for (int j = 0; j < size[amount_layers - 1]; j++)
	{
		neurons_errors[amount_layers - 1][j] = (rresult - neurons_values[amount_layers - 1][j]);
	}

	for (int i = amount_layers - 2; i > 0; i--)
	{
		for (int j = 0; j < size[i]; j++)
		{
			neurons_errors[i][j] = 0.0;
			for (int k = 0; k < size[i + 1]; k++)
			{
				neurons_errors[i][j] += neurons_errors[i + 1][k] * weights[i][j][k];
			}
		}
	}

	for (int i = 0; i < amount_layers - 1; i++)
	{
		for (int j = 0; j < size[i]; j++)
		{
			for (int k = 0; k < size[i + 1]; k++)
			{
				weights[i][j][k] += lr * neurons_errors[i + 1][k] * sigmoid_derivative(neurons_values[i + 1][k])* neurons_values[i][j];
			}
		}
	}
}
// сохранение весов
bool NeuroNet::SaveWeights() {
	ofstream fout;
	fout.open("weights.txt");
	for (int i = 0; i < amount_layers; i++) {
		if (i < amount_layers - 1) {
			for (int j = 0; j < size[i]; j++) {
				for (int k = 0; k < size[i + 1]; k++) {
					fout << weights[i][j][k] << " ";
				}
			}
		}
	}
	fout.close();
	return 1;
}
bool NeuroNet::ReadWeights()
{
	ifstream in;
	in.open("weights.txt", std::ios_base::in);

	for (int i = 0; i < amount_layers; i++) 
	{
		if (i < amount_layers - 1) 
		{
			for (int j = 0; j < size[i]; j++) 
			{
				for (int k = 0; k < size[i + 1]; k++) 
				{
					in >> weights[i][j][k];
				}
			}
		}
	}
	in.close();

	return true;
}