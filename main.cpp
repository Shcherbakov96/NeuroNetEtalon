#define _USE_MATH_DEFINES

#include "NeuroNet.h"
#include <random>
#include <time.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>


using namespace std;

std::vector<double> correct_answers;
std::vector< std::vector<double> > examples_data;

std::vector<int> amount_percs_in_layer;
int amount_layers;

void read_data()
{
	ifstream in;
	in.open("input.txt", std::ios_base::in);

	if (in)
	{
		int amount_examples = 0;
		int amount_data = 0;

		string str;
		in >> str;

		if (str == "NeuroNet")
		{
			in >> amount_layers;
			amount_percs_in_layer.resize(amount_layers);
		}
		in >> str;
		if (str == "Percs")
		{
			for (int i = 0; i < amount_layers; i++)
			{
				in >> amount_percs_in_layer[i];
			}
		}
		in >> str;
		if (str == "amount_tests")
		{
			in >> amount_examples;
			in >> amount_data;

			correct_answers.resize(amount_examples);
			examples_data.resize(amount_examples);

			for (int i = 0; i < amount_examples; i++)
			{
				examples_data[i].resize(amount_data);
				for (int j = 0; j < amount_data; j++)
				{
					in >> examples_data[i][j];
				}
				in >> correct_answers[i];
				if (fabs(correct_answers[i] < 1e-4)) correct_answers[i] = 0.0;
				correct_answers[i] += M_PI;
				correct_answers[i] /= M_PI;
				correct_answers[i] *= 0.5;
			}
		}

	}
	else
	{
		std::cout << "File was not found \n";
	}
}

int main()
{
	setlocale(LC_ALL, "Russian");
	read_data();

	NeuroNet nn;

	bool to_study = true;

	if (to_study)
	{
		nn.setLayers(amount_layers, amount_percs_in_layer);
		double error = 0.0;
		double eps = 0.1;
		int iepoch = 0;

		do
		{
			cout << "Epoch # " << iepoch;
			error = 0.0;
			for (int itest = 0; itest < correct_answers.size(); itest++)
			{
				nn.set_input(examples_data[itest]);
				double result = nn.ForwardFeed();
				error += fabs(result - correct_answers[itest]);

				if (result != correct_answers[itest])
				{
					//cout << "Right answer is " << correct_answers[itest] << " NN predicted " << result << endl;
					nn.BackPropogation(result, correct_answers[itest], 0.05);
				}
			}

			printf("\rEpoch= %d \tError= %3.6e\n", iepoch, error);
			iepoch++;

		} while (error > eps);

		if (nn.SaveWeights())
		{
			cout << "Веса сохранены!";
		}
	}
	else
	{
		nn.setLayers(amount_layers, amount_percs_in_layer);
		nn.ReadWeights();
		for (int itest = 0; itest < correct_answers.size(); itest++)
		{
			nn.set_input(examples_data[itest]);
			double result = nn.ForwardFeed();
			if (result != correct_answers[itest])
			{
				cout << "Right answer is " << correct_answers[itest] << " NN predicted " << result << endl;
				nn.BackPropogation(result, correct_answers[itest], 0.5);
			}
		}
	}

	return 0;
}