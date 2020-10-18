#include <string>
#include <vector>


using namespace std;

#pragma once
class NeuroNet
{
public:
	NeuroNet() {};
	~NeuroNet() {};
	// ����������� ��������
	double sigmoid_derivative(double x);
	//������������� ���� ��� �����
	void setLayers(int n, std::vector<int> &p);
	//������ ��� �������� �� ���� ��������� ����
	void set_input(std::vector<double> &p);
	double ForwardFeed();
	void BackPropogation(double prediction, double rresult, double lr);
	// ���������� �����
	bool SaveWeights();
	bool ReadWeights();

public:
	int amount_layers; //���������� �����
	double **neurons_values; //�������
	double **neurons_errors;
	double ***weights; //����
	int *size; //���������� �������� � ������ ����
};

