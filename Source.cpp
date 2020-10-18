#include <iostream>
using namespace std;

class neuron {
public:
	float weight = 0.5;
	float LastError;
	float correction;
	float actualResult;
	float ProcessInputData(float input) {
		return input * weight;
	}
	float RestoreInputData(float output) {
		return output / weight;
	}
	void Train(float input, float exprectedResult) {
		actualResult = input * weight;
		LastError = exprectedResult - actualResult;
		correction = (LastError / actualResult);
		weight += correction;
	}
};
int main() {
	float tmp1 = 1;
	float tmp2 = 75;
	neuron hi;
	cout << hi.ProcessInputData(tmp1) << endl;
	int i = 0;
	do {
		i++;
		hi.Train(tmp1, tmp2);
		cout << "Iteration" << " " << i << " " << "Error:" << " " << hi.LastError << endl;
	} while (hi.actualResult > tmp2 + 0.001 || hi.actualResult < tmp2 - 0.001);
	cin >> tmp1;
	cout << hi.ProcessInputData(tmp1);
	cin >> tmp1;
	cout << hi.ProcessInputData(tmp1);
	return 0;
}