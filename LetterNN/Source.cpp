// lorem ipsum dolor sit umet
// в проекте изменение
#include<fstream>
#include<thread>
#include<random>
#include<time.h>
#include<Windows.h>
#include<iostream>
using namespace std;

struct neuron {//нейрон
	double value;
	double error;
	void act() {//функция активации
		value = (1 / (1 + pow(2.71828, -value)));
	}
};

struct data_one {//данные для обучения
	double info[4096]; //"значения" изображение 64х64
	char rresult; //rightresult, нужно будет в обучении
};

class network {
public:
	int layers;//кол-во слоев
	neuron** neurons;//двумерный массив нейронов
	double*** weights;//веса нейронов([слой][номер нейрона][номер связи нейрона со следующим слоем])
	int* size;//кол-во нейронов в слое
	int threadsNum = 1;//кол-во потоков

	double sigm_proizvodnaya(double x) {
		if ((fabs(x - 1) < 1e-9) || (fabs(x) < 1e-9)) return 0.0;
		double res = x * (1.0 - x);
		return res;
	}

	void setLayersNotStudy(int n, int* p, string filename) {//если не нужно обучение
		ifstream fin;
		fin.open(filename);//открываем файл и считываем оттуда веса
		srand(time(0));
		layers = n;
		neurons = new neuron * [n];
		weights = new double** [n - 1];
		size = new int[n];
		for (int i = 0; i < n; i++) {
			size[i] = p[i];
			neurons[i] = new neuron[p[i]];
			if (i < n - 1) {
				weights[i] = new double* [p[i]];
				for (int j = 0; j < p[i]; j++) {
					weights[i][j] = new double[p[i + 1]];
					for (int k = 0; k < p[i + 1]; k++) {
						fin >> weights[i][j][k];
					}
				}
			}
		}
	}

	void setLayers(int n, int* p) {//если нужно обучение
		srand(time(0));
		layers = n;
		neurons = new neuron * [n];
		weights = new double** [n - 1];
		size = new int[n];
		for (int i = 0; i < n; i++) {
			size[i] = p[i];
			neurons[i] = new neuron[p[i]];
			if (i < n - 1) {
				weights[i] = new double* [p[i]];
				for (int j = 0; j < p[i]; j++) {
					weights[i][j] = new double[p[i + 1]];
					for (int k = 0; k < p[i + 1]; k++) {
						weights[i][j][k] = ((rand() % 100)) * 0.01 / size[i];//присваиваем рандомные веса
					}
				}
			}
		}
	}

	void set_input(double p[]) {//принимает входные значения для нейросети (от 0 до 255(оттенки цвета лежат в таком диапозоне, в нашем случае- серый)) и присываивает их нейрону
		for (int i = 0; i < size[0]; i++) {
			neurons[0][i].value = p[i];
		}
	}

	void LayersCleaner(int LayerNumber, int start, int stop) {//очищает слои (в идеале последние две переменные доолжны помогать при многопоточности)
		srand(time(0));
		for (int i = start; i < stop; i++) {
			neurons[LayerNumber][i].value = 0;
		}
	}

	void ForwardFeeder(int LayerNumber, int start, int stop) {//производит процесс ForwardFeed (разносидность нейросети, когда нейроны передают информацию от входа к выходу
		for (int j = start; j < stop; j++) {
			for (int k = 0; k < size[LayerNumber - 1]; k++) {
				neurons[LayerNumber][j].value += neurons[LayerNumber - 1][k].value * weights[LayerNumber - 1][k][j];
			}
			neurons[LayerNumber][j].act();
		}
	}

	double ForwardFeed() {//используется в обучении
		setlocale(LC_ALL, "ru");
		for (int i = 1; i < layers; i++) {
			if (threadsNum == 1) {
				thread th1([&]() {//очистка слоя
					LayersCleaner(i, 0, size[i]);
					});
				th1.join();
			}

			if (threadsNum == 1) {//"кормление" нейрона
				thread th1([&]() {
					ForwardFeeder(i, 0, size[i]);
					});
				th1.join();
			}

		}
		double max = 0;
		double prediction = 0;
		for (int i = 0; i < size[layers - 1]; i++) {//высчитывает "вероятность" буквы (т.е. с каким шансом рисунок- это та или иная буква)

			if (neurons[layers - 1][i].value > max) {
				max = neurons[layers - 1][i].value;
				prediction = i;
			}
		}
		return prediction;
	}

	double ForwardFeed(string param) {//используется, когда начинается тест, выводит "шансы" букв на экран, аналогична по сути предыдущей функции
		setlocale(LC_ALL, "ru");
		for (int i = 1; i < layers; i++) {
			if (threadsNum == 1) {
				thread th1([&]() {
					LayersCleaner(i, 0, size[i]);
					});
				th1.join();
			}

			if (threadsNum == 1) {
				thread th1([&]() {
					ForwardFeeder(i, 0, size[i]);
					});
				th1.join();
			}

		}
		double max = 0;
		double prediction = 0;
		for (int i = 0; i < size[layers - 1]; i++) {
			cout << char(i + 65) << " : " << neurons[layers - 1][i].value << endl;
			if (neurons[layers - 1][i].value > max) {
				max = neurons[layers - 1][i].value;
				prediction = i;
			}
		}
		return prediction;
	}

	void BackPropogation(double prediction, double rresult, double lr) {//функция для работы с ошибками нейросети во время обучения (или неверного ответа) по методу обратного распространения ошибки
		for (int i = layers - 1; i > 0; i--) {//все начинается с выходных нейронов, где происходит вычисление ошибки
			if (threadsNum == 1) {//должна была быть многопоточность... 
				if (i == layers - 1) {
					for (int j = 0; j < size[i]; j++) {
						if (j != int(rresult)) {
							neurons[i][j].error = -pow((neurons[i][j].value), 2);
						}
						else {
							neurons[i][j].error = pow(1.0 - neurons[i][j].value, 2);
						}

					}
				}
				else { //далее это значение идет обратно к скрытым нейронам, где идет суммирование входящих ошибок
					for (int j = 0; j < size[i]; j++) {
						double error = 0.0;
						for (int k = 0; k < size[i + 1]; k++) {
							error += neurons[i + 1][k].error * weights[i][j][k];
						}
						neurons[i][j].error = error;
					}
				}
			}
		}
		for (int i = 0; i < layers - 1; i++) {//каждый выходной нейрон меняет свои веса
			if (threadsNum == 1) {
				for (int j = 0; j < size[i]; j++) {
					for (int k = 0; k < size[i + 1]; k++) {
						weights[i][j][k] += lr * neurons[i + 1][k].error * sigm_proizvodnaya(neurons[i + 1][k].value) * neurons[i][j].value;//после вычисления ошибки происходит перевычисление весов
					}
				}
			}
		}
	}

	bool SaveWeights() {//сохранение новых весов (используется в обучении или ошибке)
		ofstream fout;
		fout.open("weights.txt");
		for (int i = 0; i < layers; i++) {
			if (i < layers - 1) {
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
};




int main() {

	srand(time(0));
	setlocale(LC_ALL, "Russian");
	ifstream fin;
	const int l = 4;//layers
	const int input_l = 4096;//изображение 64х64
	int size[l] = { input_l, 256, 64, 26 }; //4096 разделить на кол-во слоев в квадрате=256, а потом 256/4=64, а 26, т.к. букв в английском алфавите 26
	network nn;

	double input[input_l];//массив значений для нейронов
	char rresult; //right result
	double result; //результат, порлученный при обучении нейросети
	double ra = 0; //right answer
	int maxra = 0; //max right answer. максимальное кол-во угаданных букв
	int maxraepoch = 0;//нужно будет для отслежки последней эпохи, где угадано максимальное кол-во букв
	const int n = 83;
	bool to_study = 0;
	cout << "Производить обучение?";
	cin >> to_study;

	data_one* data = new data_one[n];//сюда будем "пихать" "учительские" данные

	if (to_study) {
		fin.open("lib.txt");//вот и "учитель"
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < input_l; j++) {
				fin >> data[i].info[j];//"пихаем"
			}
			fin >> data[i].rresult;
			data[i].rresult -= 65;//в конце есть буква, которая говорит, что за рисунок был считан- так нейросеть понмает, что ей только что "дали" и что выводить, когда "увидит" похожее. -65, т.к. в таблице ASCII у буквы А индекс 65
		}

		nn.setLayers(l, size);//устанавливаем слои и нужные веса
		for (int e = 0; ra / n * 100 < 100; e++) { //e- epoch(эпоха). По сути, этот цикл будет "идти", пока точность нейросети не будет 100% в угадывании "учительских "

			ra = 0; //каждый раз обнуляем rightanswer для новой эпохи обучения

			for (int i = 0; i < n; i++) {

				for (int j = 0; j < input_l; j++) {//принимаем входные значения для нейросети(все то, что ранее считали, "кладем" в нейроны)
					input[j] = data[i].info[j];
				}
				rresult = data[i].rresult;//принимаем в массив верные значения
				nn.set_input(input);
				result = nn.ForwardFeed();//"кормим" нейроны
				if (result == rresult) {//если "кормление" дало эталонный результат, то буква угадана
					cout << "Угадал букву " << char(rresult + 65) << endl;
					ra++;
				}
				else {
					nn.BackPropogation(result, rresult, 0.5);//если получилась ошибка, то происходит корректировка значений весов
				}
			}

			cout << "Right answers: " << ra / n * 100 << "% \t Max RA: " << double(maxra) / n * 100 << "(epoch " << maxraepoch << " )" << endl;
			if (ra > maxra) {
				maxra = ra;
				maxraepoch = e;//тут и находим максимальное кол-во угаданных букв и эпоху, когда это произошло
			}
			if (maxraepoch < e - 250) {
				maxra = 0;
			}
		}
		if (nn.SaveWeights()) {//переопределение весов в соответствии с обучением
			cout << "Веса сохранены!";
		}
	}
	else {//если не нужно обучение
		nn.setLayersNotStudy(l, size, "weights.txt");
	}
	fin.close();

	cout << "Начать тест:(1/0) ";
	bool to_start_test = 0;
	cin >> to_start_test;
	char right_res;
	if (to_start_test) {
		fin.open("test.txt");//открываем "текстовое изображение" и отдаем нейронам
		for (int i = 0; i < input_l; i++) {
			fin >> input[i];
		}
		nn.set_input(input);//принимаем значения "текстового изображения"
		result = nn.ForwardFeed(string("show results"));//после "кормления" получаем возможный результат
		cout << "Я считаю, что это буква " << char(result + 65) << "\n\n";
		cout << "А какая это буква на самом деле?...";
		cin >> right_res;
		if (right_res != result + 65) {//если нейросеть не угадала букву, то меняем веса и сохраняем их
			cout << "Хорошо господин, исправляю ошибку!";
			nn.BackPropogation(result, right_res - 65, 0.15);
			nn.SaveWeights();
		}
	}

	return 0;
}